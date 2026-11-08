#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Three-scope memory for chat bots: person, group, shared. The SYSTEM picks the scope.

Why this exists
---------------
Many bot platforms keep long-term memory under one user id per bot. Every person
who talks to the bot then shares the same memories: what one person said in a
private chat can resurface for someone else.

The rule that makes it safe: the scope never comes from the model
-----------------------------------------------------------------
Each tool takes a `system_scope` argument. The model never fills it: the gateway
hook (gateway_hook.py) OVERWRITES it on every memory tool call with a token it
signs (HMAC-SHA256) from the chat metadata the bot platform puts at the head of
the user message (sender id, chat id, chat type). A call without a valid, recent
signature is refused. A confused or manipulated model can therefore neither read
nor write outside its scope. Scopes are (type, key) PAIRS, never concatenated
strings, so two keys cannot collide.

Rules
-----
  private chat with A : reads person(A) + shared        ; writes person(A)
  group G             : reads group(G) + shared + person(X) for every X PRESENT
                        in G (present = has already spoken in G, recorded by
                        this module from signed scopes, never declared by the
                        model)                            ; writes group(G)
  shared              : written only when the model flags a fact as not private;
                        a newer fact never deletes an older one.
In a group, a personal memory is returned with an instruction: use it to deepen
the conversation, never quote it, nothing about health, relationships or
personal conflicts.
After every search a MEMBERSHIP CHECK drops any row whose scope is not allowed:
the full-text filter alone is not trusted to enforce isolation.

    python scoped_memory.py --self-test        # witnesses, no network, no dependency
    python scoped_memory.py --serve            # MCP server (needs `pip install mcp`)

Environment for --serve:
    SCOPED_MEMORY_PORT      port to listen on (127.0.0.1 only), e.g. 8765
    SCOPED_MEMORY_DB        path of the SQLite file, e.g. /var/lib/bot-memory/memory.sqlite
    SCOPED_MEMORY_KEY_FILE  file holding the HMAC key shared with the gateway hook

Standard library only for everything except --serve. Python 3.8+, SQLite with FTS5.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sqlite3
import sys
import time

SCOPE_WINDOW_S = 900       # a signed scope is valid for 15 minutes
PRIVATE_TYPES = set(t.strip() for t in os.environ.get("SCOPE_PRIVATE_TYPES", "private").split(",") if t.strip())
STOP_WORDS = set(("the and what that this with from have your about tell like likes does did "
                  "est les des une pour que qui dans sur avec pas mais").split())
PERSONAL_NOTE = ("PERSONAL memory of a participant: use it to deepen the conversation, never quote it "
                 "word for word, nothing about health, relationships or personal conflicts")


# --------------------------------------------------------------------- scope
def sign(scope: dict, key: bytes, now: float | None = None) -> str:
    """Signed token for a scope {t: chat type, u: sender id, c: chat id}."""
    body = dict(scope, ts=now if now is not None else time.time())
    raw = base64.urlsafe_b64encode(json.dumps(body, sort_keys=True).encode()).decode()
    return raw + "." + hmac.new(key, raw.encode(), hashlib.sha256).hexdigest()


def verify(token: str, key: bytes, now: float | None = None) -> dict:
    """Return the scope if the signature is valid and recent; raise ValueError otherwise."""
    try:
        raw, sig = (token or "").rsplit(".", 1)
    except ValueError:
        raise ValueError("missing or malformed scope -- the call must go through the gateway")
    expected = hmac.new(key, raw.encode(), hashlib.sha256).hexdigest()
    if not key or not hmac.compare_digest(expected, sig):
        raise ValueError("invalid scope signature")
    p = json.loads(base64.urlsafe_b64decode(raw.encode()).decode())
    if abs((now or time.time()) - float(p.get("ts", 0))) > SCOPE_WINDOW_S:
        raise ValueError("expired scope")
    if not str(p.get("u") or "") or not str(p.get("c") or ""):
        raise ValueError("scope without sender or chat")
    return p


def is_private(p: dict) -> bool:
    return str(p.get("t")) in PRIVATE_TYPES


def write_scope(p: dict) -> tuple:
    return ("person", str(p["u"])) if is_private(p) else ("group", str(p["c"]))


def read_scopes(p: dict, present: list) -> set:
    """The ONLY place that decides what a conversation may read."""
    if is_private(p):
        return {("person", str(p["u"])), ("shared", "")}
    s = {("group", str(p["c"])), ("shared", ""), ("person", str(p["u"]))}
    s |= {("person", str(u)) for u in present}
    return s


def keywords(text: str) -> list:
    words = "".join(ch if ch.isalnum() else " " for ch in (text or "").lower()).split()
    return [w for w in words if len(w) >= 3 and w not in STOP_WORDS][:12]


# --------------------------------------------------------------------- storage
SCHEMA = """
create table if not exists memories (
  id integer primary key,
  scope_type text not null check (scope_type in ('person','group','shared')),
  scope_key text not null, text text not null,
  author text not null, chat text not null, created_at real not null);
create virtual table if not exists memories_fts using fts5(
  text, content='memories', content_rowid='id');
create trigger if not exists memories_ai after insert on memories begin
  insert into memories_fts(rowid, text) values (new.id, new.text); end;
create table if not exists presence (
  chat text not null, author text not null, seen_at real not null,
  primary key (chat, author));
"""


class Store:
    def __init__(self, path: str, read_only: bool = False, any_thread: bool = False):
        if read_only:
            self.c = sqlite3.connect("file:%s?mode=ro" % path, uri=True)
        else:
            # any_thread: the MCP SDK may call a tool from a worker thread.
            self.c = sqlite3.connect(path, check_same_thread=not any_thread)
            self.c.executescript(SCHEMA)

    def close(self):
        self.c.close()

    def note_presence(self, p: dict):
        if not is_private(p):
            self.c.execute("insert or replace into presence values (?,?,?)",
                           (str(p["c"]), str(p["u"]), time.time()))
            self.c.commit()

    def present(self, chat: str) -> list:
        return [r[0] for r in self.c.execute("select author from presence where chat = ?", (str(chat),))]

    def add(self, scope: tuple, text: str, p: dict) -> int:
        cur = self.c.execute(
            "insert into memories (scope_type, scope_key, text, author, chat, created_at) values (?,?,?,?,?,?)",
            (scope[0], scope[1], text, str(p["u"]), str(p["c"]), time.time()))
        self.c.commit()
        return cur.lastrowid

    def search(self, question: str, allowed: set, limit: int = 12) -> list:
        words = keywords(question)
        if not words:
            return []
        rows = self.c.execute(
            "select m.id, m.scope_type, m.scope_key, m.text, m.created_at "
            "from memories_fts f join memories m on m.id = f.rowid "
            "where memories_fts match ? order by m.created_at desc limit 200",
            (" OR ".join('"%s"' % w for w in words),)).fetchall()
        # MEMBERSHIP CHECK, after the search: nothing outside the allowed scopes.
        seen, out = set(), []
        for row in rows:
            if membership_ok(row[1], row[2], allowed) and row[3] not in seen:
                seen.add(row[3])
                out.append(row)
        return out[:limit]


def membership_ok(scope_type: str, scope_key: str, allowed: set) -> bool:
    return (scope_type, scope_key if scope_type != "shared" else "") in allowed


# --------------------------------------------------------------------- tools
def remember(store: Store, key: bytes, text: str, also_shared: bool, system_scope: str) -> dict:
    try:
        p = verify(system_scope, key)
    except ValueError as e:
        return {"ok": False, "refused": str(e)}
    text = (text or "").strip()
    if not text:
        return {"ok": False, "refused": "empty text"}
    store.note_presence(p)
    store.add(write_scope(p), text, p)
    if also_shared:
        store.add(("shared", ""), text, p)
    return {"ok": True, "stored_in": [write_scope(p)[0]] + (["shared"] if also_shared else [])}


def recall(store: Store, key: bytes, question: str, system_scope: str) -> dict:
    try:
        p = verify(system_scope, key)
    except ValueError as e:
        return {"ok": False, "refused": str(e)}
    store.note_presence(p)
    allowed = read_scopes(p, [] if is_private(p) else store.present(p["c"]))
    out = []
    for _id, t, _k, text, when in store.search(question, allowed):
        r = {"scope": t, "memory": text, "date": time.strftime("%Y-%m-%d", time.localtime(when))}
        if t == "person" and not is_private(p):
            r["note"] = PERSONAL_NOTE
        out.append(r)
    return {"ok": True, "memories": out, "count": len(out)}


def memories_for(question: str, p: dict, db_path: str, limit: int = 6) -> list:
    """Read-only lookup used by the gateway's automatic recall: [(scope_type, text)]."""
    if not db_path or not os.path.isfile(db_path) or not keywords(question):
        return []
    store = Store(db_path, read_only=True)
    try:
        allowed = read_scopes(p, [] if is_private(p) else store.present(p["c"]))
        return [(t, text) for _id, t, _k, text, _w in store.search(question, allowed, limit)]
    finally:
        store.close()


# --------------------------------------------------------------------- witnesses
def self_test(verbose: bool = True) -> int:
    import tempfile
    failures = []
    key = b"witness-key"
    store = Store(os.path.join(tempfile.mkdtemp(), "t.sqlite"))

    def tok(t, u, c, age=0.0, k=key):
        return sign({"t": t, "u": u, "c": c}, k, time.time() - age)

    def check(cond, name):
        if verbose:
            print("  %-4s %s" % ("ok" if cond else "FAIL", name))
        if not cond:
            failures.append(name)

    A, B = tok("private", "A", "A"), tok("private", "B", "B")
    remember(store, key, "Alice is preparing a workshop on research valorisation", False, A)
    check(recall(store, key, "workshop valorisation", B)["count"] == 0,
          "1. B, in private, does NOT see A's personal memory")
    check(recall(store, key, "workshop valorisation", A)["count"] == 1,
          "2. A finds her own memory in private")
    remember(store, key, "hello everyone", False, tok("supergroup", "A", "G1"))
    r = recall(store, key, "workshop valorisation", tok("supergroup", "B", "G1"))
    check(r["count"] == 1 and "note" in r["memories"][0],
          "3. group where A has spoken: memory returned WITH the discretion note")
    check(recall(store, key, "workshop valorisation", tok("group", "B", "G2"))["count"] == 0,
          "4. group where A never spoke: nothing")
    remember(store, key, "The network targets call ALPHA-42", True, A)
    check(recall(store, key, "network ALPHA-42", B)["count"] >= 1,
          "5. a shared fact written by A is readable by B")
    check(recall(store, key, "workshop", tok("private", "A", "A", k=b"other-key")).get("ok") is False,
          "6. a scope signed with another key is refused")
    check(recall(store, key, "workshop", "").get("ok") is False,
          "7. a call without a scope is refused")
    check(recall(store, key, "workshop", tok("private", "A", "A", age=3600)).get("ok") is False,
          "8. an expired scope is refused")
    check(not membership_ok("person", "A", read_scopes({"t": "group", "u": "B", "c": "G2"}, ["B"])),
          "9. membership check rejects a person scope that is not present")
    store.close()
    if verbose:
        print("  %d witness(es), %d failure(s)" % (9, len(failures)))
    return len(failures)


def _server(port: int):
    """An MCP server object and its run() callable, for the MCP Python SDK 2.x or 1.x.

    SDK 2.x renamed FastMCP to MCPServer and moved host/port from the constructor to run()."""
    try:
        from mcp.server.mcpserver import MCPServer                   # SDK 2.x
        srv = MCPServer("scoped-memory")
        return srv, lambda: srv.run(transport="streamable-http", host="127.0.0.1", port=port)
    except ImportError:
        from mcp.server.fastmcp import FastMCP                       # SDK 1.x
        srv = FastMCP("scoped-memory", host="127.0.0.1", port=port)
        return srv, lambda: srv.run(transport="streamable-http")


def serve():
    port = int(os.environ.get("SCOPED_MEMORY_PORT", "0") or 0)
    db = os.environ.get("SCOPED_MEMORY_DB", "")
    key_file = os.environ.get("SCOPED_MEMORY_KEY_FILE", "")
    if not port or not db or not key_file:
        sys.exit("SCOPED_MEMORY_PORT, SCOPED_MEMORY_DB and SCOPED_MEMORY_KEY_FILE are required")
    key = open(key_file, "rb").read().strip()
    if not key:
        sys.exit("empty key file: refusing to start (every scope would be refused anyway)")
    store = Store(db, any_thread=True)
    mcp, run = _server(port)

    @mcp.tool()
    def memory_remember(text: str, also_shared: bool = False, system_scope: str = "") -> dict:
        """Remember a useful fact for later. It is stored AUTOMATICALLY in the memory of the person
        (private chat) or of the group (group chat): you do not choose where. Set also_shared=true
        only for a NON-PRIVATE fact useful to everyone (a project, a decision, a public fact).
        Never fill system_scope: the system does."""
        return remember(store, key, text, also_shared, system_scope)

    @mcp.tool()
    def memory_recall(question: str, system_scope: str = "") -> dict:
        """Find what is already known: what this person told you, what was said in this group, the
        shared knowledge -- and, in a group, what the participants present told you in private (use it
        to deepen the conversation, NEVER quote it). Never fill system_scope."""
        return recall(store, key, question, system_scope)

    run()


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        sys.exit(1 if self_test() else 0)
    if "--serve" in sys.argv:
        serve()
    else:
        print(__doc__)
