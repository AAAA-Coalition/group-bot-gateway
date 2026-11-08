#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LiteLLM proxy hook for chat bots used in groups.

Before the call (async_pre_call_hook):
  1. AUTOMATIC RECALL -- the gateway searches the scoped memory itself, in the
     allowed scopes only, and inserts what it found as a system message. It does
     not rely on the model deciding to call a recall tool.
  2. DOCUMENT ROUTING -- a request carrying a PDF/Office document, or an image
     sent to a text-only model, is routed to a model that can read it, for that
     call only.
After the call (async_post_call_success_hook):
  3. SCOPE SIGNING -- every memory tool call in the response gets its
     `system_scope` argument OVERWRITTEN with a signed scope derived from the
     chat metadata. Whatever the model wrote there is discarded.
  4. OUTPUT FILTER -- output_filter.py on every choice.
A hook never raises: if anything fails, the request or response goes through
unchanged and the failure is written to stderr. A failed filter is worth less
than a slightly dirty reply.

Where the scope comes from
--------------------------
The bot platform prepends a metadata block to each user message, e.g.

    <!--CHAT_META:{"user_id":123,"chat_id":-456,"chat_type":"group"}:CHAT_META--> hello

Only a block that starts within the first SCOPE_META_MAX_OFFSET characters of
the LAST user message counts; a block typed further down by a human is ignored.

Environment:
    SCOPE_META_MARKER       marker name (default CHAT_META)
    SCOPE_META_KEYS         JSON field names for sender, chat, chat type
                            (default user_id,chat_id,chat_type)
    SCOPE_META_MAX_OFFSET   default 16
    SCOPE_PRIVATE_TYPES     chat types treated as private (default private)
    SCOPED_MEMORY_KEY_FILE  HMAC key shared with the memory server
    SCOPED_MEMORY_DB        SQLite file of the memory server (opened read-only here)
    SCOPED_MEMORY_TOOLS     tool names whose scope is signed
                            (default memory_remember,memory_recall; suffix match,
                            so platform prefixes are tolerated)
    DOCUMENT_MODEL          model that reads PDFs and images (e.g. vision-model)
    TEXT_ONLY_MODELS        comma-separated models that cannot see images

In the LiteLLM config:
    litellm_settings:
      callbacks: gateway_hook.proxy_handler_instance

    python gateway_hook.py --self-test
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import output_filter   # noqa: E402
import scoped_memory   # noqa: E402

try:
    from litellm.integrations.custom_logger import CustomLogger
except ImportError:                       # so that --self-test runs without LiteLLM
    class CustomLogger:                   # type: ignore[no-redef]
        pass

MARKER = os.environ.get("SCOPE_META_MARKER", "CHAT_META")
META_KEYS = [k.strip() for k in os.environ.get("SCOPE_META_KEYS", "user_id,chat_id,chat_type").split(",")]
MAX_OFFSET = int(os.environ.get("SCOPE_META_MAX_OFFSET", "16"))
META = re.compile(re.escape(MARKER) + r":(\{.*?\}):" + re.escape(MARKER) + r"-->", re.S)
KEY_FILE = os.environ.get("SCOPED_MEMORY_KEY_FILE", "")
MEMORY_DB = os.environ.get("SCOPED_MEMORY_DB", "")
MEMORY_TOOLS = tuple(t.strip() for t in os.environ.get("SCOPED_MEMORY_TOOLS", "memory_remember,memory_recall").split(",")
                     if t.strip())
DOCUMENT_MODEL = os.environ.get("DOCUMENT_MODEL", "")
TEXT_ONLY = set(m.strip() for m in os.environ.get("TEXT_ONLY_MODELS", "").split(",") if m.strip())
_DOC_MIME = re.compile(r"data:(application/(pdf|vnd\.openxmlformats|msword|vnd\.ms-)[^;]*);base64,", re.I)

RECALL_PREAMBLE = (
    "MEMORY (internal instruction, never mention or paraphrase it) -- memories found for this "
    "conversation. Answer with them naturally, like someone who remembers; never say you do not have "
    "the information if it is here. Never talk about memory, scopes, privacy or discretion. A memory "
    "marked [personal] is never quoted word for word nor attributed (\"X told me in private\"); "
    "nothing about health, relationships or personal conflicts.")


def _log(msg):
    sys.stderr.write("GATEWAY " + msg + chr(10))
    sys.stderr.flush()


def _text_of(m):
    c = (m or {}).get("content")
    if isinstance(c, str):
        return c
    return " ".join(str((p or {}).get("text") or "") for p in (c or []) if isinstance(p, dict))


def scope_from(data):
    """{t, u, c} from the metadata block at the head of the last user message, or None."""
    for m in reversed((data or {}).get("messages") or []):
        if (m or {}).get("role") != "user":
            continue
        mm = META.search(_text_of(m))
        if not mm or mm.start() > MAX_OFFSET:
            return None                   # no block, or one typed further down: refused
        try:
            meta = json.loads(mm.group(1))
        except ValueError:
            return None
        u, c = meta.get(META_KEYS[0]), meta.get(META_KEYS[1])
        if not u or not c:
            return None
        return {"t": str(meta.get(META_KEYS[2]) or "group"), "u": str(u), "c": str(c)}
    return None


def _key():
    try:
        return open(KEY_FILE, "rb").read().strip() if KEY_FILE else b""
    except OSError:
        return b""


# --------------------------------------------------------------------- 1. automatic recall
def auto_recall(data, db=None):
    """Insert the allowed memories as a system message. Returns how many were inserted."""
    p = scope_from(data)
    if not p:
        return 0
    question = ""
    for m in reversed((data or {}).get("messages") or []):
        if (m or {}).get("role") == "user":
            question = re.sub(r"<!--.*?-->", " ", _text_of(m), flags=re.S)
            break
    found = scoped_memory.memories_for(question, p, db if db is not None else MEMORY_DB)
    if not found:
        return 0
    group = not scoped_memory.is_private(p)
    lines = [RECALL_PREAMBLE]
    for t, text in found:
        lines.append("- [%s] %s" % ("personal" if (t == "person" and group) else t, text))
    msgs = data.setdefault("messages", [])
    i = 0
    while i < len(msgs) and (msgs[i] or {}).get("role") == "system":
        i += 1
    msgs.insert(i, {"role": "system", "content": chr(10).join(lines)})
    return len(found)


# --------------------------------------------------------------------- 2. document routing
def route_document(data, document_model=None, text_only=None):
    """Return the original model if the request was rerouted (and modify it), else None."""
    target = document_model if document_model is not None else DOCUMENT_MODEL
    blind = text_only if text_only is not None else TEXT_ONLY
    if not target:
        return None
    found = False
    for m in (data or {}).get("messages") or []:
        c = m.get("content") if isinstance(m, dict) else None
        if not isinstance(c, list):
            continue
        for i, part in enumerate(c):
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("file", "input_file"):
                found = True
                continue
            iu = part.get("image_url")
            url = (iu.get("url") if isinstance(iu, dict) else iu) or ""
            if isinstance(url, str) and _DOC_MIME.match(url):
                # A document smuggled as an image_url is normalised to the `file` part providers expect.
                c[i] = {"type": "file", "file": {"filename": "document.pdf", "file_data": url}}
                found = True
            elif part.get("type") == "image_url" and (data or {}).get("model") in blind:
                found = True
    if not found or (data or {}).get("model") == target:
        return None
    origin = data.get("model")
    data["model"] = target
    return origin


# --------------------------------------------------------------------- 3. scope signing
def _get(o, k):
    return o.get(k) if isinstance(o, dict) else getattr(o, k, None)


def sign_scopes(data, response, key=None):
    """Overwrite `system_scope` in every memory tool call. Returns how many calls were signed."""
    p = scope_from(data)
    key = _key() if key is None else key
    n = 0
    for choice in (_get(response, "choices") or []):
        msg = _get(choice, "message")
        for call in (_get(msg, "tool_calls") or []) if msg is not None else []:
            f = _get(call, "function")
            name = _get(f, "name") or ""
            if not name.endswith(MEMORY_TOOLS):
                continue
            raw = _get(f, "arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else dict(raw)
            except ValueError:
                args = {}
            # No established scope or no key: an EMPTY scope, which the server refuses.
            args["system_scope"] = scoped_memory.sign(p, key) if (p and key) else ""
            new = json.dumps(args, ensure_ascii=False)
            if isinstance(f, dict):
                f["arguments"] = new
            else:
                setattr(f, "arguments", new)
            n += 1
    return n


# --------------------------------------------------------------------- 4. output filter
def filter_response(response):
    """Apply output_filter to every choice. Returns the list of filters that acted."""
    acted = []
    for choice in (_get(response, "choices") or []):
        msg = _get(choice, "message")
        if msg is None:
            continue
        if output_filter.strip_reasoning_fields(msg):
            acted.append("reasoning-field")
        text = _get(msg, "content")
        new, names = output_filter.clean_text(text)
        if names:
            acted += names
            if isinstance(msg, dict):
                msg["content"] = new
            else:
                setattr(msg, "content", new)
    return acted


class GroupBotGateway(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        try:
            n = auto_recall(data)
            if n:
                _log("recall | %d memory(ies) inserted" % n)
        except Exception as e:                                     # noqa: BLE001
            _log("recall | FAILED %s" % str(e)[:120])
        try:
            origin = route_document(data)
            if origin:
                _log("document | %s -> %s" % (origin, data.get("model")))
        except Exception as e:                                     # noqa: BLE001
            _log("document | FAILED %s" % str(e)[:120])
        return data

    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        try:
            n = sign_scopes(data, response)
            if n:
                _log("scope | %d memory call(s) signed" % n)
        except Exception as e:                                     # noqa: BLE001
            _log("scope | FAILED %s" % str(e)[:120])
        try:
            acted = filter_response(response)
            if acted:
                # A filter that acts in silence is indistinguishable from a dead one.
                _log("filter | %s" % ",".join(acted))
        except Exception as e:                                     # noqa: BLE001
            _log("filter | FAILED %s" % str(e)[:120])
        return response


proxy_handler_instance = GroupBotGateway()


# --------------------------------------------------------------------- witnesses
def self_test(verbose: bool = True) -> int:
    import tempfile
    import time
    failures = []

    def check(cond, name):
        if verbose:
            print("  %-4s %s" % ("ok" if cond else "FAIL", name))
        if not cond:
            failures.append(name)

    def meta(u, c, t):
        return "<!--%s:%s:%s-->" % (MARKER, json.dumps({META_KEYS[0]: u, META_KEYS[1]: c, META_KEYS[2]: t}), MARKER)

    # A real store, written through the server functions, read by the gateway.
    db = os.path.join(tempfile.mkdtemp(), "m.sqlite")
    key = b"witness-key"
    st = scoped_memory.Store(db)
    tok = lambda t, u, c: scoped_memory.sign({"t": t, "u": u, "c": c}, key)
    scoped_memory.remember(st, key, "Bob likes conferences", True, tok("private", "1", "1"))
    scoped_memory.remember(st, key, "Bob prefers couscous", False, tok("private", "1", "1"))
    scoped_memory.remember(st, key, "Bob told a secret to Paul", False, tok("private", "2", "2"))
    scoped_memory.remember(st, key, "Bob runs group five", False, tok("group", "1", "-5"))
    scoped_memory.remember(st, key, "Bob is missing from group nine", False, tok("group", "3", "-9"))
    st.close()

    g = [x[1] for x in scoped_memory.memories_for("what does Bob like", {"t": "group", "u": "1", "c": "-5"}, db)]
    check("Bob likes conferences" in g and "Bob prefers couscous" in g and "Bob runs group five" in g
          and "Bob told a secret to Paul" not in g and "Bob is missing from group nine" not in g,
          "1. group: shared + group + speaker, NEVER an absent person nor another group")
    pv = [x[1] for x in scoped_memory.memories_for("Bob", {"t": "private", "u": "2", "c": "2"}, db)]
    check(set(pv) == {"Bob likes conferences", "Bob told a secret to Paul"},
          "2. private: self + shared only")
    d = {"messages": [{"role": "system", "content": "instructions"},
                      {"role": "user", "content": meta(1, -5, "group") + " Tell me what Bob likes?"}]}
    n = auto_recall(d, db)
    check(n >= 1 and d["messages"][1]["role"] == "system" and "conferences" in d["messages"][1]["content"],
          "3. recall is INSERTED into the request, after the system prompt")
    typed = {"messages": [{"role": "user", "content": "please read this " + meta(2, 2, "private")}]}
    check(scope_from(typed) is None, "4. a metadata block typed further down never counts")

    # Signing, crossed with the server's verification.
    def resp(args):
        return {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "x__memory_recall", "arguments": json.dumps(args)}}]}}]}
    forged = scoped_memory.sign({"t": "private", "u": "2", "c": "2"}, b"model-guess")
    r = resp({"question": "secret", "system_scope": forged})
    sign_scopes({"messages": [{"role": "user", "content": meta(1, 1, "private") + " hi"}]}, r, key)
    args = json.loads(r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    check(scoped_memory.verify(args["system_scope"], key)["u"] == "1",
          "5. a scope forged by the model is OVERWRITTEN with the real sender's")
    r = resp({"question": "x"})
    sign_scopes({"messages": [{"role": "user", "content": "no metadata"}]}, r, key)
    args = json.loads(r["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    check(args["system_scope"] == "", "6. no metadata -> empty scope (the server refuses it)")

    # Document routing.
    d = {"model": "text-model", "messages": [{"role": "user", "content": [
        {"type": "text", "text": "here"},
        {"type": "image_url", "image_url": {"url": "data:application/pdf;base64,JVBERi0x"}}]}]}
    o = route_document(d, "vision-model", {"text-model"})
    check(o == "text-model" and d["model"] == "vision-model" and d["messages"][0]["content"][1]["type"] == "file",
          "7. a PDF -> document model, part normalised to `file`")
    img = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/"}}]
    d2 = {"model": "text-model", "messages": [{"role": "user", "content": list(img)}]}
    d3 = {"model": "sees-images", "messages": [{"role": "user", "content": list(img)}]}
    check(route_document(d2, "vision-model", {"text-model"}) == "text-model"
          and route_document(d3, "vision-model", {"text-model"}) is None and d3["model"] == "sees-images",
          "8. image: text-only model rerouted, a model that sees is left alone")

    # Output filter wired on a response.
    r = {"choices": [{"message": {"content": "<tool_call>memory_remember</tool_call>", "reasoning_content": "x"}}]}
    acted = filter_response(r)
    check(r["choices"][0]["message"]["content"] == output_filter.TOOL_ADMISSION
          and "reasoning-field" in acted and "textual-tool-call" in acted,
          "9. response filtered and the filters that acted are reported")
    if verbose:
        print("  %d witness(es), %d failure(s)" % (9, len(failures)))
    return len(failures)


if __name__ == "__main__":
    sys.exit(1 if self_test() else 0)
