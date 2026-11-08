#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end witness of the MCP server: a real process, real HTTP, a scope signed by the gateway code.

Starts `scoped_memory.py --serve` on a free local port with a throw-away key and
database, then: A writes in private -> B in private sees nothing -> in a group
where A has spoken the memory comes back WITH the discretion note -> in a group
where A never spoke, nothing -> an unsigned call is refused.

Needs the MCP Python SDK (`pip install mcp`); tested with SDK 1.x and 2.x APIs.
Exits 0 when every case is as expected, 77 (skipped) when `mcp` is not installed.

    python smoke_test_server.py
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scoped_memory  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def call(port, tool, args):
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}

    def post(body, sid=None):
        h = dict(headers)
        if sid:
            h["Mcp-Session-Id"] = sid
        r = urllib.request.Request("http://127.0.0.1:%d/mcp" % port, data=json.dumps(body).encode(), headers=h)
        with urllib.request.urlopen(r, timeout=20) as f:
            return f.headers.get("mcp-session-id"), f.read().decode("utf-8", "replace")
    sid, _ = post({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
        "protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "smoke", "version": "1"}}})
    post({"jsonrpc": "2.0", "method": "notifications/initialized"}, sid)
    _, raw = post({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": tool, "arguments": args}}, sid)
    lines = [l for l in raw.splitlines() if l.startswith("data:")]
    msg = json.loads(lines[-1][5:] if lines else raw)
    res = msg["result"]
    if res.get("structuredContent"):
        sc = res["structuredContent"]
        return sc.get("result", sc) if isinstance(sc, dict) else sc
    return json.loads(res["content"][0]["text"])


def main():
    try:
        import mcp  # noqa: F401
    except ImportError:
        print("  SKIPPED: the MCP Python SDK is not installed (pip install mcp)")
        return 77
    d = tempfile.mkdtemp()
    key = os.urandom(32).hex().encode()
    key_file = os.path.join(d, "scope.key")
    with open(key_file, "wb") as f:
        f.write(key)
    port = free_port()
    env = dict(os.environ, SCOPED_MEMORY_PORT=str(port), SCOPED_MEMORY_DB=os.path.join(d, "m.sqlite"),
               SCOPED_MEMORY_KEY_FILE=key_file)
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "scoped_memory.py"), "--serve"], env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    failures = []
    try:
        for _ in range(60):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                time.sleep(0.5)
        scope = lambda t, u, c: scoped_memory.sign({"t": t, "u": u, "c": c}, key)

        def check(cond, name):
            print("  %-4s %s" % ("ok" if cond else "FAIL", name))
            if not cond:
                failures.append(name)
        r = call(port, "memory_remember", {"text": "Alice is preparing a valorisation workshop",
                                           "system_scope": scope("private", "900001", "900001")})
        check(r.get("ok") is True, "1. A writes in private over HTTP")
        r = call(port, "memory_recall", {"question": "valorisation workshop",
                                         "system_scope": scope("private", "900002", "900002")})
        check(r.get("ok") is True and r.get("count") == 0, "2. B in private sees nothing")
        call(port, "memory_remember", {"text": "hello group", "system_scope": scope("group", "900001", "-900101")})
        r = call(port, "memory_recall", {"question": "valorisation workshop",
                                         "system_scope": scope("group", "900002", "-900101")})
        check(r.get("count") == 1 and "note" in r["memories"][0], "3. group where A spoke: returned with the note")
        r = call(port, "memory_recall", {"question": "valorisation workshop",
                                         "system_scope": scope("group", "900002", "-900102")})
        check(r.get("count") == 0, "4. group where A never spoke: nothing")
        r = call(port, "memory_recall", {"question": "valorisation workshop", "system_scope": ""})
        check(r.get("ok") is False, "5. unsigned call refused")
    except Exception as e:                                         # noqa: BLE001
        failures.append("server unreachable or bad answer: %s" % str(e)[:200])
        print("  FAIL", failures[-1])
    finally:
        proc.terminate()
        try:
            err = proc.communicate(timeout=10)[1].decode("utf-8", "replace")
        except Exception:                                          # noqa: BLE001
            proc.kill()
            err = ""
        if failures and err:
            print(err[-1500:])
    print("  5 case(s), %d failure(s)" % len(failures))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
