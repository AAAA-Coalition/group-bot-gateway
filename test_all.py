#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run every witness, then BREAK each guard on purpose and check that a witness turns red.

A green that cannot turn red proves nothing.

    python test_all.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import gateway_hook    # noqa: E402
import output_filter   # noqa: E402
import scoped_memory   # noqa: E402


def section(title):
    print(chr(10) + "== " + title)


def litellm_objects():
    """Same hooks on real LiteLLM ModelResponse objects (attributes, not dicts). Skipped without LiteLLM."""
    try:
        from litellm import ModelResponse
        from litellm.integrations.custom_logger import CustomLogger
    except ImportError:
        print("  skip LiteLLM not installed")
        return 0
    import asyncio
    import json
    fails = 0

    def check(cond, name):
        nonlocal fails
        print("  %-4s %s" % ("ok" if cond else "FAIL", name))
        fails += 0 if cond else 1
    check(issubclass(gateway_hook.GroupBotGateway, CustomLogger), "1. the hook is a real LiteLLM CustomLogger")
    r = ModelResponse(choices=[{"index": 0, "finish_reason": "stop", "message": {
        "role": "assistant", "content": "<tool_call>memory_remember x</tool_call>",
        "reasoning_content": "The user is asking"}}])
    out = asyncio.run(gateway_hook.proxy_handler_instance.async_post_call_success_hook({"messages": []}, None, r))
    m = out.choices[0].message
    check(m.content == output_filter.TOOL_ADMISSION and not getattr(m, "reasoning_content", None),
          "2. post-call hook filters a ModelResponse")
    r2 = ModelResponse(choices=[{"index": 0, "finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [{"id": "1", "type": "function", "function": {
            "name": "memory_recall", "arguments": json.dumps({"question": "q", "system_scope": "forged"})}}]}}])
    head = "<!--%s:%s:%s-->" % (gateway_hook.MARKER, json.dumps({gateway_hook.META_KEYS[0]: 1, gateway_hook.META_KEYS[1]: 1,
                                                               gateway_hook.META_KEYS[2]: "private"}), gateway_hook.MARKER)
    gateway_hook.sign_scopes({"messages": [{"role": "user", "content": head + " hi"}]}, r2, b"k")
    a = json.loads(r2.choices[0].message.tool_calls[0].function.arguments)
    check(scoped_memory.verify(a["system_scope"], b"k")["u"] == "1", "3. scope signed on a ModelResponse tool call")
    return fails


def main():
    total = 0
    section("scoped_memory")
    total += scoped_memory.self_test()
    section("output_filter")
    total += output_filter.self_test()
    section("gateway_hook")
    total += gateway_hook.self_test()

    section("LiteLLM response objects (optional)")
    total += litellm_objects()

    section("mutations: each broken guard must be caught")
    mutations = 0

    def mutation(name, patch, restore, run):
        nonlocal mutations
        patch()
        try:
            caught = run(verbose=False) > 0
        finally:
            restore()
        print("  %-4s %s" % ("ok" if caught else "FAIL", name + (" -> caught" if caught else " -> NOT caught")))
        if not caught:
            mutations += 1

    orig = scoped_memory.membership_ok
    mutation("membership check disabled",
             lambda: setattr(scoped_memory, "membership_ok", lambda *a: True),
             lambda: setattr(scoped_memory, "membership_ok", orig), gateway_hook.self_test)
    orig_tok, orig_meta = output_filter.SPECIAL_TOKEN, output_filter.META_LEAK
    never = output_filter.re.compile(r"(?!x)x")

    def patch_filter():
        output_filter.SPECIAL_TOKEN = never
        output_filter.META_LEAK = never

    def restore_filter():
        output_filter.SPECIAL_TOKEN, output_filter.META_LEAK = orig_tok, orig_meta
    mutation("derailment detection disabled", patch_filter, restore_filter, output_filter.self_test)
    orig_off = gateway_hook.MAX_OFFSET
    mutation("metadata accepted anywhere in the message",
             lambda: setattr(gateway_hook, "MAX_OFFSET", 10 ** 6),
             lambda: setattr(gateway_hook, "MAX_OFFSET", orig_off), gateway_hook.self_test)
    orig_win = scoped_memory.SCOPE_WINDOW_S
    mutation("scope expiry disabled",
             lambda: setattr(scoped_memory, "SCOPE_WINDOW_S", 10 ** 9),
             lambda: setattr(scoped_memory, "SCOPE_WINDOW_S", orig_win), scoped_memory.self_test)

    print(chr(10) + "TOTAL: %d witness failure(s), %d mutation(s) not caught" % (total, mutations))
    return 1 if (total or mutations) else 0


if __name__ == "__main__":
    sys.exit(main())
