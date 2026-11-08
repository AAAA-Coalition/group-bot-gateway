#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Output filter: what must never reach a human, removed at the gateway.

Pure functions, standard library only. gateway_hook.py applies them to every
response, for every agent and every model, which a per-agent setting does not.

  strip_reasoning_fields   drop `reasoning_content` / `reasoning` / `thinking`
                           from the message: some bot clients paste them back
                           into the text they send.
  cut_derailment           cut at the first chat-template special token
                           (`<|end_of_sentence|>`, full-width bar variants) or
                           at an echoed internal metadata header: what follows
                           is an invented next turn, not an answer.
  strip_think_tags         a leading <think>...</think> block and any orphan
                           `<think>` / `</think>` tag.
  admit_textual_tool_call  a tool call WRITTEN AS TEXT (`<tool_call>`,
                           `<function_call>`, `<|tool_calls_begin|>`) was never
                           executed. The reply is replaced by an honest
                           admission instead of a false "done".

    python output_filter.py --self-test

Environment (all optional):
    FILTER_META_MARKERS     comma-separated names of internal header markers
                            the platform prepends to user messages, e.g.
                            CHAT_META (default: CHAT_META)
    FILTER_TOOL_ADMISSION   text that replaces a textual tool call
    FILTER_DERAIL_APOLOGY   text used when nothing usable precedes a derailment
"""
import os
import re
import sys

REASONING_FIELDS = ("reasoning_content", "reasoning", "thinking")

TOOL_ADMISSION = os.environ.get(
    "FILTER_TOOL_ADMISSION",
    "I could not do that (the tool was not actually called). Could you ask me again, in one sentence?")
DERAIL_APOLOGY = os.environ.get("FILTER_DERAIL_APOLOGY", "Sorry, I lost the thread for a second. What were you saying?")
MIN_KEPT = 20          # below this many characters, what precedes a derailment is template noise

# The vertical bar has TWO forms in chat templates: ASCII `|` and full-width `｜` (U+FF5C).
# Some models emit the second. Recognising only the first let the real case through.
_BAR = "[|｜]"
_SEP = "(?:▁|__|_)"     # U+2581, or how it renders on some terminals
SPECIAL_TOKEN = re.compile(
    r"<\s*" + _BAR + r"?\s*(?:begin|end|start|eot|bos|eos)" + _SEP +
    r"?(?:of" + _SEP + r"?)?(?:sentence|text|turn)?\s*" + _BAR + r"?\s*>", re.I)


def _meta_leak_pattern():
    names = [re.escape(n.strip()) for n in os.environ.get("FILTER_META_MARKERS", "CHAT_META").split(",") if n.strip()]
    return re.compile(r"(?:<!--\s*)?(?:%s)\s*:" % "|".join(names)) if names else None


META_LEAK = _meta_leak_pattern()
THINK_TAG = re.compile(r"<\s*/?\s*think\s*>", re.I)
TEXTUAL_TOOL_CALL = re.compile(r"<\s*tool_call\s*>|<\s*/?\s*function_call\s*>|<\s*\|?tool_calls?_begin", re.I)


def strip_reasoning_fields(message) -> int:
    """Remove reasoning fields from a message (dict or object). Returns how many were removed."""
    n = 0
    for f in REASONING_FIELDS:
        try:
            if isinstance(message, dict):
                if message.get(f):
                    message[f] = None
                    n += 1
            elif getattr(message, f, None):
                setattr(message, f, None)
                n += 1
        except Exception:                                          # noqa: BLE001
            pass
    return n


def cut_derailment(text):
    """(text, True) if a derailment was cut.

    Looks ONLY at special tokens and the metadata marker, never at "looks like code":
    a user may legitimately talk about programming."""
    if not isinstance(text, str):
        return text, False
    hits = [m for m in (SPECIAL_TOKEN.search(text), META_LEAK.search(text) if META_LEAK else None) if m]
    if not hits:
        return text, False
    first = min(hits, key=lambda m: m.start())
    kept = text[:first.start()].strip()
    if len(kept) < MIN_KEPT:
        return DERAIL_APOLOGY, True
    return kept, True


def strip_think_tags(text):
    """Remove a leading <think>...</think> block and any orphan tag. (text, True if changed)."""
    if not isinstance(text, str) or not THINK_TAG.search(text):
        return text, False
    t = re.sub(r"(?is)^\s*<\s*think\s*>.*?<\s*/\s*think\s*>", "", text)
    return THINK_TAG.sub("", t).strip(), True


def admit_textual_tool_call(text):
    """(admission, True) if a tool call written as text was found."""
    if not isinstance(text, str) or not TEXTUAL_TOOL_CALL.search(text):
        return text, False
    return TOOL_ADMISSION, True


def clean_text(text):
    """Apply the three text filters in order. Returns (text, [names of filters that acted])."""
    acted = []
    for name, f in (("derailment", cut_derailment), ("think-tags", strip_think_tags),
                    ("textual-tool-call", admit_textual_tool_call)):
        text, changed = f(text)
        if changed:
            acted.append(name)
    return text, acted


# --------------------------------------------------------------------- witnesses
def self_test(verbose: bool = True) -> int:
    failures = []

    def check(cond, name):
        if verbose:
            print("  %-4s %s" % ("ok" if cond else "FAIL", name))
        if not cond:
            failures.append(name)

    leak = ("Do you want the minutes of Tuesday's meeting?\n\n"
            "<!--CHAT_META:{\"chat_id\":1,\"chat_type\":\"private\",\"user_id\":1}:CHAT_META-->\n"
            "Yes please, send them")
    t, f = cut_derailment(leak)
    check(f and t == "Do you want the minutes of Tuesday's meeting?",
          "1. echoed metadata header + invented user turn -> cut")
    healthy = "we talk about metadata at work, nothing serious"
    check(cut_derailment(healthy) == (healthy, False), "2. a healthy message mentioning metadata -> untouched")
    t, f = cut_derailment("Here is the answer you asked for, in full.<｜end▁of▁sentence｜>User: thanks")
    check(f and t == "Here is the answer you asked for, in full.", "3. full-width special token -> cut there")
    check(cut_derailment("אב<|end_of_text|>junk")[0] == DERAIL_APOLOGY,
          "4. nothing usable before the token -> apology, not noise")
    check(strip_think_tags("</think>") == ("", True)
          and strip_think_tags("<think>abc</think>Hello")[0] == "Hello"
          and strip_think_tags("Hello")[1] is False, "5. orphan </think> removed, healthy text untouched")
    t, f = admit_textual_tool_call("<tool_call>memory_remember text Bob likes </think>I will remember that Bob likes it.")
    check(f and t == TOOL_ADMISSION, "6. tool call written as text -> honest admission")
    check(admit_textual_tool_call("I will remember that you like conferences.")[1] is False,
          "7. a normal sentence about remembering -> untouched")
    m = {"content": "ok", "reasoning_content": "The user is asking..."}
    check(strip_reasoning_fields(m) == 1 and m["reasoning_content"] is None and m["content"] == "ok",
          "8. reasoning field dropped, content kept")
    t, acted = clean_text("<tool_call>send_photo {}</tool_call>")
    check(t == TOOL_ADMISSION and acted == ["textual-tool-call"], "9. clean_text reports which filter acted")
    if verbose:
        print("  %d witness(es), %d failure(s)" % (9, len(failures)))
    return len(failures)


if __name__ == "__main__":
    sys.exit(1 if self_test() else 0)
