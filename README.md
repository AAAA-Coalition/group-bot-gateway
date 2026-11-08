# group-bot-gateway

A [LiteLLM](https://github.com/BerriAI/litellm) proxy hook and a small MCP
server for chat bots that live in **groups**, not only in private chats.
Python, standard library only (plus `mcp` for the server), with planted
witnesses and mutation checks.

| part | what it does |
|---|---|
| `scoped_memory.py` | memory with three scopes — **person**, **group**, **shared** — chosen by the system, never by the model; MCP server with two tools |
| `gateway_hook.py` | **automatic recall** before the call, **scope signing** of memory tool calls after it, **document routing** to a model that can read the file |
| `output_filter.py` | removes what must never reach a human: reasoning fields, orphan `</think>`, chat-template special tokens, an echoed internal header — and replaces a **tool call written as text** with an honest admission |

Everything here was written the day it cost us something, and proved on a real
group chat before publication.

---

## 1. Scoped memory

**What it cost.** Our bot platform kept its long-term memory under a single
user id for the whole bot. Everyone who talked to it shared the same memories:
what one person said in private could surface for anyone else.

**The rule.** The scope comes from the chat, never from the model.

| conversation | reads | writes |
|---|---|---|
| private chat with A | person(A) + shared | person(A) |
| group G | group(G) + shared + person(X) for every X **present** in G | group(G) |

*Present* means "has already spoken in G", recorded by the server from signed
scopes, never declared by the model. A fact goes to *shared* only when the
model flags it as not private. In a group, a personal memory comes back with an
instruction: use it to deepen the conversation, never quote it, nothing
about health, relationships or personal conflicts.

**How the model is kept out of it.** Both tools take a `system_scope`
argument. The gateway **overwrites** it on every memory tool call with a token
it signs (HMAC-SHA256) from the chat metadata the platform puts at the head of
the user message. The server refuses a missing, foreign or expired signature.
After every full-text search, a **membership check** drops any row whose scope
is not in the allowed set: the search filter alone is not trusted to isolate.
Scopes are `(type, key)` pairs, never concatenated strings.

## 2. Automatic recall

**What it cost.** A fact was in the shared memory. In a group, someone asked
the bot *"Tell me what X likes?"*. The model called **no tool** and answered
*"I don't have any information"*. A long system prompt telling it to recall
first did not make it call the tool.

So the gateway searches the memory itself, in the allowed scopes only, and
inserts what it found as a system message after the existing system prompt. The
model no longer has to decide to remember.

**A second lesson from the same afternoon.** The first version tagged each
personal memory with *"personal: use discreetly"*. The bot **repeated that
note to the group**. The instructions are now stated once, at the top, as rules
never to be spoken.

## 3. Document routing

A PDF sent to a bot whose model cannot read PDFs came back as *"I had no access
to the content of the PDF"*. An image sent to a text-only model came back as
*"do you want me to load it?"*. The pre-call hook routes a request that carries
a document (or an image for a model listed as text-only) to `DOCUMENT_MODEL`,
**for that call only**, and normalises a document smuggled as an `image_url`
into the `file` part providers expect.

## 4. Output filter

Each filter matches a failure we shipped to a real person:

| filter | the real case |
|---|---|
| reasoning fields | the agents' reasoning (*"The user is asking…"*) appeared in clear in a group of twenty people. The provider response was clean: the reasoning sat in `reasoning_content`, and the bot client pasted it back. Dropping the **field** fixes it without any text heuristic. |
| derailment | a model can continue past its own reply: it echoes the platform's **internal metadata header**, then writes an **invented reply in the user's name**. The filter cuts at the first chat-template special token (both `|` and full-width `｜` forms) or at the echoed header. |
| `</think>` | an orphan `</think>` sent alone as the whole message. |
| textual tool call | `<tool_call>memory_remember … </think> I will remember that…` — the tool was **never called**, nothing was stored, and the bot said it was done. The reply becomes *"I could not do that (the tool was not actually called). Could you ask me again?"* |

The last one matters most. Stripping the markup alone would have left a false
*"I will remember that"*. **A filter that hides a failure is worse than the
failure.**

**Fix the cause too.** On OpenRouter, textual tool calls mostly come from
providers that handle tool calling badly;
[Exacto](https://openrouter.ai/docs/guides/routing/model-variants/exacto) and
[Auto Exacto](https://openrouter.ai/docs/guides/routing/auto-exacto) route
tool-calling requests to providers with better tool-call quality. This filter is
the net under that, not a substitute for it.

---

## The witnesses, and the mutations

```
$ python test_all.py
== scoped_memory
  ok   1. B, in private, does NOT see A's personal memory
  ok   3. group where A has spoken: memory returned WITH the discretion note
  ok   4. group where A never spoke: nothing
  ok   6. a scope signed with another key is refused
  …
== gateway_hook
  ok   1. group: shared + group + speaker, NEVER an absent person nor another group
  ok   4. a metadata block typed further down never counts
  ok   5. a scope forged by the model is OVERWRITTEN with the real sender's
  …
== mutations: each broken guard must be caught
  ok   membership check disabled -> caught
  ok   derailment detection disabled -> caught
  ok   metadata accepted anywhere in the message -> caught
  ok   scope expiry disabled -> caught

TOTAL: 0 witness failure(s), 0 mutation(s) not caught
```

27 witnesses, plus 3 more on real LiteLLM `ModelResponse` objects when LiteLLM
is installed. **A green that cannot turn red proves nothing**, so `test_all.py`
then breaks each guard on purpose and requires a witness to fail.

`smoke_test_server.py` starts the real MCP server on a free port and plays the
scenario over HTTP with a scope signed by the gateway code. It passes with the
MCP Python SDK **1.x and 2.x** (2.x renamed `FastMCP` to `MCPServer`; the
server handles both).

The production version this code was extracted from passed the same scenario
with real messages on a Telegram bot — a private chat and two groups — on
8 November 2026. The extract was then made generic (names, paths and markers
configurable) and re-tested with the witnesses above, the smoke test, and a
local LiteLLM proxy (1.108.3) with the hook loaded as a callback: a mock model
answering `<tool_call>…</tool_call> I will remember that.` reached the client as
the honest admission.

---

## Install

```bash
git clone https://github.com/AAAA-Coalition/group-bot-gateway
cd group-bot-gateway
python test_all.py                 # no dependency needed
pip install mcp                    # for the memory server only
```

1. **Key.** `umask 077; python3 -c "import os;print(os.urandom(32).hex())" > /etc/bot-memory/scope.key`
2. **Memory server.** See `examples/scoped-memory.service`
   (`SCOPED_MEMORY_PORT`, `SCOPED_MEMORY_DB`, `SCOPED_MEMORY_KEY_FILE`). It
   listens on `127.0.0.1` only.
3. **Gateway.** Copy the three `.py` files next to your LiteLLM config and add
   `callbacks: gateway_hook.proxy_handler_instance` (see
   `examples/litellm_config.example.yaml`).
4. **Bot.** Attach the MCP server to the bot and tell it, in its prompt, to use
   `memory_remember` / `memory_recall` and never to fill `system_scope`.

**The metadata block.** The hook reads the scope from a block your bot platform
prepends to each user message:

```
<!--CHAT_META:{"user_id":123,"chat_id":-456,"chat_type":"group"}:CHAT_META--> hello
```

Marker and field names are configurable (`SCOPE_META_MARKER`,
`SCOPE_META_KEYS`), so a platform that already emits its own block can be used
as is. If yours emits none, add one in the connector that receives the chat
messages: that is the only place that knows who is speaking.

| variable | used by | meaning |
|---|---|---|
| `SCOPE_META_MARKER`, `SCOPE_META_KEYS`, `SCOPE_META_MAX_OFFSET` | hook | where the scope is read |
| `SCOPE_PRIVATE_TYPES` | both | chat types treated as private (default `private`) |
| `SCOPED_MEMORY_KEY_FILE` | both | HMAC key file |
| `SCOPED_MEMORY_DB` | both | SQLite file (the hook opens it read-only) |
| `SCOPED_MEMORY_TOOLS` | hook | tool names to sign (suffix match) |
| `DOCUMENT_MODEL`, `TEXT_ONLY_MODELS` | hook | document routing |
| `FILTER_META_MARKERS`, `FILTER_TOOL_ADMISSION`, `FILTER_DERAIL_APOLOGY` | filter | header names and the two replacement texts |

Python 3.8+, SQLite with FTS5 (bundled with CPython on all major platforms).

---

## Limits, stated

- **The scope is only as trustworthy as the metadata block.** The hook trusts
  the block at the head of the last user message. If a client can reach the
  gateway **without** going through a connector that writes that block, a user
  could forge one. Put the gateway behind the connector, give the bot its own
  key, and keep the memory server on localhost.
- **Presence is recorded by the server**, when a memory tool is called in a
  group on behalf of a speaker. Automatic recall reads presence but does not
  write it. Someone who speaks in a group without ever triggering a memory call
  is not yet "present" there.
- **Recall is keyword search** (SQLite FTS5, words of 3+ letters, a small
  English/French stop-word list), not embeddings. It finds *"Bob likes"* from
  *"what does Bob like"*; it will not find a synonym.
- **The discretion rule is an instruction to the model**, not a guarantee. The
  isolation between scopes is enforced in code; what the model does with a
  memory it was allowed to see is not.
- **The textual-tool-call filter replaces the whole reply** when it sees
  `<tool_call>`-style markup — including a reply that legitimately explains that
  syntax to a developer. We accepted that trade-off for chat bots; change it for
  a coding assistant.
- **The filter never reconstructs** a tool call written as text; it admits the
  failure instead.

---

## Where this sits

We checked before writing this, on 8 November 2026:

| | what it does | what it does not do |
|---|---|---|
| [LiteLLM `/v1/memory`](https://docs.litellm.ai/docs/proxy/memory) | key-value memory store on the proxy, per-user keys by prefix | no search, no injection into requests, no group composition |
| [Mem0](https://docs.mem0.ai/platform/features/entity-scoped-memory) | scoped memories by `user_id` / `agent_id` / `app_id` / `run_id`, from the client SDK | the **caller** picks the scope; no "group + present members" read set |
| [Supermemory Memory Router](https://ai-sdk.dev/providers/community-providers/supermemory) | a hosted proxy that injects memory into LLM calls | hosted; scope is the namespace/conversation id the caller sends |
| [LiteLLM reasoning content](https://docs.litellm.ai/docs/reasoning_content) | standardises `reasoning_content` | raw `<think>` tags are not separated for every provider ([#24253](https://github.com/BerriAI/litellm/issues/24253)); nothing for textual tool calls |
| [OpenRouter PDF inputs](https://openrouter.ai/docs/guides/overview/multimodal/pdfs) | `file-parser` plugin turns a PDF into text for any model | OpenRouter only; we also route for self-hosted models |

So memory injected at a proxy **exists** (hosted), and scoped memory **exists**
(client-side). What this adds is the combination for group bots behind a
self-hosted LiteLLM: the scope **derived from the channel and signed** at the
gateway so the model can never choose it, the **group read set** composed from
the participants actually present, a **membership check after search**, and an
output filter that turns a silent tool-call failure into an admission.

Related, from the same coalition:
[conversational-guards](https://github.com/AAAA-Coalition/conversational-guards)
(behavioural guards for replies, including foreign-script derailment).

---

## Licence

MIT. See [LICENSE](LICENSE).
