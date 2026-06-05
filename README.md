# AI Adapter

A Rust proxy that exposes an **OpenAI Responses API** interface, translating to various upstream backends (DeepSeek, OpenAI, Anthropic, XiaomiMimo) with full streaming SSE support. Built for Codex CLI and similar tools hardcoded to OpenAI.

```
Codex / Client          AI Adapter :9090          Upstream
─────────────          ─────────────────          ────────
/v1/responses   ──▶    translate/            ──▶  DeepSeek Chat
(Responses API)        deepseek/chat.rs           (Plan A)

/v1/responses   ──▶    translate/            ──▶  DeepSeek Anthropic
(Responses API)        deepseek/anthropic.rs      (Plan B)

/v1/responses   ──▶    translate/            ──▶  OpenAI Chat
(Responses API)        openai/chat.rs

/v1/responses   ──▶    translate/            ──▶  Anthropic
(Responses API)        anthropic/convert.rs

/v1/responses   ──▶    translate/            ──▶  XiaomiMimo Chat
(Responses API)        xiaomimimo/chat.rs         (+ Anthropic)
```

## Features

- **Multi-provider**: Configure multiple upstream providers in one YAML config, switch by model name per request
- **Vendor isolation**: `src/translate/{deepseek,openai,anthropic,xiaomimimo}/` — each vendor gets its own module
- **Bidirectional streaming**: Full SSE event translation (Chat ↔ Responses, Anthropic ↔ Responses)
- **DeepSeek Plan A/B**: Chat (thinking disabled) and Anthropic (tool_use merging) protocols
- **Reasoning cache**: SQLite-backed reasoning_content store for multi-turn thinking compliance
- **State management**: SQLite for sessions, reasoning, and request logging; auto-migrates from legacy redb
- **Tool calling**: Full function call streaming, flat & nested tool format support
- **Compaction**: `POST /v1/responses/compact` for Codex CLI context management
- **Config-driven**: YAML config files with `providers` and `modelAliases`, plus CLI/env overrides
- **Auto-detect format**: `format` is optional — inferred from `base_url` (anthropic/openai-chat/responses)
- **HTTP/2 upstream**: Transparent ALPN negotiation for supported backends
- **Health & models API**: `/health` with uptime info, `/__/models` listing configured aliases
- **Error dumps**: Saves failed exchanges with auth redaction
- **Structured logging**: Human-readable to stderr, JSON to daily-rotated files

## Quick Start

```bash
# Build
cargo build

# Run with a config file
./target/debug/ai-adapter -c ai-adapter.example.yaml
```

### Codex CLI Setup

Configure Codex to use ai-adapter by editing `~/.codex/config.toml`:

```toml
[model_providers.adapter]
name = "Proxy"
base_url = "http://127.0.0.1:9090/v1"
wire_api = "responses"
```

Create a profile to use a specific model (e.g. `~/.codex/deepseek.config.toml`):

```toml
model = "deepseek-v4-pro"
model_provider = "adapter"
```

Then launch Codex:

```bash
codex --profile deepseek --dangerously-bypass-approvals-and-sandbox -s danger-full-access
```

For convenience, use one profile per model (e.g. `deepseek-flash`, `xiaomi`, `minimax-pro`) and switch via `--profile`.

## Configuration

### YAML Config File

> **Note**: `format` is optional and auto-detected from `baseUrl` when omitted. Set to `anthropic`, `openai-chat`, `openai-responses`, or `"auto"`.

```yaml
server:
  addr: "0.0.0.0:9090"
  accessLogDir: "~/.ai-adapter/logs"
  reqlog: true
  dropImages: true

providers:
  - name: deepseek
    baseUrl: "https://api.deepseek.com/anthropic"
    format: anthropic   # optional — auto-detected from base_url
    apikey: "${DEEPSEEK_API_KEY}"
    vendor: deepseek
    dropImages: true
    models:
      - deepseek-v4-pro
      - deepseek-v4-flash

modelAliases:
  "gpt-5.5": "deepseek/deepseek-v4-pro"
  "gpt-5.4": "deepseek/deepseek-v4-pro"
  "gpt-5.3-codex": "deepseek/deepseek-v4-pro"
  "*": "deepseek/deepseek-v4-pro"
```

### CLI Options

| Flag                | Env                 | Default                | Description                                  |
| ------------------- | ------------------- | ---------------------- | -------------------------------------------- |
| `-c, --config`      | —                   | —                      | Config file (YAML/JSON)                      |
| `--base-url`        | `UPSTREAM_BASE_URL` | —                      | Upstream API base URL (single provider mode) |
| `--upstream-format` | `UPSTREAM_FORMAT`   | auto-detect from URL   | `anthropic`, `openai-chat`, `openai-responses`|
| `--vendor`          | —                   | `auto`                 | `deepseek`, `openai`, `anthropic`, `xiaomimimo`, `auto` |
| `--apikey`          | `UPSTREAM_API_KEY`  | —                      | Upstream API key                             |
| `--prefer-client-key`| —                   | `false`                | Prefer Authorization header over config key  |
| `--model`           | `UPSTREAM_MODEL`    | —                      | Default model override                       |
| `-a, --addr`        | `ADDR`              | `0.0.0.0:9090`         | Server listen address                        |
| `--model-alias`     | —                   | —                      | Model alias (repeatable, e.g. `gpt-5=provider/model`) |
| `--reqlog`          | —                   | `false`                | Log requests to SQLite                       |
| `--access-log`      | —                   | `false`                | Enable HTTP access log (JSON lines)          |
| `--drop-images`     | —                   | `false`                | Strip images from requests                   |

See `--help` for the full list.

## API Endpoints

| Endpoint                | Method | Description                       |
| ----------------------- | ------ | --------------------------------- |
| `/v1/responses`         | POST   | Responses → upstream Chat/Anthropic |
| `/v1/chat/completions`  | POST   | Chat → upstream                   |
| `/v1/responses/compact` | POST   | Context compaction for Codex CLI  |
| `/v1/models`            | GET    | Pass-through                      |
| `/v1/*`                 | \*     | Catch-all pass-through            |
| `/health`               | GET    | Health check (uptime, addr, vendor) |
| `/__/models`            | GET    | List configured model aliases     |
| `/__/session`           | GET    | List active sessions (JSON)       |

## Vendor Behavior

| Vendor     | Protocol | thinking | developer role | tool format | Special handling                    |
| ---------- | -------- | -------- | -------------- | ----------- | ----------------------------------- |
| DeepSeek   | Chat     | disabled | → system       | flat        | reasoning cache, reasoning_tokens=0 |
| DeepSeek   | Anthropic| disabled | → assistant    | nested      | tool_use/tool_result merge          |
| OpenAI     | Chat     | N/A      | preserved      | nested      | standard                            |
| Anthropic  | Anthropic| N/A      | → assistant    | nested      | standard                            |
| XiaomiMimo | Chat     | none     | → system       | flat        | api-key auth, token-plan support |
| XiaomiMimo | Anthropic| none     | → assistant    | nested      | shared sanitization with DeepSeek   |

## Subcommands

```bash
# Show version info
./target/debug/ai-adapter version

# Print default config template (yaml/json)
./target/debug/ai-adapter config
./target/debug/ai-adapter config --format json

# List active sessions (requires running server)
./target/debug/ai-adapter session ls
```

## Docker

```bash
docker run -d -p 9090:9090 \
  -v ai-adapter-data:/data \
  -e ADDR=0.0.0.0:9090 \
  dyrnq/ai-adapter
```

| Volume / Env        | Description                              |
| ------------------- | ---------------------------------------- |
| `/data` (volume)    | Persistent data: state DB and logs       |
| `DATA_DIR`          | Data directory (default `/data`)         |
| `ADDR`              | Listen address (default `0.0.0.0:9090`)  |
| `AI_ADAPTER_DB`     | SQLite database path                     |

## License

MIT
