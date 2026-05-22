# AI Adapter

A Rust proxy that exposes an **OpenAI Responses API** interface, translating to various upstream backends (DeepSeek, OpenAI, Anthropic) with full streaming SSE support. Built for Codex CLI and similar tools hardcoded to OpenAI.

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
(Responses API)        anthropic/anthropic.rs
```

## Features

- **Vendor isolation**: `src/translate/{deepseek,openai,anthropic}/` — each vendor gets its own module with vendor-specific quirks
- **Bidirectional streaming**: Full SSE event translation in all directions (Chat ↔ Responses, Anthropic ↔ Responses)
- **DeepSeek Plan A/B**:
  - **Plan A** (Chat): `thinking: disabled`, `developer → system` role mapping, reasoning cache for multi-turn
  - **Plan B** (Anthropic): tool_use/tool_result merging, native thinking support
- **Reasoning cache**: On-disk `redb` cache at `~/.ai-adapter/state.redb` for multi-turn thinking compliance
- **Tool calling**: Full function call streaming, flat & nested tool format support (`get_function()`)
- **Compaction**: `POST /v1/responses/compact` for Codex CLI context management
- **Config-driven**: YAML/JSON config files, environment variables, CLI flags
- **Error dumps**: Saves failed exchanges to `logs/` with auth redaction
- **Structured logging**: Human-readable to stderr, JSON to daily-rotated files

## Quick Start

```bash
# Build
cargo build --release

# Run with inline config
./target/release/ai-adapter \
  --base-url https://api.deepseek.com/anthropic \
  --upstream-format anthropic \
  --apikey sk-your-key \
  --model deepseek-v4-pro
```

Point Codex at `http://127.0.0.1:9090/v1`.

## Subcommands

```bash
# Show version info
./target/release/ai-adapter version

# Print default config template (yaml)
./target/release/ai-adapter config
./target/release/ai-adapter config --format json

# List active sessions (requires running server)
./target/release/ai-adapter session ls
```

## CLI Options

| Flag                | Env                 | Default       | Description                                  |
| ------------------- | ------------------- | ------------- | -------------------------------------------- |
| `-c, --config`      | —                   | —             | Config file (YAML/JSON)                      |
| `--base-url`        | `UPSTREAM_BASE_URL` | —             | Upstream API base URL                        |
| `--upstream-format` | `UPSTREAM_FORMAT`   | `openai-chat` | `anthropic`, `openai-chat`, `responses`      |
| `--vendor`          | —                   | `auto`        | `deepseek`, `openai`, `anthropic`, `auto`    |
| `--apikey`          | `UPSTREAM_API_KEY`  | —             | Upstream API key                             |
| `--model`           | `UPSTREAM_MODEL`    | —             | Default model override                       |
| `-a, --addr`        | `ADDR`              | `0.0.0.0:9090`| Server listen address                        |
| `--log-level`       | `RUST_LOG`          | `info`        | `trace`, `debug`, `info`, `warn`, `error`    |
| `--log-dir`         | `LOG_DIR`           | `$DATA_DIR/logs`| Write JSON logs to directory (daily rotate) |
| `--logtostderr`     | `LOGTOSTDERROR`     | `true`        | Log to stderr (set false for file-only)      |
| `--alsologtostderr` | `ALSOLOGTOSTDERROR` | `false`       | Log to stderr as well as files               |
| `--access-log`      | —                   | —             | Log HTTP request/response bodies             |
| `--access-log-dir`  | `ACCESS_LOG_DIR`    | `$LOG_DIR`    | Write HTTP access logs to directory (JSON)   |
| `--drop-images`     | —                   | —             | Strip images from requests                   |
| `--no-cors`         | —                   | —             | Disable CORS headers                         |

## API Endpoints

| Endpoint                | Method | Description                       |
| ----------------------- | ------ | --------------------------------- |
| `/v1/chat/completions`  | POST   | Chat → upstream Responses         |
| `/v1/responses`         | POST   | Responses → upstream Chat/Anthropic |
| `/v1/responses/compact` | POST   | Context compaction for Codex CLI  |
| `/v1/models`            | GET    | Pass-through                      |
| `/v1/*`                 | \*     | Catch-all pass-through            |
| `/__/session`           | GET    | List active sessions (JSON)       |
| `/health`               | GET    | Health check                      |

## Vendor Behavior

| Vendor     | Protocol | thinking | developer role | tool format | Special handling                    |
| ---------- | -------- | -------- | -------------- | ----------- | ----------------------------------- |
| DeepSeek   | Chat     | disabled | → system       | flat        | reasoning cache, reasoning_tokens=0 |
| DeepSeek   | Anthropic| disabled | → assistant    | nested      | tool_use/tool_result merge          |
| OpenAI     | Chat     | N/A      | preserved      | nested      | standard                            |
| Anthropic  | Anthropic| N/A      | → assistant    | nested      | standard                            |

## Stream Event Mapping

### Chat SSE → Responses SSE

| Chat chunk                      | Responses event                      |
| ------------------------------- | ------------------------------------ |
| First chunk                     | `response.created` + `output_item.added` + `content_part.added` |
| `delta.content`                 | `response.output_text.delta`         |
| `delta.tool_calls` (new)        | `response.output_item.added` (function_call) |
| `delta.tool_calls` (subsequent) | `response.function_call_arguments.delta` |
| `finish_reason` / `[DONE]`      | `output_text.done` → `content_part.done` → `output_item.done` → `response.completed` |

### Anthropic SSE → Responses SSE

| Anthropic event              | Responses event                      |
| ---------------------------- | ------------------------------------ |
| `message_start`              | `response.created` + `response.in_progress` |
| `content_block_start` (text) | `output_item.added` + `content_part.added` |
| `content_block_start` (tool_use) | `output_item.added` (function_call) |
| `content_block_delta.text_delta` | `output_text.delta`              |
| `content_block_delta.input_json_delta` | `function_call_arguments.delta` |
| `content_block_stop`         | `output_item.done`                   |
| `message_stop`               | `response.completed`                 |

## Usage Examples

### curl test

```bash
curl http://localhost:9090/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-pro",
    "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Hello!"}]}],
    "stream": true
  }'
```

## Docker

Use the pre-built image from Docker Hub:

```bash
docker run -d -p 9090:9090 \
  -v ai-adapter-data:/data \
  -e UPSTREAM_BASE_URL=https://api.deepseek.com/anthropic \
  -e UPSTREAM_FORMAT=anthropic \
  -e UPSTREAM_API_KEY=sk-xxx \
  dyrnq/ai-adapter
```

| Volume / Env        | Description                              |
| ------------------- | ---------------------------------------- |
| `/data` (volume)    | Persistent data: state DB and logs       |
| `DATA_DIR`          | Data directory (default `/data`)         |
| `ADDR`              | Listen address (default `0.0.0.0:9090`)  |
| `LOGTOSTDERROR`     | Log to stderr (default `true`)           |
| `LOG_DIR`           | App log directory (default `$DATA_DIR/logs`) |
| `ACCESS_LOG_DIR`    | HTTP access log directory                |

## License

MIT
