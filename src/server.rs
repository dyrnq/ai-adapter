use axum::{
    body::Body,
    extract::{Request, State},
    http::{HeaderMap, HeaderName, HeaderValue, Method, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response, Sse},
    routing::{any, get, post},
    Router,
};
use futures::StreamExt;
use reqwest::Client;
use serde_json::Value;
use std::str::FromStr;
use std::time::Instant;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
};
use tracing::Level;

use crate::config::{RuntimeConfig, UpstreamFormat};
use crate::state::{ReasoningCache, SessionStore};
use crate::stream::sse::{
    AnthropicStreamTranslator, ChatStreamToResponsesTranslator, ResponsesStreamToChatTranslator,
};
use crate::translate::{
    chat_resp_to_responses, convert_anthropic_to_responses, convert_chat_to_responses,
    convert_chat_to_responses_response, convert_for_deepseek, convert_responses_to_anthropic,
    convert_responses_to_anthropic_for,
};
use crate::types::chat::ChatCompletionsRequest;
use crate::types::responses::{
    CompactRequest, CompactResponse, ResponsesContentPart, ResponsesOutputItem, ResponsesRequest,
    ResponsesStreamEvent,
};

/// Shared application state
#[derive(Clone)]
pub struct AppState {
    pub config: RuntimeConfig,
    pub client: Client,
    pub reason_cache: ReasoningCache,
    pub session_store: SessionStore,
}

/// Build the HTTP router
pub fn build_router(
    config: RuntimeConfig,
    reason_cache: ReasoningCache,
    session_store: SessionStore,
    access_log_dir: Option<String>,
) -> Router {
    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(300))
        .build()
        .expect("Failed to create HTTP client");

    let state = AppState {
        config,
        client,
        reason_cache,
        session_store,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers([
            HeaderName::from_static("authorization"),
            HeaderName::from_static("content-type"),
            HeaderName::from_static("x-api-key"),
            HeaderName::from_static("anthropic-version"),
            HeaderName::from_static("anthropic-beta"),
            HeaderName::from_static("anthropic-dangerous-direct-browser-access"),
        ]);

    // HTTP request/response logging via TraceLayer
    let trace_layer = TraceLayer::new_for_http()
        .make_span_with(DefaultMakeSpan::new().level(Level::INFO))
        .on_response(
            DefaultOnResponse::new()
                .level(Level::INFO)
                .latency_unit(tower_http::LatencyUnit::Millis),
        );

    let state_for_error_dump = state.clone();
    let error_dump_layer = middleware::from_fn(move |req, next: Next| {
        let state = state_for_error_dump.clone();
        error_dump_middleware(req, next, state)
    });

    let router = Router::new()
        .route("/v1/chat/completions", post(handle_chat_completions))
        .route("/v1/responses", post(handle_responses))
        .route("/v1/responses/compact", post(handle_compact))
        .route("/v1/models", get(handle_passthrough))
        .route("/health", get(handle_health))
        .route("/__/session", get(handle_session_list))
        .route("/v1/*path", any(handle_passthrough_any))
        .layer(error_dump_layer)
        .layer(trace_layer)
        .layer(cors)
        .with_state(state);

    // Attach access log middleware if configured
    let router = match access_log_dir {
        Some(ref dir) => {
            let log_dir = std::path::PathBuf::from(dir);
            std::fs::create_dir_all(&log_dir).ok();
            let writer = std::sync::Arc::new(std::sync::Mutex::new(AccessLog::new(&log_dir)));
            router.layer(axum::middleware::from_fn(
                move |req: Request, next: Next| {
                    let w = writer.clone();
                    let d = log_dir.clone();
                    async move {
                        let method = req.method().to_string();
                        let uri = req.uri().to_string();
                        let start = std::time::Instant::now();
                        let response = next.run(req).await;
                        let status = response.status().as_u16();
                        let elapsed = start.elapsed().as_millis();
                        if let Ok(mut a) = w.lock() {
                            a.write(&d, &method, &uri, status, elapsed);
                        }
                        response
                    }
                },
            ))
        }
        None => router,
    };

    router
}

async fn handle_health() -> impl IntoResponse {
    axum::Json(serde_json::json!({"status": "ok"}))
}

async fn handle_session_list(State(state): State<AppState>) -> impl IntoResponse {
    let entries = state.session_store.list().await;
    axum::Json(serde_json::json!({
        "sessions": entries,
        "total": entries.len(),
    }))
}

// ============================================================
// /v1/responses/compact — context compaction for Codex CLI
// ============================================================

async fn handle_compact(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    let compact_req: CompactRequest = match serde_json::from_str(&body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                axum::Json(serde_json::json!({"error": {"message": format!("Invalid compact request: {}", e)}})),
            )
                .into_response();
        }
    };

    let api_key = if state.config.prefer_client_key {
        extract_bearer(&headers).or_else(|| state.config.api_key.clone())
    } else {
        state
            .config
            .api_key
            .clone()
            .or_else(|| extract_bearer(&headers))
    };

    // Build a summarization prompt from the conversation history
    let conversation_text = build_compact_prompt(&compact_req);

    // Determine how to call the upstream based on format
    let result = match state.config.upstream_format {
        crate::config::UpstreamFormat::Anthropic => {
            compact_via_anthropic(&state, &compact_req, &conversation_text, api_key).await
        }
        crate::config::UpstreamFormat::OpenAiChat | crate::config::UpstreamFormat::Responses => {
            compact_via_chat(&state, &compact_req, &conversation_text, api_key).await
        }
    };

    result
}

fn build_compact_prompt(req: &CompactRequest) -> String {
    let mut parts: Vec<String> = Vec::new();

    for item in &req.input {
        match item {
            crate::types::responses::ResponsesInputItem::Message { role, content, .. } => {
                let text = crate::types::responses::content_parts_to_text(
                    content.as_deref().unwrap_or(&[]),
                );
                if !text.is_empty() {
                    parts.push(format!("[{}]: {}", role, text));
                }
            }
            crate::types::responses::ResponsesInputItem::FunctionCall {
                name, arguments, ..
            } => {
                parts.push(format!(
                    "[assistant function_call]: {} ({})",
                    name, arguments
                ));
            }
            crate::types::responses::ResponsesInputItem::FunctionCallOutput { output, .. } => {
                parts.push(format!("[tool output]: {}", output));
            }
        }
    }

    let instructions = req.instructions.as_ref().map(|i| i.0.as_str()).unwrap_or(
        "Summarize the conversation above, preserving key decisions, code changes, and context.",
    );

    format!(
        "Please summarize the following conversation. {}\n\nConversation:\n{}",
        instructions,
        parts.join("\n")
    )
}

async fn compact_via_chat(
    state: &AppState,
    req: &CompactRequest,
    prompt: &str,
    api_key: Option<String>,
) -> Response {
    let model = req
        .model
        .clone()
        .or_else(|| state.config.model.clone())
        .unwrap_or_else(|| "gpt-4o".to_string());

    let chat_req = serde_json::json!({
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 4096,
        "temperature": 0.3,
    });

    let upstream_url = build_upstream_url(&state.config.base_url, "/v1/chat/completions");

    let mut upstream_headers = HeaderMap::new();
    upstream_headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    if let Some(ref key) = api_key {
        if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", key)) {
            upstream_headers.insert(HeaderName::from_static("authorization"), val);
        }
    }

    let body_json = serde_json::to_string(&chat_req).unwrap_or_default();

    match state
        .client
        .post(&upstream_url)
        .headers(upstream_headers)
        .body(body_json)
        .send()
        .await
    {
        Ok(resp) => {
            if !resp.status().is_success() {
                // Fallback: return graceful no-op compact
                return make_noop_compact(req);
            }

            let body: serde_json::Value = match resp.json().await {
                Ok(b) => b,
                Err(_) => return make_noop_compact(req),
            };

            let summary = body
                .get("choices")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
                .and_then(|c| c.get("message"))
                .and_then(|m| m.get("content"))
                .and_then(|c| c.as_str())
                .unwrap_or("");

            let id = format!("compact_{}", uuid::Uuid::new_v4());
            let response = CompactResponse {
                id: id.clone(),
                object: "response.compact".to_string(),
                output: vec![
                    crate::types::responses::ResponsesOutputCompactItem::Message {
                        id: format!("msg_{}", uuid::Uuid::new_v4()),
                        role: "assistant".to_string(),
                        content: vec![crate::types::responses::ResponsesContentPart::OutputText {
                            text: if summary.is_empty() {
                                "Conversation context preserved.".to_string()
                            } else {
                                summary.to_string()
                            },
                        }],
                        status: "completed".to_string(),
                    },
                ],
                status: "completed".to_string(),
            };

            (
                StatusCode::OK,
                [("content-type", "application/json")],
                serde_json::to_string(&response).unwrap_or_default(),
            )
                .into_response()
        }
        Err(_) => make_noop_compact(req),
    }
}

async fn compact_via_anthropic(
    state: &AppState,
    req: &CompactRequest,
    prompt: &str,
    api_key: Option<String>,
) -> Response {
    let model = req
        .model
        .clone()
        .or_else(|| state.config.model.clone())
        .unwrap_or_else(|| "claude-sonnet-4-20250514".to_string());

    let anthropic_req = crate::types::anthropic::AnthropicRequest {
        model,
        messages: vec![crate::types::anthropic::AnthropicMessage {
            role: "user".to_string(),
            content: crate::types::anthropic::AnthropicMessageContent::String(prompt.to_string()),
        }],
        max_tokens: 4096,
        system: None,
        metadata: None,
        stop_sequences: None,
        stream: Some(false),
        temperature: None,
        top_p: None,
        top_k: None,
        tools: None,
        tool_choice: None,
        thinking: None,
    };

    let body_json = serde_json::to_string(&anthropic_req).unwrap_or_default();

    let upstream_url = build_upstream_url(&state.config.base_url, "/v1/messages");

    let mut upstream_headers = HeaderMap::new();
    upstream_headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    upstream_headers.insert(
        HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static("2023-06-01"),
    );
    if let Some(ref key) = api_key {
        match state.config.vendor {
            crate::config::UpstreamVendor::XiaomiMimo => {
                if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", key)) {
                    upstream_headers.insert(HeaderName::from_static("authorization"), val);
                }
            }
            _ => {
                if let Ok(val) = HeaderValue::from_str(key) {
                    upstream_headers.insert(HeaderName::from_static("x-api-key"), val);
                }
            }
        }
    }

    match state
        .client
        .post(&upstream_url)
        .headers(upstream_headers)
        .body(body_json)
        .send()
        .await
    {
        Ok(resp) => {
            if !resp.status().is_success() {
                return make_noop_compact(req);
            }

            let body: serde_json::Value = match resp.json().await {
                Ok(b) => b,
                Err(_) => return make_noop_compact(req),
            };

            let summary = body
                .get("content")
                .and_then(|c| c.as_array())
                .and_then(|arr| arr.first())
                .and_then(|b| b.get("text"))
                .and_then(|t| t.as_str())
                .unwrap_or("");

            let response = CompactResponse {
                id: format!("compact_{}", uuid::Uuid::new_v4()),
                object: "response.compact".to_string(),
                output: vec![
                    crate::types::responses::ResponsesOutputCompactItem::Message {
                        id: format!("msg_{}", uuid::Uuid::new_v4()),
                        role: "assistant".to_string(),
                        content: vec![crate::types::responses::ResponsesContentPart::OutputText {
                            text: if summary.is_empty() {
                                "Conversation context preserved.".to_string()
                            } else {
                                summary.to_string()
                            },
                        }],
                        status: "completed".to_string(),
                    },
                ],
                status: "completed".to_string(),
            };

            (
                StatusCode::OK,
                [("content-type", "application/json")],
                serde_json::to_string(&response).unwrap_or_default(),
            )
                .into_response()
        }
        Err(_) => make_noop_compact(req),
    }
}

fn make_noop_compact(_req: &CompactRequest) -> Response {
    let id = format!("compact_{}", uuid::Uuid::new_v4());
    let response = CompactResponse {
        id: id.clone(),
        object: "response.compact".to_string(),
        output: vec![
            crate::types::responses::ResponsesOutputCompactItem::Message {
                id: format!("msg_{}", uuid::Uuid::new_v4()),
                role: "assistant".to_string(),
                content: vec![crate::types::responses::ResponsesContentPart::OutputText {
                    text: "Conversation context preserved (no-op compact).".to_string(),
                }],
                status: "completed".to_string(),
            },
        ],
        status: "completed".to_string(),
    };

    (
        StatusCode::OK,
        [("content-type", "application/json")],
        serde_json::to_string(&response).unwrap_or_default(),
    )
        .into_response()
}

// ============================================================
// /v1/chat/completions -> upstream Responses API (Direction 1)
// ============================================================

async fn handle_chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    // Parse chat request
    let chat_req: ChatCompletionsRequest = match serde_json::from_str(&body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                serde_json::json!({"error": {"message": format!("Invalid request: {}", e)}})
                    .to_string(),
            )
                .into_response();
        }
    };

    let is_stream = chat_req.stream.unwrap_or(false);
    let api_key = if state.config.prefer_client_key {
        extract_bearer(&headers).or_else(|| state.config.api_key.clone())
    } else {
        state
            .config
            .api_key
            .clone()
            .or_else(|| extract_bearer(&headers))
    };

    // Route based on upstream format
    match state.config.upstream_format {
        UpstreamFormat::OpenAiChat => {
            // Chat -> Chat passthrough: send directly to upstream /v1/chat/completions
            return handle_chat_passthrough(&state, &chat_req, is_stream, api_key).await;
        }
        UpstreamFormat::Responses => {
            // Chat -> Responses: convert and send to upstream /v1/responses
            // (original Direction 1 logic)
        }
        UpstreamFormat::Anthropic => {
            // Chat -> Anthropic: convert and send to upstream /v1/messages
            return handle_chat_via_anthropic(&state, &chat_req, is_stream, api_key).await;
        }
    }

    // Convert to Responses API format
    let responses_req = convert_chat_to_responses(&chat_req);

    // Determine upstream URL
    let upstream_url = build_upstream_url(&state.config.base_url, "/v1/responses");
    // Build upstream request
    let mut upstream_headers = HeaderMap::new();
    upstream_headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    if let Some(ref key) = api_key {
        if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", key)) {
            upstream_headers.insert(HeaderName::from_static("authorization"), val);
        }
    }

    let body_json = serde_json::to_string(&responses_req).unwrap_or_default();

    tracing::debug!("Chat->Responses request: POST {}", upstream_url);

    let upstream_resp = match state
        .client
        .post(&upstream_url)
        .headers(upstream_headers)
        .body(body_json)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("Upstream request failed: {}", e);
            return (
                StatusCode::BAD_GATEWAY,
                serde_json::json!({"error": {"message": format!("Upstream error: {}", e)}})
                    .to_string(),
            )
                .into_response();
        }
    };

    let _status = upstream_resp.status();
    let upstream_model = upstream_resp
        .headers()
        .get("x-model")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(&chat_req.model)
        .to_string();

    if is_stream {
        // Stream: convert Responses SSE -> Chat SSE
        let stream = upstream_resp.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::channel::<
            std::result::Result<axum::response::sse::Event, std::convert::Infallible>,
        >(32);

        let model = upstream_model.clone();
        tokio::spawn(async move {
            let mut translator = ResponsesStreamToChatTranslator::new(&model);
            let mut done_sent = false;

            let byte_stream = stream.map(|r| match r {
                Ok(bytes) => Ok(bytes.to_vec()),
                Err(e) => Err(e),
            });

            // TODO: parse SSE properly from byte stream
            // For now, accumulate and process
            let mut buffer = String::new();
            futures::pin_mut!(byte_stream);

            while let Some(result) = byte_stream.next().await {
                match result {
                    Ok(bytes) => {
                        let s = String::from_utf8_lossy(&bytes);
                        buffer.push_str(&s);

                        while let Some(pos) = buffer.find("\n\n") {
                            let chunk = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();

                            for line in chunk.lines() {
                                let trimmed = line.trim();
                                if trimmed.is_empty() {
                                    continue;
                                }
                                if trimmed == "[DONE]" {
                                    if !done_sent {
                                        done_sent = true;
                                        let _ = tx
                                            .send(Ok(axum::response::sse::Event::default()
                                                .data("[DONE]")))
                                            .await;
                                    }
                                    continue;
                                }
                                if let Some(stripped) = trimmed.strip_prefix("data:") {
                                    let data = if trimmed.len() > 5 {
                                        stripped.trim()
                                    } else {
                                        ""
                                    };
                                    if let Ok(event) = serde_json::from_str::<
                                        crate::types::responses::ResponsesStreamEvent,
                                    >(data)
                                    {
                                        let chunks = translator.process_event(&event);
                                        for chunk_json in chunks {
                                            if chunk_json == Value::String("[DONE]".to_string()) {
                                                if !done_sent {
                                                    done_sent = true;
                                                    let _ = tx
                                                        .send(Ok(
                                                            axum::response::sse::Event::default()
                                                                .data("[DONE]"),
                                                        ))
                                                        .await;
                                                }
                                            } else {
                                                let _ = tx
                                                    .send(Ok(axum::response::sse::Event::default()
                                                        .data(
                                                            serde_json::to_string(&chunk_json)
                                                                .unwrap_or_default(),
                                                        )))
                                                    .await;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Stream error: {}", e);
                        break;
                    }
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Sse::new(stream).into_response()
    } else {
        // Non-stream: convert Responses response -> Chat response
        let upstream_body: Value = match upstream_resp.json().await {
            Ok(b) => b,
            Err(e) => {
                tracing::error!("Failed to parse upstream response: {}", e);
                return (
                    StatusCode::BAD_GATEWAY,
                    serde_json::json!({"error": {"message": format!("Invalid upstream response: {}", e)}}).to_string(),
                )
                    .into_response();
            }
        };

        let responses_resp: crate::types::responses::ResponsesResponse =
            match serde_json::from_value(upstream_body) {
                Ok(r) => r,
                Err(e) => {
                    return (
                        StatusCode::BAD_GATEWAY,
                        serde_json::json!({"error": {"message": format!("Failed to parse: {}", e)}}).to_string(),
                    )
                        .into_response();
                }
            };

        let chat_resp = chat_resp_to_responses(&responses_resp, &upstream_model);
        (
            StatusCode::OK,
            [("content-type", "application/json")],
            serde_json::to_string(&chat_resp).unwrap_or_default(),
        )
            .into_response()
    }
}

// ── handle_chat_passthrough: ChatCompletions -> upstream /v1/chat/completions ──
async fn handle_chat_passthrough(
    state: &AppState,
    req: &ChatCompletionsRequest,
    is_stream: bool,
    api_key: Option<String>,
) -> Response {
    let upstream_url = build_upstream_url(&state.config.base_url, "/v1/chat/completions");

    let mut upstream_headers = HeaderMap::new();
    upstream_headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    if let Some(ref key) = api_key {
        if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", key)) {
            upstream_headers.insert(HeaderName::from_static("authorization"), val);
        }
    }

    let body_json = serde_json::to_string(req).unwrap_or_default();
    tracing::debug!("Chat passthrough: POST {}", upstream_url);

    let upstream_resp = match state
        .client
        .post(&upstream_url)
        .headers(upstream_headers)
        .body(body_json)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                serde_json::json!({"error": {"message": format!("Upstream error: {}", e)}})
                    .to_string(),
            )
                .into_response();
        }
    };

    let status = upstream_resp.status();
    if !status.is_success() {
        let error_body = upstream_resp.text().await.unwrap_or_default();
        return (
            status,
            serde_json::json!({"error": {"message": format!("Upstream {}: {}", status.as_u16(), error_body)}}).to_string(),
        )
            .into_response();
    }

    if is_stream {
        let stream = upstream_resp.bytes_stream();
        let stream = stream.map(|r| match r {
            Ok(bytes) => Ok(bytes.to_vec()),
            Err(e) => Err(e),
        });
        let (tx, rx) = tokio::sync::mpsc::channel::<
            std::result::Result<axum::response::sse::Event, std::convert::Infallible>,
        >(32);

        tokio::spawn(async move {
            let mut buffer = String::new();
            futures::pin_mut!(stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(bytes) => {
                        let s = String::from_utf8_lossy(&bytes);
                        buffer.push_str(&s);
                        while let Some(pos) = buffer.find("\n\n") {
                            let chunk = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();
                            for line in chunk.lines() {
                                let trimmed = line.trim();
                                if !trimmed.is_empty() {
                                    let _ = tx
                                        .send(Ok(
                                            axum::response::sse::Event::default().data(trimmed)
                                        ))
                                        .await;
                                }
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            // Send termination on stream break
            let _ = tx
                .send(Ok(axum::response::sse::Event::default().data("[DONE]")))
                .await;
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Sse::new(stream).into_response()
    } else {
        let body = match upstream_resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    serde_json::json!({"error": {"message": format!("Read error: {}", e)}})
                        .to_string(),
                )
                    .into_response();
            }
        };
        let mut response = Response::new(Body::from(body));
        *response.status_mut() = status;
        response.headers_mut().insert(
            HeaderName::from_static("content-type"),
            HeaderValue::from_static("application/json"),
        );
        response
    }
}

// ── handle_chat_via_anthropic: ChatCompletions -> upstream /v1/messages ──
async fn handle_chat_via_anthropic(
    state: &AppState,
    req: &ChatCompletionsRequest,
    is_stream: bool,
    api_key: Option<String>,
) -> Response {
    // Two-step conversion: Chat -> Responses -> Anthropic
    let responses_req = convert_chat_to_responses(req);
    let anthropic_req = convert_responses_to_anthropic(&responses_req);

    let upstream_url = build_upstream_url(&state.config.base_url, "/v1/messages");

    let mut upstream_headers = HeaderMap::new();
    upstream_headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    upstream_headers.insert(
        HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static("2023-06-01"),
    );
    if let Some(ref key) = api_key {
        match state.config.vendor {
            crate::config::UpstreamVendor::XiaomiMimo => {
                if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", key)) {
                    upstream_headers.insert(HeaderName::from_static("authorization"), val);
                }
            }
            _ => {
                if let Ok(val) = HeaderValue::from_str(key) {
                    upstream_headers.insert(HeaderName::from_static("x-api-key"), val);
                }
            }
        }
    }

    let body_json = serde_json::to_string(&anthropic_req).unwrap_or_default();
    tracing::debug!("Chat->Anthropic: POST {}", upstream_url);

    let upstream_resp = match state
        .client
        .post(&upstream_url)
        .headers(upstream_headers)
        .body(body_json)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                serde_json::json!({"error": {"message": format!("Upstream error: {}", e)}})
                    .to_string(),
            )
                .into_response();
        }
    };

    let status = upstream_resp.status();
    if !status.is_success() {
        let error_body = upstream_resp.text().await.unwrap_or_default();
        return (
            status,
            serde_json::json!({"error": {"message": format!("Upstream {}: {}", status.as_u16(), error_body)}}).to_string(),
        )
            .into_response();
    }

    if is_stream {
        // For streaming, passthrough the SSE stream
        let stream = upstream_resp.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::channel::<
            std::result::Result<axum::response::sse::Event, std::convert::Infallible>,
        >(32);

        tokio::spawn(async move {
            let mut buffer = String::new();
            futures::pin_mut!(stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(bytes) => {
                        let s = String::from_utf8_lossy(&bytes);
                        buffer.push_str(&s);
                        while let Some(pos) = buffer.find("\n\n") {
                            let chunk = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();
                            for line in chunk.lines() {
                                let trimmed = line.trim();
                                if !trimmed.is_empty() {
                                    let _ = tx
                                        .send(Ok(
                                            axum::response::sse::Event::default().data(trimmed)
                                        ))
                                        .await;
                                }
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
            // Send termination on stream break
            let _ = tx
                .send(Ok(axum::response::sse::Event::default().data("[DONE]")))
                .await;
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Sse::new(stream).into_response()
    } else {
        // Non-stream: passthrough the response body as-is
        match upstream_resp.bytes().await {
            Ok(body) => {
                let mut response = Response::new(Body::from(body));
                *response.status_mut() = status;
                response.headers_mut().insert(
                    HeaderName::from_static("content-type"),
                    HeaderValue::from_static("application/json"),
                );
                response
            }
            Err(e) => (
                StatusCode::BAD_GATEWAY,
                serde_json::json!({"error": {"message": format!("Read error: {}", e)}}).to_string(),
            )
                .into_response(),
        }
    }
}

// ============================================================
// /v1/responses -> upstream (Responses or Chat or Anthropic)
// ============================================================

async fn handle_responses(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: String,
) -> Response {
    // Extract session_id from header for reasoning cache key
    let session_id = headers
        .get("session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Record session_id for session ls command
    if let Some(ref sid) = session_id {
        let _ = state.session_store.record(sid).await;
    }

    if body.trim().is_empty() {
        return axum::Json(serde_json::json!({})).into_response();
    }

    let responses_req: ResponsesRequest = match serde_json::from_str(&body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                serde_json::json!({"error": {"message": format!("Invalid request: {}", e)}})
                    .to_string(),
            )
                .into_response();
        }
    };

    let is_stream = responses_req.stream.unwrap_or(false);
    tracing::debug!("handle_responses: stream={}, body={}", is_stream, &body);
    let api_key = if state.config.prefer_client_key {
        extract_bearer(&headers).or_else(|| state.config.api_key.clone())
    } else {
        state
            .config
            .api_key
            .clone()
            .or_else(|| extract_bearer(&headers))
    };

    // Route based on upstream format
    match state.config.upstream_format {
        UpstreamFormat::Anthropic => {
            return handle_responses_via_anthropic(
                &state,
                &responses_req,
                is_stream,
                api_key,
                session_id,
            )
            .await;
        }
        UpstreamFormat::OpenAiChat => {
            return handle_responses_via_chat(
                &state,
                &responses_req,
                is_stream,
                api_key,
                session_id,
            )
            .await;
        }
        UpstreamFormat::Responses => {
            // Passthrough: send Responses request directly to upstream
        }
    }

    let upstream_url = build_upstream_url(&state.config.base_url, "/v1/responses");

    let mut upstream_headers = HeaderMap::new();
    upstream_headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    if let Some(ref key) = api_key {
        if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", key)) {
            upstream_headers.insert(HeaderName::from_static("authorization"), val);
        }
    }

    let body_json = serde_json::to_string(&responses_req).unwrap_or_default();

    match state
        .client
        .post(&upstream_url)
        .headers(upstream_headers)
        .body(body_json)
        .send()
        .await
    {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            let body_bytes = match resp.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    return (
                        StatusCode::BAD_GATEWAY,
                        serde_json::json!({"error": {"message": format!("Read error: {}", e)}})
                            .to_string(),
                    )
                        .into_response();
                }
            };

            let mut response = Response::new(Body::from(body_bytes));
            *response.status_mut() = status;
            for (name, value) in resp_headers.iter() {
                if name.as_str() != "transfer-encoding" && name.as_str() != "connection" {
                    response.headers_mut().insert(name.clone(), value.clone());
                }
            }
            response
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            serde_json::json!({"error": {"message": format!("Upstream error: {}", e)}}).to_string(),
        )
            .into_response(),
    }
}

async fn handle_responses_via_chat(
    state: &AppState,
    req: &ResponsesRequest,
    is_stream: bool,
    api_key: Option<String>,
    session_id: Option<String>,
) -> Response {
    // Look up reasoning from previous response for thinking mode compliance
    let previous_reasoning = if let Some(ref prev_id) = req.previous_response_id {
        let sid = session_id.as_deref().unwrap_or("unknown");
        match state.reason_cache.get(sid, prev_id).await {
            Ok(Some(r)) => {
                let len = r.len();
                let truncated = if state.config.truncate_reasoning && len > 32768 {
                    tracing::warn!("Truncated reasoning from {} to 32KB", len);
                    let boundary = r.floor_char_boundary(32768);
                    r[..boundary].to_string()
                } else {
                    r
                };
                tracing::debug!(
                    "Injected {} bytes of reasoning from previous_response_id={}",
                    truncated.len(),
                    prev_id
                );
                Some(truncated)
            }
            Ok(None) => None,
            Err(e) => {
                tracing::error!("Reasoning cache read error: {}", e);
                None
            }
        }
    } else {
        None
    };

    let chat_req = match &state.config.vendor {
        crate::config::UpstreamVendor::XiaomiMimo => {
            crate::translate::xiaomimimo::chat::convert_responses_to_chat(req, previous_reasoning)
        }
        _ => convert_for_deepseek(req, previous_reasoning),
    };

    let upstream_url = build_upstream_url(&state.config.base_url, "/v1/chat/completions");

    let mut upstream_headers = HeaderMap::new();
    upstream_headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    if let Some(ref key) = api_key {
        if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", key)) {
            upstream_headers.insert(HeaderName::from_static("authorization"), val);
        }
    }

    let body_json = serde_json::to_string(&chat_req).unwrap_or_default();
    tracing::debug!(
        "Responses->Chat body: {}",
        &body_json[..body_json.len().min(2000)]
    );

    let upstream_resp = match state
        .client
        .post(&upstream_url)
        .headers(upstream_headers)
        .body(body_json)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                serde_json::json!({"error": {"message": format!("Upstream error: {}", e)}})
                    .to_string(),
            )
                .into_response();
        }
    };

    let upstream_status = upstream_resp.status();
    let upstream_model = upstream_resp
        .headers()
        .get("x-model")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(&req.model)
        .to_string();

    // Check for upstream errors before attempting to stream/parse
    if !upstream_status.is_success() {
        let error_body = upstream_resp.text().await.unwrap_or_default();
        tracing::error!(
            "Upstream returned {}: {}",
            upstream_status.as_u16(),
            &error_body
        );
        return (
            StatusCode::BAD_GATEWAY,
            serde_json::json!({"error": {"message": format!("Upstream error {}: {}", upstream_status.as_u16(), error_body)}}).to_string(),
        )
            .into_response();
    }

    if is_stream {
        // Stream: convert Chat SSE -> Responses SSE
        let stream = upstream_resp.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::channel::<
            std::result::Result<axum::response::sse::Event, std::convert::Infallible>,
        >(64);

        let model = upstream_model.clone();
        let reason_cache = state.reason_cache.clone();
        let response_id = format!("resp_{}", uuid::Uuid::new_v4());
        let session_id_for_save = session_id.clone();
        let is_xiaomimimo = matches!(
            state.config.vendor,
            crate::config::UpstreamVendor::XiaomiMimo
        );
        tokio::spawn(async move {
            let mut translator = ChatStreamToResponsesTranslator::new(&model);
            translator.response_id = response_id.clone();
            translator.strip_xiaomimimo_markers = is_xiaomimimo;
            let mut buffer = String::new();

            futures::pin_mut!(stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(bytes) => {
                        let s = String::from_utf8_lossy(&bytes);
                        buffer.push_str(&s);

                        while let Some(pos) = buffer.find("\n\n") {
                            let chunk = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();

                            for line in chunk.lines() {
                                let trimmed = line.trim();
                                if trimmed.is_empty() {
                                    continue;
                                }
                                if trimmed == "[DONE]" {
                                    // Force finalize: emit remaining events if not finished
                                    if !translator.is_finished() {
                                        translator.set_finished();
                                        let final_events = translator.finalize();
                                        for event in final_events {
                                            send_sse(&tx, &event).await;
                                        }
                                    }
                                    continue;
                                }
                                if let Some(stripped) = trimmed.strip_prefix("data:") {
                                    let data = if trimmed.len() > 5 {
                                        stripped.trim()
                                    } else {
                                        ""
                                    };
                                    if let Ok(chunk_json) = serde_json::from_str::<Value>(data) {
                                        let events = translator.process_chunk(&chunk_json);
                                        for event in events {
                                            send_sse(&tx, &event).await;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Stream error: {}", e);
                        break;
                    }
                }
            }

            // Flush remaining events if stream ended without [DONE] or finish_reason
            if !translator.is_finished() {
                translator.set_finished();
                let final_events = translator.finalize();
                for event in final_events {
                    send_sse(&tx, &event).await;
                }
            }

            // Save reasoning content to cache for future multi-turn requests
            if !translator.reasoning_content.is_empty() {
                let sid = session_id_for_save.as_deref().unwrap_or("unknown");
                if let Err(e) = reason_cache
                    .save(sid, &translator.response_id, &translator.reasoning_content)
                    .await
                {
                    tracing::error!("Failed to save reasoning cache: {}", e);
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Sse::new(stream).into_response()
    } else {
        // Non-stream: convert Chat response -> Responses response
        let upstream_body: Value = match upstream_resp.json().await {
            Ok(b) => b,
            Err(e) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    serde_json::json!({"error": {"message": format!("Invalid upstream response: {}", e)}}).to_string(),
                )
                    .into_response();
            }
        };

        let chat_resp: crate::types::chat::ChatCompletionsResponse =
            match serde_json::from_value(upstream_body) {
                Ok(r) => r,
                Err(e) => {
                    return (
                    StatusCode::BAD_GATEWAY,
                    serde_json::json!({"error": {"message": format!("Failed to parse: {}", e)}})
                        .to_string(),
                )
                    .into_response();
                }
            };

        let mut responses_resp = convert_chat_to_responses_response(&chat_resp, &upstream_model);

        // Strip xiaomimimo reasoning markers from output text
        if matches!(
            state.config.vendor,
            crate::config::UpstreamVendor::XiaomiMimo
        ) {
            for item in &mut responses_resp.output {
                if let ResponsesOutputItem::Message { content, .. } = item {
                    for part in content {
                        if let ResponsesContentPart::OutputText { text } = part {
                            *text = text
                                .replace("[[REASONING_SUMMARY]]", "")
                                .replace("[[REASONING_DIVIDER]]", "");
                        }
                    }
                }
            }
        }

        // Save reasoning to cache for multi-turn (non-stream path)
        if let Some(ref sid) = session_id {
            for choice in &chat_resp.choices {
                if let Some(ref msg) = choice.message {
                    if let Some(ref reasoning) = msg.reasoning_content {
                        if !reasoning.is_empty() {
                            let _ = state
                                .reason_cache
                                .save(sid, &responses_resp.id, reasoning)
                                .await;
                        }
                    }
                }
            }
        }

        (
            StatusCode::OK,
            [("content-type", "application/json")],
            serde_json::to_string(&responses_resp).unwrap_or_default(),
        )
            .into_response()
    }
}

async fn handle_responses_via_anthropic(
    state: &AppState,
    req: &ResponsesRequest,
    is_stream: bool,
    api_key: Option<String>,
    session_id: Option<String>,
) -> Response {
    // Look up reasoning from previous response (mirrors Chat path)
    let _previous_reasoning = if let Some(ref prev_id) = req.previous_response_id {
        let sid = session_id.as_deref().unwrap_or("unknown");
        match state.reason_cache.get(sid, prev_id).await {
            Ok(Some(r)) => {
                tracing::debug!(
                    "Anthropic: found {} bytes of reasoning from {}",
                    r.len(),
                    prev_id
                );
                Some(r)
            }
            Ok(None) => None,
            Err(e) => {
                tracing::error!("Reasoning cache read error: {}", e);
                None
            }
        }
    } else {
        None
    };

    let anthropic_req = convert_responses_to_anthropic_for(req, &state.config.vendor);

    let upstream_url = build_upstream_url(&state.config.base_url, "/v1/messages");

    let mut upstream_headers = HeaderMap::new();
    upstream_headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    upstream_headers.insert(
        HeaderName::from_static("anthropic-version"),
        HeaderValue::from_static("2023-06-01"),
    );
    if let Some(ref key) = api_key {
        match state.config.vendor {
            crate::config::UpstreamVendor::XiaomiMimo => {
                if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", key)) {
                    upstream_headers.insert(HeaderName::from_static("authorization"), val);
                }
            }
            _ => {
                if let Ok(val) = HeaderValue::from_str(key) {
                    upstream_headers.insert(HeaderName::from_static("x-api-key"), val);
                }
            }
        }
    }

    let body_json = serde_json::to_string(&anthropic_req).unwrap_or_default();
    tracing::debug!("Responses->Anthropic request to {}", upstream_url);

    let upstream_resp = match state
        .client
        .post(&upstream_url)
        .headers(upstream_headers)
        .body(body_json)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                serde_json::json!({"error": {"message": format!("Upstream error: {}", e)}})
                    .to_string(),
            )
                .into_response();
        }
    };

    let upstream_status = upstream_resp.status();
    let upstream_model = upstream_resp
        .headers()
        .get("x-model")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(&req.model)
        .to_string();

    // Check for upstream errors before attempting to stream/parse
    if !upstream_status.is_success() {
        let error_body = upstream_resp.text().await.unwrap_or_default();
        tracing::error!(
            "Upstream returned {}: {}",
            upstream_status.as_u16(),
            &error_body
        );
        return (
            StatusCode::BAD_GATEWAY,
            serde_json::json!({"error": {"message": format!("Upstream error {}: {}", upstream_status.as_u16(), error_body)}}).to_string(),
        )
            .into_response();
    }

    if is_stream {
        let stream = upstream_resp.bytes_stream();
        let (tx, rx) = tokio::sync::mpsc::channel::<
            std::result::Result<axum::response::sse::Event, std::convert::Infallible>,
        >(64);

        let model = upstream_model.clone();
        let reason_cache = state.reason_cache.clone();
        let response_id = format!("resp_{}", uuid::Uuid::new_v4());
        let session_id_for_save = session_id.clone();
        let _is_xiaomimimo = matches!(
            state.config.vendor,
            crate::config::UpstreamVendor::XiaomiMimo
        );
        tokio::spawn(async move {
            let mut translator = AnthropicStreamTranslator::new(&model);
            translator.response_id = response_id.clone();
            let mut buffer = String::new();

            futures::pin_mut!(stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(bytes) => {
                        let s = String::from_utf8_lossy(&bytes);
                        buffer.push_str(&s);

                        while let Some(pos) = buffer.find("\n\n") {
                            let chunk = buffer[..pos].to_string();
                            buffer = buffer[pos + 2..].to_string();

                            for line in chunk.lines() {
                                let trimmed = line.trim();
                                if trimmed.is_empty() {
                                    continue;
                                }
                                if let Some(stripped) = trimmed.strip_prefix("data:") {
                                    let data = if trimmed.len() > 5 {
                                        stripped.trim()
                                    } else {
                                        ""
                                    };
                                    if let Ok(event) = serde_json::from_str::<
                                        crate::types::anthropic::AnthropicStreamEvent,
                                    >(data)
                                    {
                                        let events = translator.process_event(&event);
                                        for e in events {
                                            send_sse(&tx, &e).await;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Stream error: {}", e);
                        break;
                    }
                }
            }

            // Ensure response.completed is emitted even if stream ended prematurely
            if translator.started && !translator.event_completed {
                let final_resp = translator.make_completed_response();
                let event = ResponsesStreamEvent::ResponseCompleted {
                    response: final_resp,
                    sequence_number: 0,
                };
                send_sse(&tx, &event).await;
            }

            // Save reasoning content to cache for multi-turn
            if !translator.reasoning_content.is_empty() {
                let sid = session_id_for_save.as_deref().unwrap_or("unknown");
                if let Err(e) = reason_cache
                    .save(sid, &translator.response_id, &translator.reasoning_content)
                    .await
                {
                    tracing::error!("Failed to save reasoning cache: {}", e);
                }
            }
        });

        let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
        Sse::new(stream).into_response()
    } else {
        let upstream_body: Value = match upstream_resp.json().await {
            Ok(b) => b,
            Err(e) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    serde_json::json!({"error": {"message": format!("Parse error: {}", e)}})
                        .to_string(),
                )
                    .into_response();
            }
        };

        let anthropic_resp: crate::types::anthropic::AnthropicResponse =
            match serde_json::from_value(upstream_body) {
                Ok(r) => r,
                Err(e) => {
                    return (
                        StatusCode::BAD_GATEWAY,
                        serde_json::json!({"error": {"message": format!("Parse error: {}", e)}})
                            .to_string(),
                    )
                        .into_response();
                }
            };

        let responses_resp = convert_anthropic_to_responses(&anthropic_resp, &upstream_model);

        // Save reasoning to cache for multi-turn (non-stream)
        if let Some(ref sid) = session_id {
            let reasoning_text: String = anthropic_resp
                .content
                .iter()
                .filter_map(|b| {
                    if let crate::types::anthropic::AnthropicContentBlock::Thinking {
                        thinking,
                        ..
                    } = b
                    {
                        Some(thinking.as_str())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            if !reasoning_text.is_empty() {
                let _ = state
                    .reason_cache
                    .save(sid, &responses_resp.id, &reasoning_text)
                    .await;
            }
        }

        (
            StatusCode::OK,
            [("content-type", "application/json")],
            serde_json::to_string(&responses_resp).unwrap_or_default(),
        )
            .into_response()
    }
}

// ============================================================
// Passthrough handlers
// ============================================================

async fn handle_passthrough_any(
    State(state): State<AppState>,
    method: Method,
    headers: HeaderMap,
    uri: axum::http::Uri,
    body: Option<String>,
) -> Response {
    let body_str = body.as_deref();
    passthrough_request(&state, &headers, method, &uri, body_str).await
}

#[allow(dead_code)]
async fn handle_passthrough_get(
    State(state): State<AppState>,
    headers: HeaderMap,
    uri: axum::http::Uri,
) -> Response {
    passthrough_request(&state, &headers, Method::GET, &uri, None).await
}

#[allow(dead_code)]
async fn handle_passthrough_post(
    State(state): State<AppState>,
    headers: HeaderMap,
    uri: axum::http::Uri,
    body: String,
) -> Response {
    passthrough_request(&state, &headers, Method::POST, &uri, Some(&body)).await
}

async fn handle_passthrough(
    State(state): State<AppState>,
    headers: HeaderMap,
    uri: axum::http::Uri,
) -> Response {
    passthrough_request(&state, &headers, Method::GET, &uri, None).await
}

async fn passthrough_request(
    state: &AppState,
    headers: &HeaderMap,
    method: Method,
    uri: &axum::http::Uri,
    body: Option<&str>,
) -> Response {
    let path = uri.path();
    let upstream_url = format!("{}{}", state.config.base_url.trim_end_matches('/'), path);

    let api_key = if state.config.prefer_client_key {
        extract_bearer(headers).or_else(|| state.config.api_key.clone())
    } else {
        state
            .config
            .api_key
            .clone()
            .or_else(|| extract_bearer(headers))
    };

    let mut upstream_headers = HeaderMap::new();
    upstream_headers.insert(
        HeaderName::from_static("content-type"),
        HeaderValue::from_static("application/json"),
    );
    if let Some(ref key) = api_key {
        if let Ok(val) = HeaderValue::from_str(&format!("Bearer {}", key)) {
            upstream_headers.insert(HeaderName::from_static("authorization"), val);
        }
    }

    let request = state
        .client
        .request(method, &upstream_url)
        .headers(upstream_headers);

    let request = if let Some(body_str) = body {
        request.body(body_str.to_string())
    } else {
        request
    };

    match request.send().await {
        Ok(resp) => {
            let status = resp.status();
            let resp_headers = resp.headers().clone();
            let body_bytes = match resp.bytes().await {
                Ok(b) => b,
                Err(e) => {
                    return (
                        StatusCode::BAD_GATEWAY,
                        axum::Json(
                            serde_json::json!({"error": {"message": format!("Read error: {}", e)}}),
                        ),
                    )
                        .into_response();
                }
            };

            let mut response = Response::new(Body::from(body_bytes));
            *response.status_mut() = status;
            for (name, value) in resp_headers.iter() {
                if name.as_str() != "transfer-encoding" && name.as_str() != "connection" {
                    response.headers_mut().insert(name.clone(), value.clone());
                }
            }
            response
        }
        Err(e) => (
            StatusCode::BAD_GATEWAY,
            axum::Json(
                serde_json::json!({"error": {"message": format!("Passthrough error: {}", e)}}),
            ),
        )
            .into_response(),
    }
}

/// Send a ResponsesStreamEvent as a properly formatted axum SSE event
async fn send_sse(
    tx: &tokio::sync::mpsc::Sender<
        std::result::Result<axum::response::sse::Event, std::convert::Infallible>,
    >,
    event: &ResponsesStreamEvent,
) {
    let json = serde_json::to_string(event).unwrap_or_default();
    let event_type = crate::stream::sse::event_type_str(event);
    let _ = tx
        .send(Ok(axum::response::sse::Event::default()
            .event(event_type)
            .data(json)))
        .await;
}

/// Build upstream URL smartly: if base_url already contains the target path, use as-is.
/// Otherwise append the path, avoiding double-segment issues like /v1/v1/chat/completions.
pub fn build_upstream_url(base_url: &str, target_path: &str) -> String {
    let base = base_url.trim_end_matches('/');

    // If base already ends with the target path (or a sub-path of it), use as-is
    if base.ends_with(target_path) {
        return base.to_string();
    }

    // Check if base ends with a common API prefix like /v1
    // and target_path also starts with /v1 — avoid doubling
    // e.g. base=https://api.deepseek.com/v1, target_path=/v1/chat/completions
    // -> should become https://api.deepseek.com/v1/chat/completions
    // Look for the last /vN segment in base and see if target_path repeats it
    if let Some(last_segment) = base.rsplit('/').next() {
        if let Some(first_segment) = target_path.trim_start_matches('/').split('/').next() {
            if last_segment == first_segment {
                // base: .../v1, target_path: /v1/chat/completions
                // append only the rest of target_path after the common prefix
                let rest = target_path
                    .trim_start_matches('/')
                    .split('/')
                    .skip(1)
                    .collect::<Vec<_>>()
                    .join("/");
                return format!("{}/{}", base, rest);
            }
        }
    }

    format!("{}{}", base, target_path)
}

fn extract_bearer(headers: &HeaderMap) -> Option<String> {
    headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            if let Some(stripped) = v.strip_prefix("Bearer ") {
                stripped.to_string()
            } else {
                v.to_string()
            }
        })
}

// ============================================================
// Error dump middleware — saves failed exchanges to logs/
// ============================================================

async fn error_dump_middleware(req: Request, next: Next, state: AppState) -> Response {
    let start = Instant::now();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let req_headers = redact_headers(req.headers());

    // Capture request body for logging
    let (parts, body) = req.into_parts();
    let body_bytes = match axum::body::to_bytes(body, 50 * 1024 * 1024).await {
        Ok(b) => b,
        Err(e) => {
            tracing::warn!(
                "Request body too large ({}) for error dump, body discarded",
                e
            );
            axum::body::Bytes::new()
        }
    };
    let req_body = String::from_utf8_lossy(&body_bytes).to_string();

    // Note: when body exceeds the limit, body_bytes is empty and the
    // reconstructed request below delivers an empty body to the handler.
    // The handler's empty-body guard returns {} which Codex sees as an error.
    // If this happens, increase the limit above or use a streaming body.
    let req = Request::from_parts(parts, Body::from(body_bytes.clone()));

    let response = next.run(req).await;
    let elapsed = start.elapsed();
    let status = response.status();

    // Log every request
    tracing::info!(
        "{} {} -> {} ({:.0}ms)",
        method,
        uri.path(),
        status.as_u16(),
        elapsed.as_secs_f64() * 1000.0
    );

    // Dump failed exchanges to file on 4xx/5xx
    if status.is_client_error() || status.is_server_error() {
        let (resp_parts, resp_body) = response.into_parts();
        let resp_body_bytes = axum::body::to_bytes(resp_body, 1024 * 1024)
            .await
            .unwrap_or_default();
        let resp_body_str = String::from_utf8_lossy(&resp_body_bytes).to_string();

        let dump = serde_json::json!({
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "request": {
                "method": method.to_string(),
                "uri": uri.to_string(),
                "headers": header_map_to_json(&req_headers),
                "body": truncate_for_log(&req_body, 4096),
            },
            "response": {
                "status": status.as_u16(),
                "headers": header_map_to_json(&resp_parts.headers),
                "body": truncate_for_log(&resp_body_str, 4096),
            },
            "duration_ms": elapsed.as_secs_f64() * 1000.0,
        });

        if let Ok(dump_str) = serde_json::to_string_pretty(&dump) {
            save_error_dump(status.as_u16(), &dump_str, state.config.access_log_dir.as_deref());
        }

        return Response::from_parts(resp_parts, Body::from(resp_body_bytes));
    }

    response
}

fn redact_headers(headers: &HeaderMap) -> HeaderMap {
    let mut h = headers.clone();
    let sensitive = [
        "authorization",
        "x-api-key",
        "api-key",
        "cookie",
        "set-cookie",
    ];
    for name in &sensitive {
        if h.contains_key(*name) {
            h.insert(
                HeaderName::from_str(name).unwrap(),
                HeaderValue::from_static("***REDACTED***"),
            );
        }
    }
    h
}

fn header_map_to_json(headers: &HeaderMap) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (name, value) in headers.iter() {
        if let Ok(v) = value.to_str() {
            map.insert(name.to_string(), serde_json::Value::String(v.to_string()));
        }
    }
    serde_json::Value::Object(map)
}

fn truncate_for_log(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!(
            "{}... [truncated {} bytes]",
            &s[..max_len],
            s.len() - max_len
        )
    }
}

fn save_error_dump(status: u16, dump_json: &str, log_dir: Option<&str>) {
    let dir = match log_dir {
        Some(d) => std::path::PathBuf::from(d),
        None => std::env::var("HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::path::PathBuf::from("."))
            .join(".ai-adapter")
            .join("logs"),
    };
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::error!("Failed to create logs directory: {}", e);
        return;
    }

    let filename = format!(
        "proxy-error-{}-{}.json",
        chrono::Utc::now().format("%Y%m%dT%H%M%S"),
        status
    );
    let path = dir.join(&filename);

    match std::fs::write(&path, dump_json) {
        Ok(_) => tracing::warn!("Error dump saved to {}", path.display()),
        Err(e) => tracing::error!("Failed to save error dump: {}", e),
    }
}

struct AccessLog {
    file: std::fs::File,
    date: String,
}

impl AccessLog {
    fn new(dir: &std::path::Path) -> Self {
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        let path = dir.join(format!("access.{}.log", today));
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .unwrap_or_else(|_| std::fs::File::create(&path).unwrap());
        Self { file, date: today }
    }

    fn write(
        &mut self,
        dir: &std::path::Path,
        method: &str,
        uri: &str,
        status: u16,
        elapsed_ms: u128,
    ) {
        use std::io::Write;
        let today = chrono::Utc::now().format("%Y-%m-%d").to_string();
        if self.date != today {
            let path = dir.join(format!("access.{}.log", today));
            self.file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .unwrap_or_else(|_| std::fs::File::create(&path).unwrap());
            self.date = today;
        }
        let now = chrono::Utc::now().to_rfc3339();
        let _ = writeln!(
            self.file,
            r#"{{"time":"{}","method":"{}","uri":"{}","status":{},"latency_ms":{}}}"#,
            now, method, uri, status, elapsed_ms
        );
        let _ = self.file.flush();
    }
}
