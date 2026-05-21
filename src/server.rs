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
};
use crate::types::chat::ChatCompletionsRequest;
use crate::types::responses::{
    CompactRequest, CompactResponse, ResponsesRequest, ResponsesStreamEvent,
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

    Router::new()
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
        .with_state(state)
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

    let api_key = state
        .config
        .api_key
        .clone()
        .or_else(|| extract_bearer(&headers));

    // Build a summarization prompt from the conversation history
    let conversation_text = build_compact_prompt(&compact_req);

    // Determine how to call the upstream based on format
    match state.config.upstream_format {
        crate::config::UpstreamFormat::Anthropic => {
            compact_via_anthropic(&state, &compact_req, &conversation_text, api_key).await
        }
        crate::config::UpstreamFormat::OpenAiChat | crate::config::UpstreamFormat::Responses => {
            compact_via_chat(&state, &compact_req, &conversation_text, api_key).await
        }
    }
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

    let instructions = req.instructions.as_deref().unwrap_or(
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

    let anthropic_req = serde_json::json!({
        "model": model,
        "max_tokens": 4096,
        "messages": [
            {"role": "user", "content": prompt}
        ],
    });

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
        if let Ok(val) = HeaderValue::from_str(key) {
            upstream_headers.insert(HeaderName::from_static("x-api-key"), val);
        }
    }

    let body_json = serde_json::to_string(&anthropic_req).unwrap_or_default();

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

    // Convert to Responses API format
    let responses_req = convert_chat_to_responses(&chat_req);

    // Determine upstream URL
    let upstream_url = build_upstream_url(&state.config.base_url, "/v1/responses");
    let api_key = state
        .config
        .api_key
        .clone()
        .or_else(|| extract_bearer(&headers));

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
                        if let Ok(s) = String::from_utf8(bytes) {
                            buffer.push_str(&s);

                            while let Some(pos) = buffer.find("\n\n") {
                                let chunk = buffer[..pos].to_string();
                                buffer = buffer[pos + 2..].to_string();

                                for line in chunk.lines() {
                                    let trimmed = line.trim();
                                    if trimmed.is_empty() || trimmed == "[DONE]" {
                                        continue;
                                    }
                                    if trimmed.starts_with("data:") {
                                        let data = if trimmed.len() > 5 {
                                            trimmed[5..].trim()
                                        } else {
                                            ""
                                        };
                                        if let Ok(event) =
                                            serde_json::from_str::<
                                                crate::types::responses::ResponsesStreamEvent,
                                            >(data)
                                        {
                                            let chunks = translator.process_event(&event);
                                            for chunk_json in chunks {
                                                if chunk_json == Value::String("[DONE]".to_string())
                                                {
                                                    let _ = tx
                                                        .send(Ok(
                                                            axum::response::sse::Event::default()
                                                                .data("[DONE]"),
                                                        ))
                                                        .await;
                                                } else {
                                                    let _ = tx
                                                        .send(Ok(
                                                            axum::response::sse::Event::default()
                                                                .data(
                                                                    serde_json::to_string(
                                                                        &chunk_json,
                                                                    )
                                                                    .unwrap_or_default(),
                                                                ),
                                                        ))
                                                        .await;
                                                }
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
    let api_key = state
        .config
        .api_key
        .clone()
        .or_else(|| extract_bearer(&headers));

    // Save session data from Codex requests (only when session-id header present)
    if let Some(ref sid) = session_id {
        let _ = state.session_store.save(sid, &body).await;
    }

    match state.config.upstream_format {
        UpstreamFormat::Responses => {
            // Pass-through to Responses API
            handle_responses_passthrough(&state, &responses_req, is_stream, api_key).await
        }
        UpstreamFormat::OpenAiChat => {
            // Convert to Chat Completions
            handle_responses_via_chat(
                &state,
                &responses_req,
                is_stream,
                api_key,
                session_id.clone(),
            )
            .await
        }
        UpstreamFormat::Anthropic => {
            // Convert to Anthropic Messages
            handle_responses_via_anthropic(
                &state,
                &responses_req,
                is_stream,
                api_key,
                session_id.clone(),
            )
            .await
        }
    }
}

async fn handle_responses_passthrough(
    state: &AppState,
    req: &ResponsesRequest,
    _is_stream: bool,
    api_key: Option<String>,
) -> Response {
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

    let body_json = serde_json::to_string(req).unwrap_or_default();

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
                tracing::debug!(
                    "Injected {} bytes of reasoning from previous_response_id={}",
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

    let chat_req = convert_for_deepseek(req, previous_reasoning);

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
        tokio::spawn(async move {
            let mut translator = ChatStreamToResponsesTranslator::new(&model);
            translator.response_id = response_id.clone();
            let mut buffer = String::new();

            futures::pin_mut!(stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(bytes) => {
                        if let Ok(s) = String::from_utf8(bytes.to_vec()) {
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
                                    if trimmed.starts_with("data:") {
                                        let data = if trimmed.len() > 5 {
                                            trimmed[5..].trim()
                                        } else {
                                            ""
                                        };
                                        if let Ok(chunk_json) = serde_json::from_str::<Value>(data)
                                        {
                                            let events = translator.process_chunk(&chunk_json);
                                            for event in events {
                                                send_sse(&tx, &event).await;
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

        let responses_resp = convert_chat_to_responses_response(&chat_resp, &upstream_model);
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
    _session_id: Option<String>,
) -> Response {
    let anthropic_req = convert_responses_to_anthropic(req);

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
        if let Ok(val) = HeaderValue::from_str(key) {
            upstream_headers.insert(HeaderName::from_static("x-api-key"), val);
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
        tokio::spawn(async move {
            let mut translator = AnthropicStreamTranslator::new(&model);
            let mut buffer = String::new();

            futures::pin_mut!(stream);
            while let Some(result) = stream.next().await {
                match result {
                    Ok(bytes) => {
                        if let Ok(s) = String::from_utf8(bytes.to_vec()) {
                            buffer.push_str(&s);

                            while let Some(pos) = buffer.find("\n\n") {
                                let chunk = buffer[..pos].to_string();
                                buffer = buffer[pos + 2..].to_string();

                                for line in chunk.lines() {
                                    let trimmed = line.trim();
                                    if trimmed.is_empty() {
                                        continue;
                                    }
                                    if trimmed.starts_with("data:") {
                                        let data = if trimmed.len() > 5 {
                                            trimmed[5..].trim()
                                        } else {
                                            ""
                                        };
                                        if let Ok(event) =
                                            serde_json::from_str::<
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

async fn handle_passthrough_get(
    State(state): State<AppState>,
    headers: HeaderMap,
    uri: axum::http::Uri,
) -> Response {
    passthrough_request(&state, &headers, Method::GET, &uri, None).await
}

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

    let api_key = state
        .config
        .api_key
        .clone()
        .or_else(|| extract_bearer(headers));

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
fn build_upstream_url(base_url: &str, target_path: &str) -> String {
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
        .and_then(|v| {
            if v.starts_with("Bearer ") {
                Some(v[7..].to_string())
            } else {
                Some(v.to_string())
            }
        })
}

// ============================================================
// Error dump middleware — saves failed exchanges to logs/
// ============================================================

async fn error_dump_middleware(req: Request, next: Next, _state: AppState) -> Response {
    let start = Instant::now();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let req_headers = redact_headers(req.headers());

    // Capture request body for logging
    let (parts, body) = req.into_parts();
    let body_bytes = axum::body::to_bytes(body, 1024 * 1024)
        .await
        .unwrap_or_default();
    let req_body = String::from_utf8_lossy(&body_bytes).to_string();

    // Reconstruct request for the next handler
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
            save_error_dump(status.as_u16(), &dump_str);
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

fn save_error_dump(status: u16, dump_json: &str) {
    let dir = std::env::var("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("."))
        .join(".ai-adapter")
        .join("logs");
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
