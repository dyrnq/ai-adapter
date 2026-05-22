use crate::types::chat::*;
use crate::types::responses::*;
use serde_json::Value;

// =================================================================================================
// DeepSeek A方案: Responses API ↔ Chat Completions bidirectional translation
//
// DeepSeek quirks handled here:
//   - "developer" role is not supported → mapped to "system"
//   - thinking.type is set to "disabled" on all requests (DeepSeek requires explicit opt-in)
//   - previous_reasoning from a prior turn is injected into the last assistant message's
//     reasoning_content field so DeepSeek can continue chain-of-thought
// =================================================================================================

// -------------------------------------------------------------------------------------------------
// 1. Responses API request → Chat Completions request
// -------------------------------------------------------------------------------------------------

/// Convert a Responses API request into a Chat Completions request suitable for DeepSeek.
///
/// * `previous_reasoning` — when provided, injected into the last assistant message's
///   `reasoning_content` so DeepSeek can continue a chain-of-thought across turns.
pub fn convert_responses_to_chat(
    responses: &ResponsesRequest,
    previous_reasoning: Option<String>,
) -> ChatCompletionsRequest {
    let mut messages: Vec<ChatMessage> = Vec::new();

    // -- instructions → system message ------------------------------------------------
    if let Some(ref instructions) = responses.instructions {
        if !instructions.is_empty() {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(ChatContent::String(instructions.clone())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
                reasoning_content: None,
            });
        }
    }

    // -- input items → messages --------------------------------------------------------
    if let Some(ref input) = responses.input {
        for item in input {
            match item {
                ResponsesInputItem::Message {
                    role,
                    content,
                    id: _,
                    name,
                } => {
                    let mapped_role = if role == "developer" { "system" } else { role };

                    messages.push(ChatMessage {
                        role: mapped_role.to_string(),
                        content: content
                            .as_ref()
                            .map(|parts| responses_content_parts_to_chat_content(parts)),
                        name: name.clone(),
                        tool_calls: None,
                        tool_call_id: None,
                        refusal: None,
                        reasoning_content: None,
                    });
                }

                ResponsesInputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
                } => {
                    messages.push(ChatMessage {
                        role: "assistant".to_string(),
                        content: None,
                        name: None,
                        tool_calls: Some(vec![ToolCall {
                            id: call_id.clone(),
                            tool_type: "function".to_string(),
                            function: FunctionCall {
                                name: name.clone(),
                                arguments: arguments.clone(),
                            },
                            index: None,
                        }]),
                        tool_call_id: None,
                        refusal: None,
                        reasoning_content: None,
                    });
                }

                ResponsesInputItem::FunctionCallOutput {
                    call_id, output, ..
                } => {
                    messages.push(ChatMessage {
                        role: "tool".to_string(),
                        content: Some(ChatContent::String(output.clone())),
                        name: None,
                        tool_calls: None,
                        tool_call_id: Some(call_id.clone()),
                        refusal: None,
                        reasoning_content: None,
                    });
                }
            }
        }
    }

    // -- inject previous_reasoning into the last assistant message ---------------------
    if let Some(reasoning) = previous_reasoning {
        for msg in messages.iter_mut().rev() {
            if msg.role == "assistant" {
                msg.reasoning_content = Some(reasoning);
                break;
            }
        }
    }

    // -- tools -------------------------------------------------------------------------
    let chat_tools: Option<Vec<ChatTool>> = responses.tools.as_ref().map(|tools| {
        tools
            .iter()
            .filter_map(|t| {
                t.get_function().map(|f| ChatTool {
                    tool_type: "function".to_string(),
                    function: ChatFunction {
                        name: f.name,
                        description: f.description,
                        parameters: f.parameters,
                    },
                })
            })
            .collect()
    });

    // -- response_format from text.format ----------------------------------------------
    let response_format = responses.text.as_ref().and_then(|tc| {
        tc.format.as_ref().map(|f| match f {
            TextFormat::Text => ResponseFormat::Text,
            TextFormat::JsonObject => ResponseFormat::JsonObject,
            TextFormat::JsonSchema {
                name,
                schema,
                strict,
            } => ResponseFormat::JsonSchema {
                json_schema: JsonSchemaDef {
                    name: name.clone(),
                    schema: schema.clone(),
                    strict: *strict,
                },
            },
        })
    });

    // -- reasoning_effort from reasoning -----------------------------------------------
    let reasoning_effort = responses.reasoning.as_ref().and_then(|r| r.effort.clone());

    // -- logprobs / top_logprobs -------------------------------------------------------
    let (logprobs, top_logprobs) = match responses.top_logprobs {
        Some(n) if n > 0 => (Some(true), Some(n)),
        _ => (None, None),
    };

    ChatCompletionsRequest {
        model: responses.model.clone(),
        messages,
        stream: responses.stream,
        max_tokens: None,
        max_completion_tokens: responses.max_output_tokens,
        temperature: Some(responses.temperature.unwrap_or(1.0)),
        top_p: responses.top_p,
        frequency_penalty: responses.frequency_penalty,
        presence_penalty: responses.presence_penalty,
        tools: chat_tools,
        tool_choice: responses.tool_choice.clone(),
        parallel_tool_calls: responses.parallel_tool_calls,
        stop: None,
        n: None,
        seed: None,
        stream_options: if responses.stream.unwrap_or(false) {
            Some(StreamOptions {
                include_usage: Some(true),
            })
        } else {
            None
        },
        user: responses.user.clone(),
        response_format,
        logprobs,
        top_logprobs,
        reasoning_effort,
        service_tier: responses.service_tier.clone(),
        store: responses.store,
        metadata: responses.metadata.clone(),
        // DeepSeek requires explicit thinking opt-in; default to disabled
        thinking: Some(ThinkingConfig {
            thinking_type: "disabled".to_string(),
        }),
    }
}

// -------------------------------------------------------------------------------------------------
// 2. Chat Completions response → Responses API response
// -------------------------------------------------------------------------------------------------

/// Convert a Chat Completions response (from DeepSeek) into a Responses API response.
pub fn convert_chat_to_responses_response(
    chat: &ChatCompletionsResponse,
    model: &str,
) -> ResponsesResponse {
    let mut output: Vec<ResponsesOutputItem> = Vec::new();
    let mut status = "completed";
    let mut incomplete_details: Option<Value> = None;

    for choice in &chat.choices {
        // -- finish_reason → status ------------------------------------------------
        if let Some(ref finish) = choice.finish_reason {
            if finish == "length" {
                status = "incomplete";
                incomplete_details = Some(serde_json::json!({"reason": "max_output_tokens"}));
            }
        }

        if let Some(ref msg) = choice.message {
            let text = chat_content_to_string(&msg.content);

            // -- refusal -----------------------------------------------------------
            if let Some(ref refusal) = msg.refusal {
                if !refusal.is_empty() {
                    let msg_id = gen_id("msg");
                    output.push(ResponsesOutputItem::Message {
                        id: msg_id,
                        role: Some("assistant".to_string()),
                        content: vec![ResponsesContentPart::OutputText {
                            text: refusal.clone(),
                        }],
                        status: Some("completed".to_string()),
                    });
                }
            } else if !text.is_empty() || msg.tool_calls.is_none() {
                let msg_id = gen_id("msg");
                output.push(ResponsesOutputItem::Message {
                    id: msg_id,
                    role: Some("assistant".to_string()),
                    content: vec![ResponsesContentPart::OutputText { text }],
                    status: Some("completed".to_string()),
                });
            }

            // -- tool_calls → function_call output items ----------------------------
            if let Some(ref tool_calls) = msg.tool_calls {
                for tc in tool_calls {
                    output.push(ResponsesOutputItem::FunctionCall {
                        id: tc.id.clone(),
                        call_id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                        status: Some("completed".to_string()),
                    });
                }
            }
        }
    }

    // -- usage -------------------------------------------------------------------------
    let usage = chat.usage.as_ref().map(|u| {
        let output_tokens_details = u
            .completion_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens)
            .map(|rt| OutputTokensDetails {
                reasoning_tokens: Some(rt),
            });

        let input_tokens_details = u
            .prompt_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .map(|ct| InputTokensDetails {
                cached_tokens: Some(ct),
            });

        ResponsesUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
            input_tokens_details,
            output_tokens_details,
        }
    });

    ResponsesResponse {
        id: chat.id.clone(),
        object: "response".to_string(),
        output,
        status: Some(status.to_string()),
        usage,
        model: Some(model.to_string()),
        incomplete_details,
        error: None,
        created_at: Some(chat.created as i64),
        completed_at: None,
    }
}

// -------------------------------------------------------------------------------------------------
// 3. Chat Completions request → Responses API request  (reverse of #1)
// -------------------------------------------------------------------------------------------------

/// Convert a Chat Completions request into a Responses API request.
pub fn convert_chat_to_responses(chat: &ChatCompletionsRequest) -> ResponsesRequest {
    let mut input: Vec<ResponsesInputItem> = Vec::new();
    let mut instructions: Option<String> = None;

    for msg in &chat.messages {
        match msg.role.as_str() {
            "system" | "developer" => {
                let text = chat_content_to_string(&msg.content);
                if !text.is_empty() {
                    instructions = Some(text);
                }
            }

            "user" => {
                input.push(ResponsesInputItem::Message {
                    role: "user".to_string(),
                    content: chat_content_to_responses_content_parts(&msg.content),
                    id: None,
                    name: msg.name.clone(),
                });
            }

            "assistant" => {
                if let Some(ref tool_calls) = msg.tool_calls {
                    // If there's text content, emit a message item first
                    let text = chat_content_to_string(&msg.content);
                    if !text.is_empty() {
                        input.push(ResponsesInputItem::Message {
                            role: "assistant".to_string(),
                            content: Some(vec![ResponsesContentPart::OutputText { text }]),
                            id: None,
                            name: None,
                        });
                    }
                    // Emit each tool call
                    for tc in tool_calls {
                        input.push(ResponsesInputItem::FunctionCall {
                            call_id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                            id: None,
                            status: Some("completed".to_string()),
                        });
                    }
                } else {
                    input.push(ResponsesInputItem::Message {
                        role: "assistant".to_string(),
                        content: chat_content_to_responses_content_parts(&msg.content),
                        id: None,
                        name: msg.name.clone(),
                    });
                }
            }

            "tool" => {
                if let Some(ref call_id) = msg.tool_call_id {
                    input.push(ResponsesInputItem::FunctionCallOutput {
                        call_id: call_id.clone(),
                        output: chat_content_to_string(&msg.content),
                        id: None,
                        status: None,
                    });
                }
            }

            _ => {
                // Unknown roles: treat as message with the given role
                input.push(ResponsesInputItem::Message {
                    role: msg.role.clone(),
                    content: chat_content_to_responses_content_parts(&msg.content),
                    id: None,
                    name: msg.name.clone(),
                });
            }
        }
    }

    // -- text.format from response_format ----------------------------------------------
    let text = chat.response_format.as_ref().map(|rf| TextConfig {
        format: Some(match rf {
            ResponseFormat::Text => TextFormat::Text,
            ResponseFormat::JsonObject => TextFormat::JsonObject,
            ResponseFormat::JsonSchema { json_schema } => TextFormat::JsonSchema {
                name: json_schema.name.clone(),
                schema: json_schema.schema.clone(),
                strict: json_schema.strict,
            },
        }),
    });

    // -- reasoning from reasoning_effort -----------------------------------------------
    let reasoning = chat
        .reasoning_effort
        .as_ref()
        .map(|effort| ReasoningConfig {
            effort: Some(effort.clone()),
            summary: None,
        });

    // -- tools -------------------------------------------------------------------------
    let tools: Option<Vec<ResponsesTool>> = chat.tools.as_ref().map(|chat_tools| {
        chat_tools
            .iter()
            .map(|ct| ResponsesTool {
                tool_type: "function".to_string(),
                function: Some(FunctionDefinition {
                    name: ct.function.name.clone(),
                    description: ct.function.description.clone(),
                    parameters: ct.function.parameters.clone(),
                }),
                name: None,
                description: None,
                parameters: None,
                strict: None,
            })
            .collect()
    });

    ResponsesRequest {
        model: chat.model.clone(),
        input: Some(input),
        instructions,
        stream: chat.stream,
        max_output_tokens: chat.max_completion_tokens.or(chat.max_tokens),
        temperature: chat.temperature,
        top_p: chat.top_p,
        tools,
        tool_choice: chat.tool_choice.clone(),
        reasoning,
        text,
        truncation: None,
        store: chat.store,
        metadata: chat.metadata.clone(),
        previous_response_id: None,
        session_id: None,
        user: chat.user.clone(),
        service_tier: chat.service_tier.clone(),
        parallel_tool_calls: chat.parallel_tool_calls,
        top_logprobs: chat.top_logprobs,
        frequency_penalty: chat.frequency_penalty,
        presence_penalty: chat.presence_penalty,
    }
}

// -------------------------------------------------------------------------------------------------
// 4. Responses API response → Chat Completions response  (reverse of #2)
// -------------------------------------------------------------------------------------------------

/// Convert a Responses API response into a Chat Completions response.
pub fn convert_responses_to_chat_response(
    responses: &ResponsesResponse,
    model: &str,
) -> ChatCompletionsResponse {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let refusal: Option<String> = None;
    let mut finish_reason = "stop";
    let mut reasoning_content: Option<String> = None;

    for item in &responses.output {
        match item {
            ResponsesOutputItem::Message { content, .. } => {
                for part in content {
                    match part {
                        ResponsesContentPart::OutputText { text }
                        | ResponsesContentPart::Text { text } => {
                            text_parts.push(text.clone());
                        }
                        ResponsesContentPart::ToolResult { output, .. } => {
                            text_parts.push(output.clone());
                        }
                        ResponsesContentPart::ImageUrl { .. }
                        | ResponsesContentPart::InputText { .. }
                        | ResponsesContentPart::InputImage { .. }
                        | ResponsesContentPart::InputFile { .. } => {
                            // Non-text output content in a message — not treated as
                            // assistant text but preserved through as needed
                        }
                    }
                }
            }

            ResponsesOutputItem::FunctionCall {
                id,
                call_id: _,
                name,
                arguments,
                ..
            } => {
                tool_calls.push(ToolCall {
                    id: id.clone(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                    index: None,
                });
                finish_reason = "tool_calls";
            }

            ResponsesOutputItem::Reasoning {
                content, summary, ..
            } => {
                // Prefer summary (visible reasoning) over raw content
                if let Some(ref summary_parts) = summary {
                    let text = content_parts_to_text(summary_parts);
                    if !text.is_empty() {
                        reasoning_content = Some(text);
                    }
                } else if let Some(ref content_parts) = content {
                    let text = content_parts_to_text(content_parts);
                    if !text.is_empty() {
                        reasoning_content = Some(text);
                    }
                }
            }
        }
    }

    // Check status for incomplete / length
    if let Some(ref s) = responses.status {
        if s == "incomplete" {
            finish_reason = "length";
        }
    }

    let combined_text = text_parts.join("");

    let message = ChatMessage {
        role: "assistant".to_string(),
        content: if combined_text.is_empty() && tool_calls.is_empty() {
            None
        } else {
            Some(ChatContent::String(combined_text))
        },
        name: None,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id: None,
        refusal,
        reasoning_content,
    };

    // -- usage -------------------------------------------------------------------------
    let usage = responses.usage.as_ref().map(|u| {
        let completion_tokens_details = u
            .output_tokens_details
            .as_ref()
            .and_then(|d| d.reasoning_tokens)
            .map(|rt| CompletionTokensDetails {
                reasoning_tokens: Some(rt),
            });

        let prompt_tokens_details = u
            .input_tokens_details
            .as_ref()
            .and_then(|d| d.cached_tokens)
            .map(|ct| PromptTokensDetails {
                cached_tokens: Some(ct),
            });

        ChatUsage {
            prompt_tokens: u.input_tokens,
            completion_tokens: u.output_tokens,
            total_tokens: u.total_tokens,
            prompt_tokens_details,
            completion_tokens_details,
        }
    });

    ChatCompletionsResponse {
        id: responses.id.clone(),
        object: "chat.completion".to_string(),
        created: responses.created_at.unwrap_or(0) as u64,
        model: model.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: Some(message),
            delta: None,
            finish_reason: Some(finish_reason.to_string()),
            logprobs: None,
        }],
        usage,
        system_fingerprint: None,
    }
}

// =================================================================================================
// Internal helpers
// =================================================================================================

/// Convert Responses API content parts → Chat Completions content enum.
fn responses_content_parts_to_chat_content(parts: &[ResponsesContentPart]) -> ChatContent {
    if parts.is_empty() {
        return ChatContent::String(String::new());
    }

    // If there is only a single input_text / text part, return it as a simple string
    // for maximum compatibility (many providers choke on content arrays).
    let text_only: Vec<&str> = parts
        .iter()
        .filter_map(|p| match p {
            ResponsesContentPart::InputText { text }
            | ResponsesContentPart::OutputText { text }
            | ResponsesContentPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();

    let has_non_text = parts.iter().any(|p| {
        matches!(
            p,
            ResponsesContentPart::InputImage { .. }
                | ResponsesContentPart::InputFile { .. }
                | ResponsesContentPart::ImageUrl { .. }
        )
    });

    if !has_non_text && text_only.len() == 1 {
        return ChatContent::String(text_only[0].to_string());
    }

    if !has_non_text && text_only.is_empty() {
        return ChatContent::String(String::new());
    }

    // Multi-part content (e.g. vision)
    let chat_parts: Vec<ChatContentPart> = parts
        .iter()
        .filter_map(|p| match p {
            ResponsesContentPart::InputText { text }
            | ResponsesContentPart::OutputText { text }
            | ResponsesContentPart::Text { text } => {
                Some(ChatContentPart::Text { text: text.clone() })
            }
            ResponsesContentPart::InputImage {
                image_url, detail, ..
            } => {
                let url = image_url.clone().unwrap_or_default();
                if url.is_empty() {
                    None
                } else {
                    Some(ChatContentPart::ImageUrl {
                        image_url: ChatImageUrl {
                            url,
                            detail: detail.clone(),
                        },
                    })
                }
            }
            ResponsesContentPart::ImageUrl { url, detail } => Some(ChatContentPart::ImageUrl {
                image_url: ChatImageUrl {
                    url: url.clone(),
                    detail: detail.clone(),
                },
            }),
            ResponsesContentPart::InputFile { .. } => {
                // Chat Completions has no direct equivalent for file inputs;
                // skip rather than silently misrepresent.
                None
            }
            ResponsesContentPart::ToolResult { .. } => {
                // Tool results in input context are unusual; skip.
                None
            }
        })
        .collect();

    ChatContent::Parts(chat_parts)
}

/// Convert Chat Completions content → Responses API content parts for input.
fn chat_content_to_responses_content_parts(
    content: &Option<ChatContent>,
) -> Option<Vec<ResponsesContentPart>> {
    match content {
        None => None,
        Some(ChatContent::String(s)) => {
            if s.is_empty() {
                None
            } else {
                Some(vec![ResponsesContentPart::InputText { text: s.clone() }])
            }
        }
        Some(ChatContent::Parts(parts)) => {
            let converted: Vec<ResponsesContentPart> = parts
                .iter()
                .filter_map(|p| match p {
                    ChatContentPart::Text { text } => {
                        Some(ResponsesContentPart::InputText { text: text.clone() })
                    }
                    ChatContentPart::ImageUrl { image_url } => {
                        Some(ResponsesContentPart::InputImage {
                            image_url: Some(image_url.url.clone()),
                            detail: image_url.detail.clone(),
                        })
                    }
                    ChatContentPart::InputAudio { .. } => None,
                    ChatContentPart::Refusal { refusal } => Some(ResponsesContentPart::InputText {
                        text: refusal.clone(),
                    }),
                })
                .collect();

            if converted.is_empty() {
                None
            } else {
                Some(converted)
            }
        }
    }
}

/// Simple monotonic ID generator for output items when no upstream ID is available.
fn gen_id(prefix: &str) -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{}_{}", prefix, n)
}
