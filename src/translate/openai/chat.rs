use crate::types::chat::*;
use crate::types::responses::*;

// ---------------------------------------------------------------------------
// ResponsesRequest -> ChatCompletionsRequest
// ---------------------------------------------------------------------------

/// Convert an OpenAI Responses API request into a Chat Completions request.
///
/// * `instructions` becomes a leading `system` message.
/// * Developer-role messages are preserved as-is (OpenAI Chat Completions
///   supports the `developer` role).
/// * `thinking` is always set to `None`.
/// * Tools use `ResponsesTool::get_function()` for the nested/flat format.
#[allow(dead_code)]
pub fn convert_responses_to_chat(responses: &ResponsesRequest) -> ChatCompletionsRequest {
    let mut messages = Vec::new();

    // instructions -> leading system message
    if let Some(ref instructions) = responses.instructions {
        if !instructions.0.is_empty() {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: Some(ChatContent::String(instructions.0.clone())),
                name: None,
                tool_calls: None,
                tool_call_id: None,
                refusal: None,
                reasoning_content: None,
            });
        }
    }

    // input items -> messages
    if let Some(ref input_items) = responses.input {
        for item in input_items {
            match item {
                ResponsesInputItem::Message {
                    role,
                    content,
                    name,
                    ..
                } => {
                    messages.push(ChatMessage {
                        role: role.clone(),
                        content: responses_content_to_chat_content(content),
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

    // tools via get_function()
    let tools = responses.tools.as_ref().map(|ts| {
        ts.iter()
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

    // text.format -> response_format
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

    ChatCompletionsRequest {
        model: responses.model.clone(),
        messages,
        stream: responses.stream,
        max_tokens: None,
        max_completion_tokens: responses.max_output_tokens,
        temperature: Some(responses.temperature.unwrap_or(0.0)),
        top_p: responses.top_p,
        frequency_penalty: responses.frequency_penalty,
        presence_penalty: responses.presence_penalty,
        tools: tools.clone(),
        tool_choice: if tools.as_ref().is_some_and(|t| !t.is_empty())
            && responses.tool_choice.as_ref().is_none_or(|v| v == "auto")
        {
            Some(serde_json::json!("required"))
        } else {
            responses.tool_choice.clone()
        },
        parallel_tool_calls: responses.parallel_tool_calls,
        stop: None,
        n: None,
        seed: None,
        stream_options: None,
        user: responses.user.clone(),
        response_format,
        logprobs: None,
        top_logprobs: responses.top_logprobs,
        reasoning_effort: responses.reasoning.as_ref().and_then(|r| r.effort.clone()),
        service_tier: responses.service_tier.clone(),
        store: responses.store,
        metadata: responses.metadata.clone(),
        thinking: None,
    }
}

// ---------------------------------------------------------------------------
// ChatCompletionsResponse -> ResponsesResponse
// ---------------------------------------------------------------------------

/// Convert a Chat Completions response into a Responses API response.
///
/// Assistant text content becomes an `output` `Message` item.  Tool calls
/// become separate `function_call` output items.
#[allow(dead_code)]
pub fn convert_chat_to_responses_response(
    chat: &ChatCompletionsResponse,
    model: &str,
) -> ResponsesResponse {
    let mut output = Vec::new();

    for choice in &chat.choices {
        let finish_status = map_finish_reason_to_status(choice.finish_reason.as_deref());

        if let Some(ref message) = choice.message {
            // text content -> message output item
            let text = chat_content_to_string(&message.content);
            if !text.is_empty() {
                output.push(ResponsesOutputItem::Message {
                    id: chat.id.clone(),
                    role: Some(message.role.clone()),
                    content: vec![ResponsesContentPart::OutputText { text }],
                    status: Some(finish_status.clone()),
                });
            }

            // tool calls -> function_call output items
            if let Some(ref tool_calls) = message.tool_calls {
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

    // usage mapping
    let usage = chat.usage.as_ref().map(|u| ResponsesUsage {
        input_tokens: u.prompt_tokens,
        output_tokens: u.completion_tokens,
        total_tokens: u.total_tokens,
        input_tokens_details: u
            .prompt_tokens_details
            .as_ref()
            .map(|d| InputTokensDetails {
                cached_tokens: d.cached_tokens,
            }),
        output_tokens_details: u
            .completion_tokens_details
            .as_ref()
            .map(|d| OutputTokensDetails {
                reasoning_tokens: d.reasoning_tokens,
            }),
    });

    ResponsesResponse {
        id: chat.id.clone(),
        object: "response".to_string(),
        output,
        status: Some("completed".to_string()),
        usage,
        model: Some(model.to_string()),
        incomplete_details: None,
        error: None,
        created_at: Some(chat.created as i64),
        completed_at: None,
    }
}

// ---------------------------------------------------------------------------
// ChatCompletionsRequest -> ResponsesRequest
// ---------------------------------------------------------------------------

/// Convert a Chat Completions request into a Responses API request.
///
/// The first `system` message becomes `instructions`.  Subsequent system
/// messages and `developer` messages are kept as input items.
#[allow(dead_code)]
pub fn convert_chat_to_responses(chat: &ChatCompletionsRequest) -> ResponsesRequest {
    let mut input_items = Vec::new();
    let mut instructions: Option<String> = None;
    let mut seen_first_system = false;

    for msg in &chat.messages {
        match msg.role.as_str() {
            "system" => {
                if !seen_first_system && instructions.is_none() {
                    instructions = Some(chat_content_to_string(&msg.content));
                    seen_first_system = true;
                } else {
                    input_items.push(chat_message_to_input_item(msg));
                }
            }
            "developer" => {
                input_items.push(chat_message_to_input_item(msg));
            }
            "assistant" => {
                let text = chat_content_to_string(&msg.content);
                let parts = chat_content_to_responses_parts(&msg.content);

                // Push a Message item only when there is text content
                if !text.is_empty() {
                    input_items.push(ResponsesInputItem::Message {
                        role: "assistant".to_string(),
                        content: if parts.is_empty() { None } else { Some(parts) },
                        id: None,
                        name: msg.name.clone(),
                    });
                }

                // Push FunctionCall items for tool calls
                if let Some(ref tool_calls) = msg.tool_calls {
                    for tc in tool_calls {
                        input_items.push(ResponsesInputItem::FunctionCall {
                            call_id: tc.id.clone(),
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                            id: None,
                            status: None,
                        });
                    }
                }
            }
            "tool" => {
                let output = chat_content_to_string(&msg.content);
                input_items.push(ResponsesInputItem::FunctionCallOutput {
                    call_id: msg.tool_call_id.clone().unwrap_or_default(),
                    output,
                    id: None,
                    status: None,
                });
            }
            _ => {
                // user and any unknown role -> message input item
                input_items.push(chat_message_to_input_item(msg));
            }
        }
    }

    // tools
    let tools = chat.tools.as_ref().map(|ts| {
        ts.iter()
            .map(|ct| ResponsesTool {
                tool_type: ct.tool_type.clone(),
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

    // response_format -> text.format
    let text = chat.response_format.as_ref().map(|rf| {
        let format = match rf {
            ResponseFormat::Text => TextFormat::Text,
            ResponseFormat::JsonObject => TextFormat::JsonObject,
            ResponseFormat::JsonSchema { json_schema } => TextFormat::JsonSchema {
                name: json_schema.name.clone(),
                schema: json_schema.schema.clone(),
                strict: json_schema.strict,
            },
        };
        TextConfig {
            format: Some(format),
        }
    });

    ResponsesRequest {
        model: chat.model.clone(),
        input: Some(input_items),
        instructions: instructions.map(Instructions),
        stream: chat.stream,
        max_output_tokens: chat.max_completion_tokens.or(chat.max_tokens),
        temperature: chat.temperature,
        top_p: chat.top_p,
        tools,
        tool_choice: chat.tool_choice.clone(),
        reasoning: chat.reasoning_effort.as_ref().map(|e| ReasoningConfig {
            effort: Some(e.clone()),
            summary: None,
        }),
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

// ---------------------------------------------------------------------------
// ResponsesResponse -> ChatCompletionsResponse
// ---------------------------------------------------------------------------

/// Convert a Responses API response into a Chat Completions response.
///
/// All output items are aggregated into a single `assistant` choice: text
/// from `Message` items, tool calls from `function_call` items, and
/// reasoning text from `reasoning` items.
#[allow(dead_code)]
pub fn convert_responses_to_chat_response(
    responses: &ResponsesResponse,
    model: &str,
) -> ChatCompletionsResponse {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut reasoning_text = String::new();
    let mut role = "assistant".to_string();
    let mut finish_reason = "stop".to_string();

    for item in &responses.output {
        match item {
            ResponsesOutputItem::Message {
                role: item_role,
                content,
                status,
                ..
            } => {
                if let Some(r) = item_role {
                    role = r.clone();
                }
                for part in content {
                    if let Some(t) = extract_text_from_content_part(part) {
                        text_parts.push(t);
                    }
                }
                if let Some(s) = status {
                    finish_reason = map_status_to_finish_reason(s);
                }
            }
            ResponsesOutputItem::FunctionCall {
                id: _,
                call_id,
                name,
                arguments,
                ..
            } => {
                tool_calls.push(ToolCall {
                    id: call_id.clone(),
                    tool_type: "function".to_string(),
                    function: FunctionCall {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                    index: None,
                });
            }
            ResponsesOutputItem::Reasoning { content, .. } => {
                if let Some(parts) = content {
                    for part in parts {
                        if let ResponsesContentPart::Text { text } = part {
                            reasoning_text.push_str(text);
                        } else if let ResponsesContentPart::OutputText { text } = part {
                            reasoning_text.push_str(text);
                        }
                    }
                }
            }
        }
    }

    // build chat content
    let chat_content = if text_parts.is_empty() {
        None
    } else if text_parts.len() == 1 {
        Some(ChatContent::String(text_parts.into_iter().next().unwrap()))
    } else {
        Some(ChatContent::Parts(
            text_parts
                .into_iter()
                .map(|t| ChatContentPart::Text { text: t })
                .collect(),
        ))
    };

    let message = ChatMessage {
        role,
        content: chat_content,
        name: None,
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id: None,
        refusal: None,
        reasoning_content: if reasoning_text.is_empty() {
            None
        } else {
            Some(reasoning_text)
        },
    };

    let choice = ChatChoice {
        index: 0,
        message: Some(message),
        delta: None,
        finish_reason: Some(finish_reason),
        logprobs: None,
    };

    let usage = responses.usage.as_ref().map(|u| ChatUsage {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        total_tokens: u.total_tokens,
        prompt_tokens_details: u
            .input_tokens_details
            .as_ref()
            .map(|d| PromptTokensDetails {
                cached_tokens: d.cached_tokens,
            }),
        completion_tokens_details: u.output_tokens_details.as_ref().map(|d| {
            CompletionTokensDetails {
                reasoning_tokens: d.reasoning_tokens,
            }
        }),
    });

    ChatCompletionsResponse {
        id: responses.id.clone(),
        object: "chat.completion".to_string(),
        created: responses.created_at.unwrap_or(0) as u64,
        model: model.to_string(),
        choices: vec![choice],
        usage,
        system_fingerprint: None,
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Convert Responses content parts into Chat content.
#[allow(dead_code)]
fn responses_content_to_chat_content(
    content: &Option<Vec<ResponsesContentPart>>,
) -> Option<ChatContent> {
    let parts = content.as_ref()?;
    if parts.is_empty() {
        return None;
    }

    // Optimisation: single InputText part -> ChatContent::String
    if parts.len() == 1 {
        if let ResponsesContentPart::InputText { text } = &parts[0] {
            return Some(ChatContent::String(text.clone()));
        }
    }

    let chat_parts: Vec<ChatContentPart> = parts
        .iter()
        .filter_map(|p| match p {
            ResponsesContentPart::InputText { text } => {
                Some(ChatContentPart::Text { text: text.clone() })
            }
            ResponsesContentPart::InputImage { image_url, detail } => {
                Some(ChatContentPart::ImageUrl {
                    image_url: ChatImageUrl {
                        url: image_url.clone().unwrap_or_default(),
                        detail: detail.clone(),
                    },
                })
            }
            _ => None,
        })
        .collect();

    if chat_parts.is_empty() {
        None
    } else if chat_parts.len() == 1 {
        match &chat_parts[0] {
            ChatContentPart::Text { text } => Some(ChatContent::String(text.clone())),
            _ => Some(ChatContent::Parts(chat_parts)),
        }
    } else {
        Some(ChatContent::Parts(chat_parts))
    }
}

/// Convert a Chat message into a Responses input item (Message variant).
#[allow(dead_code)]
fn chat_message_to_input_item(msg: &ChatMessage) -> ResponsesInputItem {
    let parts = chat_content_to_responses_parts(&msg.content);
    ResponsesInputItem::Message {
        role: msg.role.clone(),
        content: if parts.is_empty() { None } else { Some(parts) },
        id: None,
        name: msg.name.clone(),
    }
}

/// Convert Chat content into Responses content parts (for input).
#[allow(dead_code)]
fn chat_content_to_responses_parts(content: &Option<ChatContent>) -> Vec<ResponsesContentPart> {
    match content {
        Some(ChatContent::String(s)) => {
            vec![ResponsesContentPart::InputText { text: s.clone() }]
        }
        Some(ChatContent::Parts(parts)) => parts
            .iter()
            .filter_map(|p| match p {
                ChatContentPart::Text { text } => {
                    Some(ResponsesContentPart::InputText { text: text.clone() })
                }
                ChatContentPart::ImageUrl { image_url } => Some(ResponsesContentPart::InputImage {
                    image_url: Some(image_url.url.clone()),
                    detail: image_url.detail.clone(),
                }),
                _ => None,
            })
            .collect(),
        None => vec![],
    }
}

/// Extract a text string from any Responses content part variant.
#[allow(dead_code)]
fn extract_text_from_content_part(part: &ResponsesContentPart) -> Option<String> {
    match part {
        ResponsesContentPart::InputText { text }
        | ResponsesContentPart::OutputText { text }
        | ResponsesContentPart::Text { text } => Some(text.clone()),
        ResponsesContentPart::ToolResult { output, .. } => Some(output.clone()),
        _ => None,
    }
}

#[allow(dead_code)]
fn map_finish_reason_to_status(reason: Option<&str>) -> String {
    match reason {
        Some("stop") | Some("tool_calls") => "completed",
        Some("length") | Some("content_filter") => "incomplete",
        _ => "completed",
    }
    .to_string()
}

#[allow(dead_code)]
fn map_status_to_finish_reason(status: &str) -> String {
    match status {
        "completed" => "stop",
        "incomplete" => "length",
        _ => "stop",
    }
    .to_string()
}

pub(crate) fn extract_xml_tool_calls(content: &str) -> (String, Vec<ToolCall>) {
    let mut cleaned = content.to_string();
    let mut tool_calls = Vec::new();
    let mut idx = 0u32;

    // Try <function-call>...</function-call> first, then <tools>...</tools>
    for (tag, tag_len) in [("<function-call>", 15), ("<tools>", 7)] {
        let close_tag = if tag_len == 15 {
            "</function-call>"
        } else {
            "</tools>"
        };
        let close_len = close_tag.len();
        loop {
            let rest = cleaned.clone();
            let Some(start) = rest.find(tag) else {
                break;
            };
            let Some(end) = rest[start..].find(close_tag) else {
                break;
            };
            let inner = rest[start + tag_len..start + end].trim();
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(inner) {
                // Skip <tools> blocks that echo the schema (contain "type":)
                if tag_len == 7 && parsed.get("type").is_some() {
                    cleaned.replace_range(start..start + end + close_len, "");
                    continue;
                }
                let call_id = format!("call_{}", uuid::Uuid::new_v4());
                let name = parsed
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                let arguments = parsed
                    .get("arguments")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "{}".to_string());
                tool_calls.push(ToolCall {
                    id: call_id.clone(),
                    tool_type: "function".to_string(),
                    function: FunctionCall { name, arguments },
                    index: Some(idx),
                });
                idx += 1;
            }
            cleaned.replace_range(start..start + end + close_len, "");
        }
    }

    cleaned = cleaned.trim().to_string();
    (cleaned, tool_calls)
}

#[cfg(test)]
mod tests_xml {
    use super::*;

    #[test]
    fn test_extract_xml_tool_calls_single() {
        let input = concat!(
            "<tools>\n",
            r#"{"type": "function", "function": {"name": "get_weather"}}"#,
            "\n</tools>\n\n",
            "<function-call>\n",
            r#"{"name": "get_weather", "arguments": {"city": "Beijing"}}"#,
            "\n</function-call>"
        );
        let (cleaned, tcs) = extract_xml_tool_calls(input);
        assert_eq!(cleaned, "");
        assert_eq!(tcs.len(), 1);
        assert_eq!(tcs[0].function.name, "get_weather");
        let args: serde_json::Value = serde_json::from_str(&tcs[0].function.arguments).unwrap();
        assert_eq!(args["city"], "Beijing");
    }

    #[test]
    fn test_no_xml() {
        let input = "Hello, how can I help?";
        let (cleaned, tcs) = extract_xml_tool_calls(input);
        assert_eq!(cleaned, "Hello, how can I help?");
        assert!(tcs.is_empty());
    }
}
