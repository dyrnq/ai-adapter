use crate::types::chat::*;
use crate::types::responses::*;

/// Convert Responses request to Chat Completions for xiaomimimo/mimo-v2.5-pro.
/// - thinking enabled for better tool calling
/// - injects previous_reasoning into the last assistant tool message
///   to satisfy xiaomimimo's reasoning_content requirement
pub fn convert_responses_to_chat(
    responses: &ResponsesRequest,
    previous_reasoning: Option<String>,
) -> ChatCompletionsRequest {
    let mut messages: Vec<ChatMessage> = Vec::new();

    // instructions → system message
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

    // input items → messages
    for item in responses.input.as_deref().unwrap_or(&[]) {
        match item {
            ResponsesInputItem::Message { role, content, .. } => {
                let text = extract_text(content);
                let mapped_role = if role == "developer" {
                    "system"
                } else {
                    role.as_str()
                };
                messages.push(ChatMessage {
                    role: mapped_role.to_string(),
                    content: Some(ChatContent::String(text)),
                    name: None,
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
                    // No reasoning required when thinking:disabled
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

    // Inject previous_reasoning into the last assistant tool message
    if let Some(reasoning) = previous_reasoning {
        for msg in messages.iter_mut().rev() {
            if msg.role == "assistant" && msg.tool_calls.is_some() {
                msg.reasoning_content = Some(reasoning);
                break;
            }
        }
    }

    let chat_tools = responses.tools.as_ref().map(|tools| {
        tools
            .iter()
            .filter_map(|t| t.get_function())
            .map(|f| ChatTool {
                tool_type: "function".to_string(),
                function: ChatFunction {
                    name: f.name.clone(),
                    description: f.description.clone(),
                    parameters: Some(
                        f.parameters
                            .clone()
                            .unwrap_or(serde_json::json!({"type": "object"})),
                    ),
                },
            })
            .collect()
    });

    ChatCompletionsRequest {
        model: responses.model.clone(),
        messages,
        stream: responses.stream,
        max_tokens: None,
        max_completion_tokens: responses.max_output_tokens,
        temperature: Some(1.0),
        top_p: Some(responses.top_p.unwrap_or(0.95)),
        // xiaomimimo tool calling docs explicitly use thinking:disabled
        thinking: Some(ThinkingConfig {
            thinking_type: "disabled".to_string(),
        }),
        stop: None,
        n: Some(1),
        seed: None,
        frequency_penalty: Some(0.0),
        presence_penalty: Some(0.0),
        logprobs: None,
        top_logprobs: None,
        tools: chat_tools,
        tool_choice: Some(serde_json::json!("auto")),
        user: None,
        response_format: None,
        stream_options: None,
        parallel_tool_calls: None,
        reasoning_effort: None,
        service_tier: None,
        store: None,
        metadata: None,
    }
}

fn extract_text(content: &Option<Vec<ResponsesContentPart>>) -> String {
    content
        .as_ref()
        .map(|parts| {
            parts
                .iter()
                .filter_map(|p| match p {
                    ResponsesContentPart::InputText { text }
                    | ResponsesContentPart::Text { text }
                    | ResponsesContentPart::OutputText { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}
