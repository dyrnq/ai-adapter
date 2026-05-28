use crate::types::anthropic::*;
use crate::types::responses::*;
use serde_json::Value;

use crate::translate::deepseek::anthropic as shared;

/// Convert a Responses API request to an Anthropic-compatible request
/// for Xiaomi MIMO (mimo-v2.5-pro).
///
/// Uses shared message building and sanitization from the DeepSeek module.
/// Differences from DeepSeek:
/// - No forced `thinking: disabled` (let upstream decide)
/// - Standard Anthropic auth (handled by server, not translator)
pub fn convert_responses_to_anthropic(responses: &ResponsesRequest) -> AnthropicRequest {
    let system = responses
        .instructions
        .as_ref()
        .map(|inst| AnthropicSystem::String(inst.0.clone()));

    let messages = shared::build_anthropic_messages(responses.input.as_deref().unwrap_or(&[]));

    let tools = responses.tools.as_ref().map(|ts| {
        ts.iter()
            .filter_map(|t| t.get_function())
            .map(|f| {
                let mut schema = f
                    .parameters
                    .unwrap_or_else(|| serde_json::json!({"type": "object"}));
                if let Some(obj) = schema.as_object_mut() {
                    if !obj.contains_key("type") {
                        obj.insert("type".to_string(), Value::String("object".to_string()));
                    }
                }
                AnthropicTool {
                    name: f.name,
                    description: f.description,
                    input_schema: schema,
                    cache_control: None,
                }
            })
            .collect()
    });

    let thinking = match responses
        .reasoning
        .as_ref()
        .and_then(|r| r.effort.as_deref())
    {
        Some("low") => Some(AnthropicThinkingConfig::Enabled {
            budget_tokens: 1024,
        }),
        Some("medium") => Some(AnthropicThinkingConfig::Enabled {
            budget_tokens: 4096,
        }),
        Some("high") => Some(AnthropicThinkingConfig::Enabled {
            budget_tokens: 8192,
        }),
        _ => None,
    };

    let thinking_budget = match &thinking {
        Some(AnthropicThinkingConfig::Enabled { budget_tokens }) => *budget_tokens,
        _ => 0,
    };

    let max_tokens = responses.max_output_tokens.unwrap_or(4096);
    let max_tokens = if thinking_budget > 0 && max_tokens <= thinking_budget {
        thinking_budget + 4096
    } else {
        max_tokens
    };

    AnthropicRequest {
        model: responses.model.clone(),
        messages: shared::sanitize_anthropic_messages(messages),
        system,
        max_tokens,
        metadata: responses.metadata.clone(),
        stop_sequences: None,
        stream: responses.stream,
        temperature: responses.temperature,
        top_p: responses.top_p,
        top_k: None,
        tools,
        tool_choice: shared::map_tool_choice(&responses.tool_choice),
        thinking,
    }
}

/// Convert a xiaomimimo Anthropic response back to OpenAI Responses format.
#[allow(dead_code)]
pub fn convert_anthropic_to_responses(
    anthropic: &AnthropicResponse,
    model: &str,
) -> ResponsesResponse {
    let mut output: Vec<ResponsesOutputItem> = Vec::new();

    for (i, block) in anthropic.content.iter().enumerate() {
        match block {
            AnthropicContentBlock::Text { text, .. } => {
                output.push(ResponsesOutputItem::Message {
                    id: format!("{}_{}", anthropic.id, i),
                    role: Some("assistant".to_string()),
                    content: vec![ResponsesContentPart::OutputText { text: text.clone() }],
                    status: Some("completed".to_string()),
                });
            }
            AnthropicContentBlock::ToolUse { id, name, input } => {
                let arguments = serde_json::to_string(input).unwrap_or_default();
                output.push(ResponsesOutputItem::FunctionCall {
                    id: format!("{}_{}", anthropic.id, i),
                    call_id: id.clone(),
                    name: name.clone(),
                    arguments,
                    status: Some("completed".to_string()),
                });
            }
            AnthropicContentBlock::Thinking { thinking, .. } => {
                output.push(ResponsesOutputItem::Reasoning {
                    id: format!("{}_{}", anthropic.id, i),
                    content: Some(vec![ResponsesContentPart::OutputText {
                        text: thinking.clone(),
                    }]),
                    summary: None,
                    status: Some("completed".to_string()),
                });
            }
            AnthropicContentBlock::RedactedThinking { data } => {
                output.push(ResponsesOutputItem::Reasoning {
                    id: format!("{}_{}", anthropic.id, i),
                    content: Some(vec![ResponsesContentPart::OutputText {
                        text: data.clone(),
                    }]),
                    summary: None,
                    status: Some("completed".to_string()),
                });
            }
            _ => {}
        }
    }

    let total_tokens = anthropic.usage.input_tokens + anthropic.usage.output_tokens;

    let usage = ResponsesUsage {
        input_tokens: anthropic.usage.input_tokens,
        output_tokens: anthropic.usage.output_tokens,
        total_tokens,
        input_tokens_details: Some(InputTokensDetails {
            cached_tokens: anthropic.usage.cache_read_input_tokens,
        }),
        output_tokens_details: None,
    };

    ResponsesResponse {
        id: anthropic.id.clone(),
        object: "response".to_string(),
        output,
        status: Some("completed".to_string()),
        usage: Some(usage),
        model: Some(model.to_string()),
        incomplete_details: None,
        error: None,
        created_at: None,
        completed_at: None,
    }
}
