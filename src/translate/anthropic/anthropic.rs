use crate::types::anthropic::*;
use crate::types::responses::*;

/// Convert a native Anthropic Messages API response into an OpenAI Responses API response.
///
/// This handles genuine Anthropic upstream — not DeepSeek's Anthropic-compatible endpoint
/// (which may require different mapping for thinking/content blocks).
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
            // Image, Document, and ToolResult blocks are not produced in
            // Anthropic response content — skip them silently.
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
