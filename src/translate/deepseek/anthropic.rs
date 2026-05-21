use crate::types::anthropic::*;
use crate::types::responses::*;
use serde_json::Value;

// ============================================================================
// Public API
// ============================================================================

/// Convert a Responses API request to an Anthropic Messages API request.
///
/// - Merges consecutive `FunctionCall` items into one assistant message with
///   multiple `tool_use` blocks.
/// - Merges consecutive `FunctionCallOutput` items into one user message with
///   multiple `tool_result` blocks.
/// - Maps `"developer"` role to `"assistant"`.
/// - Converts `instructions` to a system prompt (as a text block).
/// - Maps `reasoning.effort` to `AnthropicThinkingConfig` (with budget_tokens),
///   defaulting to `Disabled`.
/// - Adjusts `max_tokens` so it exceeds the thinking budget when enabled.
/// - Sanitizes: ensures the conversation ends with a user message and repairs
///   tool_use/tool_result adjacency.
/// - Converts tools via `ResponsesTool::get_function()`.
pub fn convert_responses_to_anthropic(responses: &ResponsesRequest) -> AnthropicRequest {
    // --- system prompt from instructions ---
    let system = responses.instructions.as_ref().map(|inst| {
        AnthropicSystem::Blocks(vec![AnthropicTextBlock {
            block_type: "text".to_string(),
            text: inst.clone(),
            cache_control: None,
        }])
    });

    // --- build messages from input items ---
    let messages = build_anthropic_messages(responses.input.as_deref().unwrap_or(&[]));

    // --- build tools via get_function() ---
    let tools = responses.tools.as_ref().map(|ts| {
        ts.iter()
            .filter_map(|t| t.get_function())
            .map(|f| AnthropicTool {
                name: f.name,
                description: f.description,
                input_schema: f
                    .parameters
                    .unwrap_or(Value::Object(serde_json::Map::new())),
                cache_control: None,
            })
            .collect()
    });

    // --- reasoning.effort -> AnthropicThinkingConfig (default Disabled) ---
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
        _ => Some(AnthropicThinkingConfig::Disabled),
    };

    // --- max_tokens must exceed thinking budget when enabled ---
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
        messages: sanitize_anthropic_messages(messages),
        system,
        max_tokens,
        metadata: responses.metadata.clone(),
        stop_sequences: None,
        stream: responses.stream,
        temperature: responses.temperature,
        top_p: responses.top_p,
        top_k: None,
        tools,
        tool_choice: map_tool_choice(&responses.tool_choice),
        thinking,
    }
}

fn map_tool_choice(tc: &Option<Value>) -> Option<AnthropicToolChoice> {
    match tc {
        Some(Value::String(s)) if s == "auto" => Some(AnthropicToolChoice::Auto),
        Some(Value::String(s)) if s == "required" || s == "any" => Some(AnthropicToolChoice::Any),
        Some(Value::Object(obj)) => obj
            .get("function")
            .and_then(|f| f.get("name"))
            .and_then(|n| n.as_str())
            .map(|name| AnthropicToolChoice::Tool {
                name: name.to_string(),
            }),
        _ => None,
    }
}

/// Convert an Anthropic Messages API response to a Responses API response.
///
/// - `Text` blocks -> `OutputText` content parts (consecutive texts are merged
///   into one `Message` output item).
/// - `ToolUse` blocks -> `FunctionCall` output items.
/// - `Thinking` / `RedactedThinking` blocks -> `Reasoning` output items.
/// - Usage is mapped including `cached_tokens` from both `cache_read_input_tokens`
///   and `cache_creation_input_tokens`.
pub fn convert_anthropic_to_responses(
    anthropic: &AnthropicResponse,
    model: &str,
) -> ResponsesResponse {
    let mut output: Vec<ResponsesOutputItem> = Vec::new();
    let mut item_id_counter: u32 = 0;

    for block in &anthropic.content {
        match block {
            AnthropicContentBlock::Text { text, .. } => {
                // Append to the last Message item if present, otherwise start one.
                match output.last_mut() {
                    Some(ResponsesOutputItem::Message { content, .. }) => {
                        content.push(ResponsesContentPart::OutputText { text: text.clone() });
                    }
                    _ => {
                        item_id_counter += 1;
                        output.push(ResponsesOutputItem::Message {
                            id: format!("msg_{}", item_id_counter),
                            role: Some("assistant".to_string()),
                            content: vec![ResponsesContentPart::OutputText { text: text.clone() }],
                            status: Some("completed".to_string()),
                        });
                    }
                }
            }

            AnthropicContentBlock::ToolUse { id, name, input } => {
                item_id_counter += 1;
                output.push(ResponsesOutputItem::FunctionCall {
                    id: format!("fc_{}", item_id_counter),
                    call_id: id.clone(),
                    name: name.clone(),
                    arguments: input.to_string(),
                    status: Some("completed".to_string()),
                });
            }

            AnthropicContentBlock::Thinking { thinking, .. } => {
                item_id_counter += 1;
                output.push(ResponsesOutputItem::Reasoning {
                    id: format!("rs_{}", item_id_counter),
                    content: Some(vec![ResponsesContentPart::OutputText {
                        text: thinking.clone(),
                    }]),
                    summary: None,
                    status: Some("completed".to_string()),
                });
            }

            AnthropicContentBlock::RedactedThinking { data } => {
                item_id_counter += 1;
                output.push(ResponsesOutputItem::Reasoning {
                    id: format!("rs_{}", item_id_counter),
                    content: Some(vec![ResponsesContentPart::OutputText {
                        text: format!("[redacted: {}]", data),
                    }]),
                    summary: None,
                    status: Some("completed".to_string()),
                });
            }

            // ToolResult, Image, Document are not expected in a response.
            _ => {}
        }
    }

    // If the response produced no output items at all, emit an empty message.
    if output.is_empty() {
        output.push(ResponsesOutputItem::Message {
            id: "msg_1".to_string(),
            role: Some("assistant".to_string()),
            content: vec![],
            status: Some("completed".to_string()),
        });
    }

    // --- usage with cached_tokens ---
    let cached_tokens = match (
        anthropic.usage.cache_read_input_tokens,
        anthropic.usage.cache_creation_input_tokens,
    ) {
        (Some(read), Some(create)) => Some(read + create),
        (Some(read), None) => Some(read),
        (None, Some(create)) => Some(create),
        (None, None) => None,
    };

    let usage = ResponsesUsage {
        input_tokens: anthropic.usage.input_tokens,
        output_tokens: anthropic.usage.output_tokens,
        total_tokens: anthropic.usage.input_tokens + anthropic.usage.output_tokens,
        input_tokens_details: cached_tokens.map(|c| InputTokensDetails {
            cached_tokens: Some(c),
        }),
        output_tokens_details: None,
    };

    // --- map stop_reason -> status ---
    let status = anthropic.stop_reason.as_deref().map(|r| match r {
        "end_turn" | "stop_sequence" | "tool_use" => "completed",
        "max_tokens" => "incomplete",
        _ => "completed",
    });

    ResponsesResponse {
        id: anthropic.id.clone(),
        object: "response".to_string(),
        output,
        status: status.map(|s| s.to_string()),
        usage: Some(usage),
        model: Some(model.to_string()),
        incomplete_details: None,
        error: None,
        created_at: None,
        completed_at: None,
    }
}

/// Build a `ToolUse` content block suitable for an Anthropic assistant message.
///
/// Parses `arguments` (a JSON string) into `serde_json::Value`. If parsing fails
/// the input falls back to an empty JSON object.
pub fn build_tool_use_block(
    call_id: String,
    name: String,
    arguments: String,
) -> AnthropicContentBlock {
    let input: Value =
        serde_json::from_str(&arguments).unwrap_or(Value::Object(serde_json::Map::new()));
    AnthropicContentBlock::ToolUse {
        id: call_id,
        name,
        input,
    }
}

// ============================================================================
// Internal helpers — message construction
// ============================================================================

/// Tracks consecutive FunctionCall / FunctionCallOutput items so they can be
/// merged into a single message.
enum PendingGroup {
    FunctionCalls(Vec<(String, String, String)>), // (call_id, name, arguments)
    FunctionCallOutputs(Vec<(String, String)>),   // (call_id, output)
}

/// Build a `Vec<AnthropicMessage>` from `ResponsesInputItem` items, merging
/// consecutive `FunctionCall` / `FunctionCallOutput` items as specified.
fn build_anthropic_messages(items: &[ResponsesInputItem]) -> Vec<AnthropicMessage> {
    let mut messages: Vec<AnthropicMessage> = Vec::new();
    let mut pending: Option<PendingGroup> = None;

    for item in items {
        match item {
            ResponsesInputItem::Message { role, content, .. } => {
                flush_pending(&mut pending, &mut messages);
                let mapped_role = map_role(role);
                let blocks = convert_message_content(content);
                messages.push(AnthropicMessage {
                    role: mapped_role,
                    content: AnthropicMessageContent::Blocks(blocks),
                });
            }
            ResponsesInputItem::FunctionCall {
                call_id,
                name,
                arguments,
                ..
            } => match &mut pending {
                Some(PendingGroup::FunctionCalls(calls)) => {
                    calls.push((call_id.clone(), name.clone(), arguments.clone()));
                }
                _ => {
                    flush_pending(&mut pending, &mut messages);
                    pending = Some(PendingGroup::FunctionCalls(vec![(
                        call_id.clone(),
                        name.clone(),
                        arguments.clone(),
                    )]));
                }
            },
            ResponsesInputItem::FunctionCallOutput {
                call_id, output, ..
            } => match &mut pending {
                Some(PendingGroup::FunctionCallOutputs(outputs)) => {
                    outputs.push((call_id.clone(), output.clone()));
                }
                _ => {
                    flush_pending(&mut pending, &mut messages);
                    pending = Some(PendingGroup::FunctionCallOutputs(vec![(
                        call_id.clone(),
                        output.clone(),
                    )]));
                }
            },
        }
    }
    flush_pending(&mut pending, &mut messages);

    messages
}

/// Flush the pending group into `messages` as an `AnthropicMessage`.
fn flush_pending(pending: &mut Option<PendingGroup>, messages: &mut Vec<AnthropicMessage>) {
    match pending.take() {
        Some(PendingGroup::FunctionCalls(calls)) => {
            let blocks: Vec<AnthropicContentBlock> = calls
                .into_iter()
                .map(|(id, name, args)| build_tool_use_block(id, name, args))
                .collect();
            messages.push(AnthropicMessage {
                role: "assistant".to_string(),
                content: AnthropicMessageContent::Blocks(blocks),
            });
        }
        Some(PendingGroup::FunctionCallOutputs(outputs)) => {
            let blocks: Vec<AnthropicContentBlock> = outputs
                .into_iter()
                .map(|(tool_use_id, output)| AnthropicContentBlock::ToolResult {
                    tool_use_id,
                    content: AnthropicMessageContent::String(output),
                    is_error: None,
                })
                .collect();
            messages.push(AnthropicMessage {
                role: "user".to_string(),
                content: AnthropicMessageContent::Blocks(blocks),
            });
        }
        None => {}
    }
}

// ============================================================================
// Internal helpers — role & content mapping
// ============================================================================

/// Map Responses API roles to Anthropic roles.
fn map_role(role: &str) -> String {
    match role {
        "developer" => "assistant".to_string(),
        other => other.to_string(),
    }
}

/// Convert a `ResponsesContentPart` slice into `AnthropicContentBlock` list.
fn convert_message_content(
    content: &Option<Vec<ResponsesContentPart>>,
) -> Vec<AnthropicContentBlock> {
    match content {
        Some(parts) => parts.iter().filter_map(convert_content_part).collect(),
        None => vec![],
    }
}

fn convert_content_part(part: &ResponsesContentPart) -> Option<AnthropicContentBlock> {
    match part {
        ResponsesContentPart::InputText { text }
        | ResponsesContentPart::Text { text }
        | ResponsesContentPart::OutputText { text } => Some(AnthropicContentBlock::Text {
            text: text.clone(),
            cache_control: None,
        }),
        ResponsesContentPart::InputImage { image_url, .. } => {
            image_url.as_ref().and_then(|url| convert_image_url(url))
        }
        ResponsesContentPart::ImageUrl { url, .. } => convert_image_url(url),
        // InputFile, ToolResult, OutputText in input — skip.
        _ => None,
    }
}

/// Convert a `data:` URL to an `Image` content block. Non-data URLs are skipped.
fn convert_image_url(url: &str) -> Option<AnthropicContentBlock> {
    if url.starts_with("data:") {
        let comma_pos = url.find(',')?;
        let header = &url[5..comma_pos]; // after "data:"
        let media_type = header.trim_end_matches(";base64").to_string();
        let data = url[comma_pos + 1..].to_string();
        Some(AnthropicContentBlock::Image {
            source: AnthropicImageSource {
                source_type: "base64".to_string(),
                media_type,
                data,
            },
        })
    } else {
        None
    }
}

// ============================================================================
// Internal helpers — sanitization
// ============================================================================

/// Post-process messages so the Anthropic API will accept them:
///
/// 1. Merges consecutive messages with the same role.
/// 2. Ensures the conversation ends with a `user` message.
/// 3. Ensures every assistant message containing `tool_use` blocks is followed by
///    a user message containing matching `tool_result` blocks (inserting a
///    placeholder if necessary).
fn sanitize_anthropic_messages(messages: Vec<AnthropicMessage>) -> Vec<AnthropicMessage> {
    if messages.is_empty() {
        return messages;
    }

    // --- Step 1: merge consecutive messages with the same role ---
    let mut merged: Vec<AnthropicMessage> = Vec::new();
    for msg in messages {
        if let Some(last) = merged.last_mut() {
            if last.role == msg.role {
                merge_message_into(last, msg);
            } else {
                merged.push(msg);
            }
        } else {
            merged.push(msg);
        }
    }

    // --- Step 1.5: remove messages with empty content ---
    merged.retain(|m| !is_empty_content(&m.content));

    // --- Step 2: ensure conversation ends with a user message ---
    if merged.last().map(|m| m.role.as_str()) == Some("assistant") {
        merged.push(AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicMessageContent::String("Please continue.".to_string()),
        });
    }

    // --- Step 3: repair tool_use / tool_result adjacency ---
    let mut repaired: Vec<AnthropicMessage> = Vec::new();
    let mut iter = merged.into_iter().peekable();

    while let Some(msg) = iter.next() {
        let tool_use_ids = extract_tool_use_ids(&msg);
        repaired.push(msg);

        if !tool_use_ids.is_empty() {
            // The next message *must* be a user message containing
            // tool_result blocks for every tool_use id.
            let has_valid_follow = iter.peek().is_some_and(|next| {
                next.role == "user"
                    && tool_use_ids
                        .iter()
                        .all(|id| extract_tool_result_ids(next).contains(id))
            });

            if !has_valid_follow {
                // Skip the non-conforming message if it's a user message
                // (we'll replace it), otherwise insert before whatever follows.
                if iter.peek().map(|m| m.role.as_str()) == Some("user") {
                    iter.next();
                }
                let blocks: Vec<AnthropicContentBlock> = tool_use_ids
                    .into_iter()
                    .map(|id| AnthropicContentBlock::ToolResult {
                        tool_use_id: id,
                        content: AnthropicMessageContent::String(String::new()),
                        is_error: Some(true),
                    })
                    .collect();
                repaired.push(AnthropicMessage {
                    role: "user".to_string(),
                    content: AnthropicMessageContent::Blocks(blocks),
                });
            }
        }
    }

    repaired
}

// ============================================================================
// Internal helpers — message inspection & merging
// ============================================================================

/// Merge `source` into `target`. Both are assumed to have the same role.
fn merge_message_into(target: &mut AnthropicMessage, source: AnthropicMessage) {
    match (&mut target.content, source.content) {
        (
            AnthropicMessageContent::Blocks(target_blocks),
            AnthropicMessageContent::Blocks(source_blocks),
        ) => {
            target_blocks.extend(source_blocks);
        }
        (AnthropicMessageContent::Blocks(target_blocks), AnthropicMessageContent::String(text)) => {
            target_blocks.push(AnthropicContentBlock::Text {
                text,
                cache_control: None,
            });
        }
        (
            AnthropicMessageContent::String(ref mut target_text),
            AnthropicMessageContent::String(text),
        ) => {
            target_text.push('\n');
            target_text.push_str(&text);
        }
        (
            AnthropicMessageContent::String(target_text),
            AnthropicMessageContent::Blocks(source_blocks),
        ) => {
            let mut blocks = vec![AnthropicContentBlock::Text {
                text: target_text.clone(),
                cache_control: None,
            }];
            blocks.extend(source_blocks);
            target.content = AnthropicMessageContent::Blocks(blocks);
        }
    }
}

fn is_empty_content(content: &AnthropicMessageContent) -> bool {
    match content {
        AnthropicMessageContent::String(s) => s.trim().is_empty(),
        AnthropicMessageContent::Blocks(blocks) => blocks.is_empty(),
    }
}

fn extract_tool_use_ids(msg: &AnthropicMessage) -> Vec<String> {
    match &msg.content {
        AnthropicMessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| {
                if let AnthropicContentBlock::ToolUse { id, .. } = b {
                    Some(id.clone())
                } else {
                    None
                }
            })
            .collect(),
        _ => vec![],
    }
}

fn extract_tool_result_ids(msg: &AnthropicMessage) -> Vec<String> {
    match &msg.content {
        AnthropicMessageContent::Blocks(blocks) => blocks
            .iter()
            .filter_map(|b| {
                if let AnthropicContentBlock::ToolResult { tool_use_id, .. } = b {
                    Some(tool_use_id.clone())
                } else {
                    None
                }
            })
            .collect(),
        _ => vec![],
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ------------------------------------------------------------------
    // build_tool_use_block
    // ------------------------------------------------------------------

    #[test]
    fn build_tool_use_block_valid_json() {
        let block = build_tool_use_block(
            "call_1".into(),
            "get_weather".into(),
            r#"{"city":"Berlin"}"#.into(),
        );
        match block {
            AnthropicContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call_1");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "Berlin");
            }
            _ => panic!("expected ToolUse block"),
        }
    }

    #[test]
    fn build_tool_use_block_invalid_json() {
        let block = build_tool_use_block("call_1".into(), "f".into(), "not-json".into());
        match block {
            AnthropicContentBlock::ToolUse { input, .. } => {
                assert!(input.is_object());
            }
            _ => panic!("expected ToolUse block"),
        }
    }

    // ------------------------------------------------------------------
    // convert_responses_to_anthropic — basics
    // ------------------------------------------------------------------

    #[test]
    fn instructions_become_system() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            instructions: Some("Be helpful.".into()),
            input: Some(vec![]),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        match result.system {
            Some(AnthropicSystem::Blocks(blocks)) => {
                assert_eq!(blocks[0].text, "Be helpful.");
            }
            _ => panic!("expected system blocks"),
        }
    }

    #[test]
    fn no_instructions_no_system() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![]),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        assert!(result.system.is_none());
    }

    #[test]
    fn developer_role_maps_to_assistant() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![ResponsesInputItem::Message {
                role: "developer".into(),
                content: Some(vec![ResponsesContentPart::InputText {
                    text: "You are a bot.".into(),
                }]),
                id: None,
                name: None,
            }]),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        assert_eq!(result.messages[0].role, "assistant");
    }

    #[test]
    fn user_role_passes_through() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![ResponsesInputItem::Message {
                role: "user".into(),
                content: Some(vec![ResponsesContentPart::InputText {
                    text: "Hello".into(),
                }]),
                id: None,
                name: None,
            }]),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        assert_eq!(result.messages[0].role, "user");
    }

    // ------------------------------------------------------------------
    // reasoning -> thinking config
    // ------------------------------------------------------------------

    #[test]
    fn reasoning_low_becomes_thinking_budget_1024() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![]),
            reasoning: Some(ReasoningConfig {
                effort: Some("low".into()),
                summary: None,
            }),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        match result.thinking {
            Some(AnthropicThinkingConfig::Enabled { budget_tokens }) => {
                assert_eq!(budget_tokens, 1024);
            }
            _ => panic!("expected Enabled thinking config"),
        }
    }

    #[test]
    fn reasoning_medium_becomes_budget_4096() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![]),
            reasoning: Some(ReasoningConfig {
                effort: Some("medium".into()),
                summary: None,
            }),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        match result.thinking {
            Some(AnthropicThinkingConfig::Enabled { budget_tokens }) => {
                assert_eq!(budget_tokens, 4096);
            }
            _ => panic!("expected Enabled thinking config"),
        }
    }

    #[test]
    fn reasoning_high_becomes_budget_8192() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![]),
            reasoning: Some(ReasoningConfig {
                effort: Some("high".into()),
                summary: None,
            }),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        match result.thinking {
            Some(AnthropicThinkingConfig::Enabled { budget_tokens }) => {
                assert_eq!(budget_tokens, 8192);
            }
            _ => panic!("expected Enabled thinking config"),
        }
    }

    #[test]
    fn no_reasoning_defaults_to_disabled() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![]),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        assert!(matches!(
            result.thinking,
            Some(AnthropicThinkingConfig::Disabled)
        ));
    }

    #[test]
    fn max_tokens_adjusted_when_thinking_enabled() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![]),
            max_output_tokens: Some(500), // less than budget
            reasoning: Some(ReasoningConfig {
                effort: Some("low".into()),
                summary: None,
            }),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        assert_eq!(result.max_tokens, 1024 + 4096);
    }

    // ------------------------------------------------------------------
    // merging consecutive FunctionCall / FunctionCallOutput
    // ------------------------------------------------------------------

    #[test]
    fn consecutive_function_calls_merge_into_one_assistant_message() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![
                ResponsesInputItem::FunctionCall {
                    call_id: "c1".into(),
                    name: "add".into(),
                    arguments: r#"{"a":1}"#.into(),
                    id: None,
                    status: None,
                },
                ResponsesInputItem::FunctionCall {
                    call_id: "c2".into(),
                    name: "sub".into(),
                    arguments: r#"{"a":2}"#.into(),
                    id: None,
                    status: None,
                },
            ]),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        // One assistant message with two tool_use blocks, plus trailing user
        assert_eq!(result.messages.len(), 2);
        assert_eq!(result.messages[0].role, "assistant");
        if let AnthropicMessageContent::Blocks(ref blocks) = result.messages[0].content {
            assert_eq!(blocks.len(), 2);
            assert!(matches!(blocks[0], AnthropicContentBlock::ToolUse { .. }));
            assert!(matches!(blocks[1], AnthropicContentBlock::ToolUse { .. }));
        } else {
            panic!("expected blocks");
        }
    }

    #[test]
    fn consecutive_function_call_outputs_merge_into_one_user_message() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![
                ResponsesInputItem::FunctionCallOutput {
                    call_id: "c1".into(),
                    output: "result1".into(),
                    id: None,
                    status: None,
                },
                ResponsesInputItem::FunctionCallOutput {
                    call_id: "c2".into(),
                    output: "result2".into(),
                    id: None,
                    status: None,
                },
            ]),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        assert_eq!(result.messages[0].role, "user"); // not "assistant" first
        if let AnthropicMessageContent::Blocks(ref blocks) = result.messages[0].content {
            assert_eq!(blocks.len(), 2);
            assert!(matches!(
                blocks[0],
                AnthropicContentBlock::ToolResult { .. }
            ));
            assert!(matches!(
                blocks[1],
                AnthropicContentBlock::ToolResult { .. }
            ));
        } else {
            panic!("expected blocks");
        }
    }

    // ------------------------------------------------------------------
    // sanitization — ends with user
    // ------------------------------------------------------------------

    #[test]
    fn appends_user_message_when_ends_with_assistant() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![ResponsesInputItem::Message {
                role: "assistant".into(),
                content: Some(vec![ResponsesContentPart::OutputText {
                    text: "done".into(),
                }]),
                id: None,
                name: None,
            }]),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        assert_eq!(result.messages.last().unwrap().role, "user");
    }

    // ------------------------------------------------------------------
    // tools conversion
    // ------------------------------------------------------------------

    #[test]
    fn tools_converted_via_get_function() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![]),
            tools: Some(vec![ResponsesTool {
                tool_type: "function".into(),
                function: Some(FunctionDefinition {
                    name: "search".into(),
                    description: Some("Search the web".into()),
                    parameters: Some(json!({"type": "object", "properties": {}})),
                }),
                name: None,
                description: None,
                parameters: None,
                strict: None,
            }]),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);
        let tools = result.tools.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "search");
        assert_eq!(tools[0].description.as_deref(), Some("Search the web"));
    }

    // ------------------------------------------------------------------
    // convert_anthropic_to_responses
    // ------------------------------------------------------------------

    #[test]
    fn text_block_becomes_output_text() {
        let resp = AnthropicResponse {
            id: "msg_1".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            content: vec![AnthropicContentBlock::Text {
                text: "Hello!".into(),
                cache_control: None,
            }],
            model: "claude-sonnet-4-20250514".into(),
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
        };
        let result = convert_anthropic_to_responses(&resp, "deepseek");
        assert_eq!(result.output.len(), 1);
        if let ResponsesOutputItem::Message { content, role, .. } = &result.output[0] {
            assert_eq!(role.as_deref(), Some("assistant"));
            assert!(matches!(
                content[0],
                ResponsesContentPart::OutputText { .. }
            ));
        } else {
            panic!("expected Message");
        }
    }

    #[test]
    fn consecutive_text_blocks_merge_into_single_message() {
        let resp = AnthropicResponse {
            id: "msg_1".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            content: vec![
                AnthropicContentBlock::Text {
                    text: "Hello".into(),
                    cache_control: None,
                },
                AnthropicContentBlock::Text {
                    text: " world".into(),
                    cache_control: None,
                },
            ],
            model: "claude-sonnet-4-20250514".into(),
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
        };
        let result = convert_anthropic_to_responses(&resp, "deepseek");
        assert_eq!(result.output.len(), 1);
        if let ResponsesOutputItem::Message { content, .. } = &result.output[0] {
            assert_eq!(content.len(), 2);
        } else {
            panic!("expected single Message");
        }
    }

    #[test]
    fn tool_use_becomes_function_call() {
        let resp = AnthropicResponse {
            id: "msg_1".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            content: vec![AnthropicContentBlock::ToolUse {
                id: "toolu_1".into(),
                name: "get_weather".into(),
                input: json!({"city": "Berlin"}),
            }],
            model: "claude-sonnet-4-20250514".into(),
            stop_reason: Some("tool_use".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
        };
        let result = convert_anthropic_to_responses(&resp, "deepseek");
        assert_eq!(result.output.len(), 1);
        if let ResponsesOutputItem::FunctionCall {
            call_id,
            name,
            arguments,
            ..
        } = &result.output[0]
        {
            assert_eq!(call_id, "toolu_1");
            assert_eq!(name, "get_weather");
            let parsed: Value = serde_json::from_str(arguments).unwrap();
            assert_eq!(parsed["city"], "Berlin");
        } else {
            panic!("expected FunctionCall");
        }
    }

    #[test]
    fn thinking_becomes_reasoning() {
        let resp = AnthropicResponse {
            id: "msg_1".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            content: vec![AnthropicContentBlock::Thinking {
                thinking: "Let me think...".into(),
                signature: Some("sig".into()),
            }],
            model: "claude-sonnet-4-20250514".into(),
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
        };
        let result = convert_anthropic_to_responses(&resp, "deepseek");
        assert_eq!(result.output.len(), 1);
        match &result.output[0] {
            ResponsesOutputItem::Reasoning { content, .. } => {
                assert_eq!(content.as_ref().unwrap()[0].to_text(), "Let me think...");
            }
            _ => panic!("expected Reasoning item"),
        }
    }

    #[test]
    fn redacted_thinking_becomes_reasoning() {
        let resp = AnthropicResponse {
            id: "msg_1".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            content: vec![AnthropicContentBlock::RedactedThinking { data: "xyz".into() }],
            model: "claude-sonnet-4-20250514".into(),
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 10,
                output_tokens: 5,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
        };
        let result = convert_anthropic_to_responses(&resp, "deepseek");
        match &result.output[0] {
            ResponsesOutputItem::Reasoning { content, .. } => {
                assert!(content.as_ref().unwrap()[0]
                    .to_text()
                    .starts_with("[redacted:"));
            }
            _ => panic!("expected Reasoning item"),
        }
    }

    #[test]
    fn usage_includes_cached_tokens() {
        let resp = AnthropicResponse {
            id: "msg_1".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            content: vec![],
            model: "claude-sonnet-4-20250514".into(),
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_read_input_tokens: Some(20),
                cache_creation_input_tokens: Some(10),
            },
        };
        let result = convert_anthropic_to_responses(&resp, "deepseek");
        let u = result.usage.unwrap();
        assert_eq!(u.input_tokens, 100);
        assert_eq!(u.output_tokens, 50);
        assert_eq!(u.total_tokens, 150);
        let details = u.input_tokens_details.unwrap();
        assert_eq!(details.cached_tokens, Some(30));
    }

    #[test]
    fn stop_reason_max_tokens_becomes_incomplete() {
        let resp = AnthropicResponse {
            id: "msg_1".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            content: vec![],
            model: "claude-sonnet-4-20250514".into(),
            stop_reason: Some("max_tokens".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 1,
                output_tokens: 1,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
        };
        let result = convert_anthropic_to_responses(&resp, "deepseek");
        assert_eq!(result.status.as_deref(), Some("incomplete"));
    }

    #[test]
    fn empty_content_produces_empty_message() {
        let resp = AnthropicResponse {
            id: "msg_1".into(),
            response_type: "message".into(),
            role: "assistant".into(),
            content: vec![],
            model: "claude-sonnet-4-20250514".into(),
            stop_reason: Some("end_turn".into()),
            stop_sequence: None,
            usage: AnthropicUsage {
                input_tokens: 1,
                output_tokens: 1,
                cache_read_input_tokens: None,
                cache_creation_input_tokens: None,
            },
        };
        let result = convert_anthropic_to_responses(&resp, "deepseek");
        assert_eq!(result.output.len(), 1);
    }

    // ------------------------------------------------------------------
    // sanitization — tool_use/tool_result adjacency
    // ------------------------------------------------------------------

    #[test]
    fn tool_use_without_tool_result_gets_placeholder() {
        let req = ResponsesRequest {
            model: "deepseek".into(),
            input: Some(vec![
                ResponsesInputItem::Message {
                    role: "user".into(),
                    content: Some(vec![ResponsesContentPart::InputText { text: "go".into() }]),
                    id: None,
                    name: None,
                },
                ResponsesInputItem::FunctionCall {
                    call_id: "c1".into(),
                    name: "f".into(),
                    arguments: "{}".into(),
                    id: None,
                    status: None,
                },
                // No FunctionCallOutput following — sanitization should insert one.
                ResponsesInputItem::Message {
                    role: "user".into(),
                    content: Some(vec![ResponsesContentPart::InputText {
                        text: "next turn".into(),
                    }]),
                    id: None,
                    name: None,
                },
            ]),
            ..default_responses_request()
        };
        let result = convert_responses_to_anthropic(&req);

        // Find the assistant message with tool_use
        let tool_use_pos = result
            .messages
            .iter()
            .position(|m| {
                m.role == "assistant"
                    && matches!(&m.content, AnthropicMessageContent::Blocks(blocks)
                        if blocks.iter().any(|b| matches!(b, AnthropicContentBlock::ToolUse { .. })))
            })
            .unwrap();

        // The next message should be a user message with tool_result
        let next = &result.messages[tool_use_pos + 1];
        assert_eq!(next.role, "user");
        assert!(matches!(
            &next.content,
            AnthropicMessageContent::Blocks(blocks)
            if blocks.iter().any(|b| matches!(b, AnthropicContentBlock::ToolResult { .. }))
        ));
    }

    // ------------------------------------------------------------------
    // helper
    // ------------------------------------------------------------------

    /// Minimal ResponsesRequest with mostly-empty fields for DRY tests.
    fn default_responses_request() -> ResponsesRequest {
        ResponsesRequest {
            model: "deepseek".into(),
            input: None,
            instructions: None,
            stream: None,
            max_output_tokens: None,
            temperature: None,
            top_p: None,
            tools: None,
            tool_choice: None,
            reasoning: None,
            text: None,
            truncation: None,
            store: None,
            metadata: None,
            previous_response_id: None,
            session_id: None,
            user: None,
            service_tier: None,
            parallel_tool_calls: None,
            top_logprobs: None,
            frequency_penalty: None,
            presence_penalty: None,
        }
    }

    /// Quick text extraction for test assertions.
    trait TextExtract {
        fn to_text(&self) -> &str;
    }

    impl TextExtract for ResponsesContentPart {
        fn to_text(&self) -> &str {
            match self {
                ResponsesContentPart::OutputText { text }
                | ResponsesContentPart::InputText { text }
                | ResponsesContentPart::Text { text } => text.as_str(),
                ResponsesContentPart::ToolResult { output, .. } => output.as_str(),
                _ => "",
            }
        }
    }
}
