use crate::types::anthropic::AnthropicStreamEvent;
use crate::types::chat::{ChatContent, ChatDelta, FunctionCallDelta, ToolCallDelta};
use crate::types::responses::{
    ResponsesContentPart, ResponsesOutputItem, ResponsesResponse, ResponsesStreamEvent,
    ResponsesUsage,
};
use std::collections::HashMap;
use uuid::Uuid;

/// Parsed SSE event
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Parse an SSE byte stream line by line into structured events
#[allow(dead_code)]
pub fn parse_sse_line(line: &str) -> Option<SseEvent> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }
    if let Some(stripped) = line.strip_prefix("event:") {
        let event_type = stripped.trim().to_string();
        Some(SseEvent {
            event: Some(event_type),
            data: String::new(),
        })
    } else if let Some(stripped) = line.strip_prefix("data:") {
        let data = if line.len() > 5 {
            stripped.trim().to_string()
        } else {
            String::new()
        };
        Some(SseEvent { event: None, data })
    } else {
        None
    }
}

pub fn event_type_str(event: &ResponsesStreamEvent) -> &str {
    match event {
        ResponsesStreamEvent::ResponseCreated { .. } => "response.created",
        ResponsesStreamEvent::ResponseInProgress { .. } => "response.in_progress",
        ResponsesStreamEvent::OutputItemAdded { .. } => "response.output_item.added",
        ResponsesStreamEvent::ContentPartAdded { .. } => "response.content_part.added",
        ResponsesStreamEvent::OutputTextDelta { .. } => "response.output_text.delta",
        ResponsesStreamEvent::RefusalDelta { .. } => "response.refusal.delta",
        ResponsesStreamEvent::FunctionCallArgsDelta { .. } => {
            "response.function_call_arguments.delta"
        }
        ResponsesStreamEvent::FunctionCallArgsDone { .. } => {
            "response.function_call_arguments.done"
        }
        ResponsesStreamEvent::ReasoningTextDelta { .. } => "response.reasoning_text.delta",
        ResponsesStreamEvent::OutputItemDone { .. } => "response.output_item.done",
        ResponsesStreamEvent::ContentPartDone { .. } => "response.content_part.done",
        ResponsesStreamEvent::OutputTextDone { .. } => "response.output_text.done",
        ResponsesStreamEvent::ResponseCompleted { .. } => "response.completed",
        ResponsesStreamEvent::Error { .. } => "error",
    }
}

// ============================================================
// Anthropic SSE -> Responses SSE converter
// ============================================================

#[allow(dead_code)]
pub struct AnthropicStreamTranslator {
    pub response_id: String,
    model: String,
    output_index: u32,
    content_index: u32,
    input_tokens: u32,
    output_tokens: u32,
    cache_read_tokens: Option<u32>,
    cache_creation_tokens: Option<u32>,
    current_text_content: String,
    current_tool_name: String,
    current_tool_id: String,
    current_tool_arguments: String,
    current_reasoning: String,
    current_block_type: Option<String>,
    blocks: Vec<ResponsesOutputItem>,
    msg_id: String,
    item_id: String,
    pub started: bool,
    has_tool_call: bool,
    seq: u32,
    pub event_completed: bool,
    created: i64,
    pub reasoning_content: String,
}

impl AnthropicStreamTranslator {
    pub fn new(model: &str) -> Self {
        Self {
            response_id: format!("resp_{}", Uuid::new_v4()),
            model: model.to_string(),
            output_index: 0,
            content_index: 0,
            input_tokens: 0,
            output_tokens: 0,
            cache_read_tokens: None,
            cache_creation_tokens: None,
            current_text_content: String::new(),
            current_tool_name: String::new(),
            current_tool_id: String::new(),
            current_tool_arguments: String::new(),
            current_reasoning: String::new(),
            current_block_type: None,
            blocks: Vec::new(),
            msg_id: String::new(),
            item_id: String::new(),
            started: false,
            has_tool_call: false,
            seq: 0,
            event_completed: false,
            created: Self::now_unix(),
            reasoning_content: String::new(),
        }
    }

    fn now_unix() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }

    fn next_seq(&mut self) -> u32 {
        let s = self.seq;
        self.seq += 1;
        s
    }

    pub fn process_event(&mut self, event: &AnthropicStreamEvent) -> Vec<ResponsesStreamEvent> {
        let mut events = Vec::new();

        match event {
            AnthropicStreamEvent::MessageStart { message } => {
                self.input_tokens = message.usage.input_tokens;
                self.cache_read_tokens = message.usage.cache_read_input_tokens;
                self.cache_creation_tokens = message.usage.cache_creation_input_tokens;
                self.model = message.model.clone();

                if !self.started {
                    self.started = true;
                    let resp = self.make_response();
                    events.push(ResponsesStreamEvent::ResponseCreated {
                        response: resp,
                        sequence_number: self.next_seq(),
                    });
                    events.push(ResponsesStreamEvent::ResponseInProgress {
                        response: self.make_response(),
                        sequence_number: self.next_seq(),
                    });
                }
            }

            AnthropicStreamEvent::ContentBlockStart {
                index,
                content_block,
            } => {
                // Flush previous text content if any
                self.flush_text_if_needed(&mut events);

                match content_block {
                    crate::types::anthropic::AnthropicContentBlock::Text { .. } => {
                        self.current_block_type = Some("text".to_string());
                        self.output_index = *index;
                        self.content_index = 0;

                        self.item_id = format!("msg_{}", Uuid::new_v4());
                        let item = ResponsesOutputItem::Message {
                            id: self.item_id.clone(),
                            role: Some("assistant".to_string()),
                            content: vec![],
                            status: Some("in_progress".to_string()),
                        };
                        events.push(ResponsesStreamEvent::OutputItemAdded {
                            output_index: *index,
                            item,
                            sequence_number: self.next_seq(),
                        });

                        let part = ResponsesContentPart::OutputText {
                            text: String::new(),
                        };
                        events.push(ResponsesStreamEvent::ContentPartAdded {
                            output_index: *index,
                            content_index: 0,
                            part,
                            sequence_number: self.next_seq(),
                        });
                    }
                    crate::types::anthropic::AnthropicContentBlock::ToolUse {
                        id,
                        name,
                        input: _,
                    } => {
                        self.current_block_type = Some("tool_use".to_string());
                        self.current_tool_id = id.clone();
                        self.current_tool_name = name.clone();
                        self.current_tool_arguments = String::new();
                        self.has_tool_call = true;
                        self.item_id = id.clone();

                        let item = ResponsesOutputItem::FunctionCall {
                            id: format!("fc_{}", Uuid::new_v4()),
                            call_id: id.clone(),
                            name: name.clone(),
                            arguments: String::new(),
                            status: Some("in_progress".to_string()),
                        };
                        events.push(ResponsesStreamEvent::OutputItemAdded {
                            output_index: *index,
                            item,
                            sequence_number: self.next_seq(),
                        });
                    }
                    crate::types::anthropic::AnthropicContentBlock::Thinking { .. } => {
                        self.current_block_type = Some("thinking".to_string());
                        self.output_index = *index;
                        self.content_index = 0;

                        self.item_id = format!("rs_{}", Uuid::new_v4());
                        let item = ResponsesOutputItem::Reasoning {
                            id: self.item_id.clone(),
                            content: Some(vec![]),
                            summary: None,
                            status: Some("in_progress".to_string()),
                        };
                        events.push(ResponsesStreamEvent::OutputItemAdded {
                            output_index: *index,
                            item,
                            sequence_number: self.next_seq(),
                        });
                    }
                    _ => {}
                }
            }

            AnthropicStreamEvent::ContentBlockDelta { index, delta } => match delta {
                crate::types::anthropic::AnthropicContentBlockDelta::TextDelta { text } => {
                    self.current_text_content.push_str(text);
                    events.push(ResponsesStreamEvent::OutputTextDelta {
                        output_index: *index,
                        content_index: 0,
                        delta: text.clone(),
                        item_id: Some(self.item_id.clone()),
                        sequence_number: self.next_seq(),
                    });
                }
                crate::types::anthropic::AnthropicContentBlockDelta::InputJsonDelta {
                    partial_json,
                } => {
                    self.current_tool_arguments.push_str(partial_json);
                    events.push(ResponsesStreamEvent::FunctionCallArgsDelta {
                        output_index: *index,
                        delta: partial_json.clone(),
                        item_id: Some(self.item_id.clone()),
                        sequence_number: self.next_seq(),
                    });
                }
                crate::types::anthropic::AnthropicContentBlockDelta::ThinkingDelta { thinking } => {
                    self.current_reasoning.push_str(thinking);
                    self.reasoning_content.push_str(thinking);
                    events.push(ResponsesStreamEvent::ReasoningTextDelta {
                        output_index: *index,
                        content_index: 0,
                        delta: thinking.clone(),
                        sequence_number: self.next_seq(),
                    });
                }
                _ => {}
            },

            AnthropicStreamEvent::ContentBlockStop { index } => {
                match self.current_block_type.as_deref() {
                    Some("text") => {
                        self.flush_text_if_needed(&mut events);

                        let item = ResponsesOutputItem::Message {
                            id: self.item_id.clone(),
                            role: Some("assistant".to_string()),
                            content: vec![ResponsesContentPart::OutputText {
                                text: std::mem::take(&mut self.current_text_content),
                            }],
                            status: Some("completed".to_string()),
                        };

                        // Content part done
                        events.push(ResponsesStreamEvent::ContentPartDone {
                            output_index: *index,
                            content_index: 0,
                            part: ResponsesContentPart::OutputText {
                                text: String::new(),
                            },
                            sequence_number: self.next_seq(),
                        });

                        // Output text done
                        events.push(ResponsesStreamEvent::OutputTextDone {
                            output_index: *index,
                            content_index: 0,
                            text: String::new(),
                            sequence_number: self.next_seq(),
                        });

                        // Output item done
                        events.push(ResponsesStreamEvent::OutputItemDone {
                            output_index: *index,
                            item,
                            sequence_number: self.next_seq(),
                        });

                        self.current_block_type = None;
                    }
                    Some("tool_use") => {
                        let tool_name = std::mem::take(&mut self.current_tool_name);
                        let tool_args = std::mem::take(&mut self.current_tool_arguments);
                        let tool_id = std::mem::take(&mut self.current_tool_id);
                        let item_id = self.item_id.clone();

                        // Function call arguments done
                        events.push(ResponsesStreamEvent::FunctionCallArgsDone {
                            output_index: *index,
                            arguments: tool_args.clone(),
                            name: tool_name.clone(),
                            item_id: Some(item_id.clone()),
                            sequence_number: self.next_seq(),
                        });

                        // Output item done
                        let id = format!("fc_{}", Uuid::new_v4());
                        let item = ResponsesOutputItem::FunctionCall {
                            id: id.clone(),
                            call_id: tool_id,
                            name: tool_name,
                            arguments: tool_args,
                            status: Some("completed".to_string()),
                        };
                        events.push(ResponsesStreamEvent::OutputItemDone {
                            output_index: *index,
                            item,
                            sequence_number: self.next_seq(),
                        });

                        self.current_block_type = None;
                    }
                    Some("thinking") => {
                        let item = ResponsesOutputItem::Reasoning {
                            id: self.item_id.clone(),
                            content: Some(vec![ResponsesContentPart::OutputText {
                                text: std::mem::take(&mut self.current_reasoning),
                            }]),
                            summary: None,
                            status: Some("completed".to_string()),
                        };
                        events.push(ResponsesStreamEvent::OutputItemDone {
                            output_index: *index,
                            item,
                            sequence_number: self.next_seq(),
                        });
                        self.current_block_type = None;
                    }
                    _ => {}
                }
            }

            AnthropicStreamEvent::MessageDelta { delta: _, usage } => {
                self.output_tokens = usage.output_tokens;
            }

            AnthropicStreamEvent::MessageStop => {
                // Flush any remaining text
                self.flush_text_if_needed(&mut events);

                self.event_completed = true;
                let resp = self.make_response_with_status("completed");
                events.push(ResponsesStreamEvent::ResponseCompleted {
                    response: resp,
                    sequence_number: self.next_seq(),
                });
            }

            AnthropicStreamEvent::Error { error } => {
                events.push(ResponsesStreamEvent::Error {
                    code: Some(error.error_type.clone()),
                    message: error.message.clone(),
                    sequence_number: self.next_seq(),
                });
            }

            _ => {}
        }

        events
    }

    fn flush_text_if_needed(&mut self, _events: &mut Vec<ResponsesStreamEvent>) {
        // Text flushing is handled at ContentBlockStop
    }

    fn make_response(&self) -> ResponsesResponse {
        self.make_response_with_status("in_progress")
    }

    fn make_response_with_status(&self, status: &str) -> ResponsesResponse {
        let cached_tokens = self
            .cache_read_tokens
            .map(|c| c + self.cache_creation_tokens.unwrap_or(0));

        ResponsesResponse {
            id: self.response_id.clone(),
            object: "response".to_string(),
            output: vec![],
            status: Some(status.to_string()),
            usage: Some(ResponsesUsage {
                input_tokens: self.input_tokens,
                output_tokens: self.output_tokens,
                total_tokens: self.input_tokens + self.output_tokens,
                input_tokens_details: Some(crate::types::responses::InputTokensDetails {
                    cached_tokens: Some(cached_tokens.unwrap_or(0)),
                }),
                output_tokens_details: None,
            }),
            model: Some(self.model.clone()),
            incomplete_details: None,
            error: None,
            created_at: Some(self.created),
            completed_at: Some(Self::now_unix()),
        }
    }

    pub fn make_completed_response(&self) -> ResponsesResponse {
        self.make_response_with_status("completed")
    }
}

// ============================================================
// Chat SSE -> Responses SSE converter
// ============================================================

pub struct ChatStreamToResponsesTranslator {
    pub response_id: String,
    model: String,
    output_index: u32,
    started: bool,
    finished: bool,
    current_text: String,
    current_refusal: String,
    tool_calls: HashMap<u32, PendingToolCall>,
    item_id: String,
    msg_item_added: bool,
    content_part_added: bool,
    finish_reason: Option<String>,
    usage: Option<ResponsesUsage>,
    seq: u32,
    created: i64,
    pub reasoning_content: String,
    pub strip_xiaomimimo_markers: bool,
}

#[derive(Debug, Clone)]
struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

impl ChatStreamToResponsesTranslator {
    pub fn new(model: &str) -> Self {
        Self {
            response_id: format!("resp_{}", Uuid::new_v4()),
            model: model.to_string(),
            output_index: 0,
            started: false,
            finished: false,
            current_text: String::new(),
            current_refusal: String::new(),
            tool_calls: HashMap::new(),
            item_id: format!("msg_{}", Uuid::new_v4()),
            msg_item_added: false,
            content_part_added: false,
            finish_reason: None,
            usage: None,
            seq: 0,
            created: Self::now_unix(),
            reasoning_content: String::new(),
            strip_xiaomimimo_markers: false,
        }
    }

    fn next_seq(&mut self) -> u32 {
        let s = self.seq;
        self.seq += 1;
        s
    }

    fn now_unix() -> i64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    }

    pub fn process_chunk(&mut self, chunk: &serde_json::Value) -> Vec<ResponsesStreamEvent> {
        let mut events = Vec::new();

        let choices = chunk.get("choices").and_then(|c| c.as_array());
        let chunk_usage = chunk.get("usage");

        if !self.started {
            self.started = true;
            let resp = self.make_response("in_progress");
            events.push(ResponsesStreamEvent::ResponseCreated {
                response: resp.clone(),
                sequence_number: self.next_seq(),
            });
            events.push(ResponsesStreamEvent::ResponseInProgress {
                response: resp,
                sequence_number: self.next_seq(),
            });
        }

        // Track usage from any chunk
        if let Some(usage) = chunk_usage {
            let input_tokens = usage
                .get("prompt_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let output_tokens = usage
                .get("completion_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let total = usage
                .get("total_tokens")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u32;
            let cached = usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);

            let reasoning = usage
                .get("completion_tokens_details")
                .and_then(|d| d.get("reasoning_tokens"))
                .and_then(|v| v.as_u64())
                .map(|v| v as u32);

            self.usage = Some(ResponsesUsage {
                input_tokens,
                output_tokens,
                total_tokens: total,
                input_tokens_details: Some(crate::types::responses::InputTokensDetails {
                    cached_tokens: Some(cached.unwrap_or(0)),
                }),
                output_tokens_details: Some(crate::types::responses::OutputTokensDetails {
                    reasoning_tokens: Some(reasoning.unwrap_or(0)),
                }),
            });
        }

        if let Some(choices) = choices {
            for choice in choices {
                let finish_reason = choice
                    .get("finish_reason")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string());

                if let Some(ref fr) = finish_reason {
                    self.finish_reason = Some(fr.clone());
                }

                let delta = choice.get("delta");

                if !self.msg_item_added {
                    self.msg_item_added = true;
                    let item = ResponsesOutputItem::Message {
                        id: self.item_id.clone(),
                        role: Some("assistant".to_string()),
                        content: vec![],
                        status: Some("in_progress".to_string()),
                    };
                    events.push(ResponsesStreamEvent::OutputItemAdded {
                        output_index: self.output_index,
                        item,
                        sequence_number: self.next_seq(),
                    });
                }

                if !self.content_part_added {
                    self.content_part_added = true;
                    let part = ResponsesContentPart::OutputText {
                        text: String::new(),
                    };
                    events.push(ResponsesStreamEvent::ContentPartAdded {
                        output_index: self.output_index,
                        content_index: 0,
                        part,
                        sequence_number: self.next_seq(),
                    });
                }

                if let Some(delta) = delta {
                    // Text content delta
                    if let Some(content) = delta.get("content") {
                        if let Some(text) = content.as_str() {
                            if !text.is_empty() {
                                let clean = if self.strip_xiaomimimo_markers {
                                    text.replace("[[REASONING_SUMMARY]]", "")
                                        .replace("[[REASONING_DIVIDER]]", "")
                                } else {
                                    text.to_string()
                                };
                                if !clean.is_empty() {
                                    self.current_text.push_str(&clean);
                                    events.push(ResponsesStreamEvent::OutputTextDelta {
                                        output_index: self.output_index,
                                        content_index: 0,
                                        delta: clean,
                                        item_id: Some(self.item_id.clone()),
                                        sequence_number: self.next_seq(),
                                    });
                                }
                            }
                        }
                    }

                    // Refusal delta
                    if let Some(refusal) = delta.get("refusal") {
                        if let Some(text) = refusal.as_str() {
                            if !text.is_empty() {
                                self.current_refusal.push_str(text);
                                events.push(ResponsesStreamEvent::RefusalDelta {
                                    output_index: self.output_index,
                                    content_index: 0,
                                    delta: text.to_string(),
                                    sequence_number: self.next_seq(),
                                });
                            }
                        }
                    }

                    // Reasoning content delta
                    if let Some(reasoning) = delta.get("reasoning_content") {
                        if let Some(text) = reasoning.as_str() {
                            if !text.is_empty() {
                                self.reasoning_content.push_str(text);
                                events.push(ResponsesStreamEvent::ReasoningTextDelta {
                                    output_index: self.output_index,
                                    content_index: 0,
                                    delta: text.to_string(),
                                    sequence_number: self.next_seq(),
                                });
                            }
                        }
                    }

                    // Tool calls delta
                    if let Some(tool_calls) = delta.get("tool_calls").and_then(|tc| tc.as_array()) {
                        for tc in tool_calls {
                            let index =
                                tc.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;

                            let is_new = !self.tool_calls.contains_key(&index);

                            if is_new {
                                let id = tc
                                    .get("id")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                                    .unwrap_or_else(|| format!("call_{}", Uuid::new_v4()));
                                let name = tc
                                    .get("function")
                                    .and_then(|f| f.get("name"))
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string())
                                    .unwrap_or_default();

                                self.tool_calls.insert(
                                    index,
                                    PendingToolCall {
                                        id: id.clone(),
                                        name: name.clone(),
                                        arguments: String::new(),
                                    },
                                );

                                // Emit output_item.added for function call
                                let item = ResponsesOutputItem::FunctionCall {
                                    id: format!("fc_{}", Uuid::new_v4()),
                                    call_id: id.clone(),
                                    name: name.clone(),
                                    arguments: String::new(),
                                    status: Some("in_progress".to_string()),
                                };
                                events.push(ResponsesStreamEvent::OutputItemAdded {
                                    output_index: index + 1, // offset by 1 since message is index 0
                                    item,
                                    sequence_number: self.next_seq(),
                                });
                            }

                            // Arguments delta
                            if let Some(args) = tc
                                .get("function")
                                .and_then(|f| f.get("arguments"))
                                .and_then(|v| v.as_str())
                            {
                                if let Some(pending) = self.tool_calls.get_mut(&index) {
                                    pending.arguments.push_str(args);
                                }
                                events.push(ResponsesStreamEvent::FunctionCallArgsDelta {
                                    output_index: index + 1,
                                    delta: args.to_string(),
                                    item_id: None,
                                    sequence_number: self.next_seq(),
                                });
                            }
                        }
                    }
                }

                // Check for finish
                if let Some(ref fr) = finish_reason {
                    if fr != "stop" || !self.finished {
                        // --- Emit closing events in Go order: text_done → content_part_done → output_item_done ---

                        let full_text = std::mem::take(&mut self.current_text);

                        // output_text.done (with full text)
                        events.push(ResponsesStreamEvent::OutputTextDone {
                            output_index: self.output_index,
                            content_index: 0,
                            text: full_text.clone(),
                            sequence_number: self.next_seq(),
                        });

                        // content_part.done
                        events.push(ResponsesStreamEvent::ContentPartDone {
                            output_index: self.output_index,
                            content_index: 0,
                            part: ResponsesContentPart::OutputText {
                                text: full_text.clone(),
                            },
                            sequence_number: self.next_seq(),
                        });

                        // output_item.done for message
                        let msg_item = ResponsesOutputItem::Message {
                            id: self.item_id.clone(),
                            role: Some("assistant".to_string()),
                            content: vec![ResponsesContentPart::OutputText { text: full_text }],
                            status: Some("completed".to_string()),
                        };
                        events.push(ResponsesStreamEvent::OutputItemDone {
                            output_index: self.output_index,
                            item: msg_item,
                            sequence_number: self.next_seq(),
                        });

                        // Output item done for each tool call
                        let tool_call_count = self.tool_calls.len();
                        let mut seqs: Vec<u32> = Vec::with_capacity(tool_call_count * 2);
                        for _ in 0..tool_call_count {
                            seqs.push(self.next_seq());
                            seqs.push(self.next_seq());
                        }
                        for (i, (index, pending)) in self.tool_calls.iter().enumerate() {
                            let item = ResponsesOutputItem::FunctionCall {
                                id: format!("fc_{}", Uuid::new_v4()),
                                call_id: pending.id.clone(),
                                name: pending.name.clone(),
                                arguments: pending.arguments.clone(),
                                status: Some("completed".to_string()),
                            };

                            events.push(ResponsesStreamEvent::FunctionCallArgsDone {
                                output_index: *index + 1,
                                arguments: pending.arguments.clone(),
                                name: pending.name.clone(),
                                item_id: Some(pending.id.clone()),
                                sequence_number: seqs[i * 2],
                            });

                            events.push(ResponsesStreamEvent::OutputItemDone {
                                output_index: *index + 1,
                                item,
                                sequence_number: seqs[i * 2 + 1],
                            });
                        }

                        // Response completed
                        let mut resp = self.make_response("completed");
                        if let Some(ref u) = self.usage {
                            resp.usage = Some(u.clone());
                        }
                        if fr == "length" {
                            resp.status = Some("incomplete".to_string());
                            resp.incomplete_details =
                                Some(serde_json::json!({"reason": "max_output_tokens"}));
                        }
                        events.push(ResponsesStreamEvent::ResponseCompleted {
                            response: resp,
                            sequence_number: self.next_seq(),
                        });

                        self.finished = true;
                    }
                }
            }
        }

        events
    }

    pub fn is_finished(&self) -> bool {
        self.finished
    }

    pub fn set_finished(&mut self) {
        self.finished = true;
    }

    /// Emit closing events when stream terminates without a proper finish_reason chunk
    pub fn finalize(&mut self) -> Vec<ResponsesStreamEvent> {
        let mut events = Vec::new();

        if self.msg_item_added {
            let full_text = std::mem::take(&mut self.current_text);

            // output_text.done (with full text)
            events.push(ResponsesStreamEvent::OutputTextDone {
                output_index: self.output_index,
                content_index: 0,
                text: full_text.clone(),
                sequence_number: self.next_seq(),
            });

            // content_part.done
            events.push(ResponsesStreamEvent::ContentPartDone {
                output_index: self.output_index,
                content_index: 0,
                part: ResponsesContentPart::OutputText {
                    text: full_text.clone(),
                },
                sequence_number: self.next_seq(),
            });

            // Extract XML tool calls from full_text (Qwen2.5-Coder format)
            let (clean_text, xml_tool_calls) =
                crate::translate::openai::chat::extract_xml_tool_calls(&full_text);

            // output_text.done (with clean text)
            events.push(ResponsesStreamEvent::OutputTextDone {
                output_index: self.output_index,
                content_index: 0,
                text: clean_text.clone(),
                sequence_number: self.next_seq(),
            });

            // content_part.done
            events.push(ResponsesStreamEvent::ContentPartDone {
                output_index: self.output_index,
                content_index: 0,
                part: ResponsesContentPart::OutputText {
                    text: clean_text.clone(),
                },
                sequence_number: self.next_seq(),
            });

            // output_item.done for message
            let msg_item = ResponsesOutputItem::Message {
                id: self.item_id.clone(),
                role: Some("assistant".to_string()),
                content: vec![ResponsesContentPart::OutputText { text: clean_text }],
                status: Some("completed".to_string()),
            };
            events.push(ResponsesStreamEvent::OutputItemDone {
                output_index: self.output_index,
                item: msg_item,
                sequence_number: self.next_seq(),
            });

            // Output item done for each XML tool call (Qwen format)
            for tc in &xml_tool_calls {
                let xml_index = self.tool_calls.len() as u32 + 1;
                events.push(ResponsesStreamEvent::OutputItemAdded {
                    output_index: xml_index,
                    item: ResponsesOutputItem::FunctionCall {
                        id: tc.id.clone(),
                        call_id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                        status: Some("completed".to_string()),
                    },
                    sequence_number: self.next_seq(),
                });
                events.push(ResponsesStreamEvent::OutputItemDone {
                    output_index: xml_index,
                    item: ResponsesOutputItem::FunctionCall {
                        id: tc.id.clone(),
                        call_id: tc.id.clone(),
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                        status: Some("completed".to_string()),
                    },
                    sequence_number: self.next_seq(),
                });
            }

            // Output item done for each tool call (from standard delta)
            let tool_call_count = self.tool_calls.len();

            let mut seqs: Vec<u32> = Vec::with_capacity(tool_call_count * 2);
            for _ in 0..tool_call_count {
                seqs.push(self.next_seq());
                seqs.push(self.next_seq());
            }
            for (i, (index, pending)) in self.tool_calls.iter().enumerate() {
                let item = ResponsesOutputItem::FunctionCall {
                    id: format!("fc_{}", Uuid::new_v4()),
                    call_id: pending.id.clone(),
                    name: pending.name.clone(),
                    arguments: pending.arguments.clone(),
                    status: Some("completed".to_string()),
                };

                events.push(ResponsesStreamEvent::FunctionCallArgsDone {
                    output_index: *index + 1,
                    arguments: pending.arguments.clone(),
                    name: pending.name.clone(),
                    item_id: Some(pending.id.clone()),
                    sequence_number: seqs[i * 2],
                });

                events.push(ResponsesStreamEvent::OutputItemDone {
                    output_index: *index + 1,
                    item,
                    sequence_number: seqs[i * 2 + 1],
                });
            }
        }

        // Response completed
        let mut resp = self.make_response("completed");
        if let Some(ref u) = self.usage {
            resp.usage = Some(u.clone());
        }
        resp.status = Some("completed".to_string());
        events.push(ResponsesStreamEvent::ResponseCompleted {
            response: resp,
            sequence_number: self.next_seq(),
        });

        events
    }

    fn make_response(&self, status: &str) -> ResponsesResponse {
        let completed_at = if status == "completed" {
            Some(Self::now_unix())
        } else {
            None
        };

        // Build output from accumulated state
        let mut output: Vec<ResponsesOutputItem> = Vec::new();

        // Extract XML tool calls from accumulated text (Qwen format)
        let (clean_text, xml_tool_calls) =
            crate::translate::openai::chat::extract_xml_tool_calls(&self.current_text);

        // Message item
        if !clean_text.is_empty() {
            output.push(ResponsesOutputItem::Message {
                id: self.item_id.clone(),
                role: Some("assistant".to_string()),
                content: vec![ResponsesContentPart::OutputText { text: clean_text }],
                status: Some("completed".to_string()),
            });
        }

        // XML tool calls
        for tc in &xml_tool_calls {
            output.push(ResponsesOutputItem::FunctionCall {
                id: tc.id.clone(),
                call_id: tc.id.clone(),
                name: tc.function.name.clone(),
                arguments: tc.function.arguments.clone(),
                status: Some("completed".to_string()),
            });
        }

        // Standard tool calls (from delta)
        for pending in self.tool_calls.values() {
            output.push(ResponsesOutputItem::FunctionCall {
                id: pending.id.clone(),
                call_id: pending.id.clone(),
                name: pending.name.clone(),
                arguments: pending.arguments.clone(),
                status: Some("completed".to_string()),
            });
        }

        ResponsesResponse {
            id: self.response_id.clone(),
            object: "response".to_string(),
            output,
            status: Some(status.to_string()),
            usage: self.usage.clone(),
            model: Some(self.model.clone()),
            incomplete_details: None,
            error: None,
            created_at: Some(self.created),
            completed_at,
        }
    }
}

// ============================================================
// Responses SSE -> Chat SSE converter
// ============================================================

#[allow(dead_code)]
pub struct ResponsesStreamToChatTranslator {
    chat_id: String,
    model: String,
    created: u64,
    finished: bool,
    current_text: String,
    current_tool_calls: Vec<ToolCallDelta>,
    current_refusal: String,
    usage: Option<Box<crate::types::chat::ChatUsage>>,
}

impl ResponsesStreamToChatTranslator {
    pub fn new(model: &str) -> Self {
        Self {
            chat_id: format!("chatcmpl-{}", Uuid::new_v4()),
            model: model.to_string(),
            created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            finished: false,
            current_text: String::new(),
            current_tool_calls: Vec::new(),
            current_refusal: String::new(),
            usage: None,
        }
    }

    pub fn process_event(&mut self, event: &ResponsesStreamEvent) -> Vec<serde_json::Value> {
        let mut chunks = Vec::new();

        match event {
            ResponsesStreamEvent::OutputTextDelta { delta, .. } => {
                let delta_obj = ChatDelta {
                    role: None,
                    content: Some(ChatContent::String(delta.clone())),
                    tool_calls: None,
                    refusal: None,
                    reasoning_content: None,
                };
                chunks.push(self.make_chunk(Some(delta_obj), None));
            }
            ResponsesStreamEvent::RefusalDelta { delta, .. } => {
                let delta_obj = ChatDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                    refusal: Some(delta.clone()),
                    reasoning_content: None,
                };
                chunks.push(self.make_chunk(Some(delta_obj), None));
            }
            ResponsesStreamEvent::FunctionCallArgsDelta {
                output_index,
                delta,
                item_id,
                ..
            } => {
                let index = *output_index;

                // Ensure we have enough entries
                while self.current_tool_calls.len() <= index as usize {
                    self.current_tool_calls.push(ToolCallDelta {
                        index: self.current_tool_calls.len() as u32,
                        id: None,
                        tool_type: None,
                        function: None,
                    });
                }

                let tc = &mut self.current_tool_calls[index as usize];

                // On first delta, include id and name from item_id
                let delta_obj = if tc.id.is_none() {
                    tc.id = item_id.clone();
                    tc.tool_type = Some("function".to_string());

                    ChatDelta {
                        role: None,
                        content: None,
                        tool_calls: Some(vec![ToolCallDelta {
                            index,
                            id: item_id.clone(),
                            tool_type: Some("function".to_string()),
                            function: Some(FunctionCallDelta {
                                name: None,
                                arguments: Some(delta.clone()),
                            }),
                        }]),
                        refusal: None,
                        reasoning_content: None,
                    }
                } else {
                    ChatDelta {
                        role: None,
                        content: None,
                        tool_calls: Some(vec![ToolCallDelta {
                            index,
                            id: None,
                            tool_type: None,
                            function: Some(FunctionCallDelta {
                                name: None,
                                arguments: Some(delta.clone()),
                            }),
                        }]),
                        refusal: None,
                        reasoning_content: None,
                    }
                };

                chunks.push(self.make_chunk(Some(delta_obj), None));
            }
            ResponsesStreamEvent::ReasoningTextDelta { delta, .. } => {
                let delta_obj = ChatDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                    reasoning_content: Some(delta.clone()),
                };
                chunks.push(self.make_chunk(Some(delta_obj), None));
            }
            ResponsesStreamEvent::ResponseCompleted { response, .. } if !self.finished => {
                self.finished = true;

                let finish_reason = match response.status.as_deref() {
                    Some("incomplete") => "length",
                    _ => "stop",
                };

                let usage = response
                    .usage
                    .as_ref()
                    .map(|u| crate::types::chat::ChatUsage {
                        prompt_tokens: u.input_tokens,
                        completion_tokens: u.output_tokens,
                        total_tokens: u.total_tokens,
                        prompt_tokens_details: u.input_tokens_details.as_ref().map(|d| {
                            crate::types::chat::PromptTokensDetails {
                                cached_tokens: d.cached_tokens,
                            }
                        }),
                        completion_tokens_details: u.output_tokens_details.as_ref().map(|d| {
                            crate::types::chat::CompletionTokensDetails {
                                reasoning_tokens: d.reasoning_tokens,
                            }
                        }),
                    });

                let final_chunk = ChatDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                    reasoning_content: None,
                };

                chunks.push(serde_json::json!({
                    "id": self.chat_id,
                    "object": "chat.completion.chunk",
                    "created": self.created,
                    "model": self.model,
                    "choices": [{
                        "index": 0,
                        "delta": final_chunk,
                        "finish_reason": finish_reason,
                    }],
                    "usage": usage,
                }));

                // Send [DONE]
                chunks.push(serde_json::Value::String("[DONE]".to_string()));
            }
            _ => {}
        }

        chunks
    }

    fn make_chunk(
        &self,
        delta: Option<ChatDelta>,
        finish_reason: Option<&str>,
    ) -> serde_json::Value {
        serde_json::json!({
            "id": self.chat_id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [{
                "index": 0,
                "delta": delta.unwrap_or(ChatDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                    refusal: None,
                    reasoning_content: None,
                }),
                "finish_reason": finish_reason,
            }],
        })
    }
}
