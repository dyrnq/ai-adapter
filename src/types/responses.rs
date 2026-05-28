use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;

/// Instructions may be a string or an array of strings (Codex CLI sends both forms).
/// When serializing we always produce a string (DeepSeek requirement).
#[derive(Debug, Clone, Serialize)]
pub struct Instructions(pub String);

impl<'de> Deserialize<'de> for Instructions {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let val = Value::deserialize(deserializer)?;
        match val {
            Value::String(s) => Ok(Instructions(s)),
            Value::Array(arr) => {
                let text: String = arr
                    .iter()
                    .filter_map(|v| {
                        v.as_str().map(|s| s.to_string()).or_else(|| {
                            v.get("text")
                                .and_then(|t| t.as_str())
                                .map(|s| s.to_string())
                        })
                    })
                    .collect::<Vec<_>>()
                    .join("\n\n");
                Ok(Instructions(text))
            }
            other => Err(serde::de::Error::custom(format!(
                "instructions: expected string or array of strings, got: {}",
                other
            ))),
        }
    }
}

/// Deserialize FunctionCallOutput::output which may be:
/// - String (normal)
/// - Array of strings or {text: "..."} objects (Codex long-session bug)
pub fn deser_output_or_content<'de, D>(d: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let val = Value::deserialize(d)?;
    match val {
        Value::String(s) => Ok(s),
        Value::Array(arr) => {
            let text: String = arr
                .iter()
                .filter_map(|v| {
                    v.as_str().map(|s| s.to_string()).or_else(|| {
                        v.get("text")
                            .and_then(|t| t.as_str())
                            .map(|s| s.to_string())
                    })
                })
                .collect::<Vec<_>>()
                .join("\n");
            Ok(text)
        }
        other => Ok(serde_json::to_string(&other).unwrap_or_default()),
    }
}

/// OpenAI Responses API request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesRequest {
    pub model: String,
    pub input: Option<Vec<ResponsesInputItem>>,
    pub instructions: Option<Instructions>,
    pub stream: Option<bool>,
    pub max_output_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub tools: Option<Vec<ResponsesTool>>,
    pub tool_choice: Option<Value>,
    pub reasoning: Option<ReasoningConfig>,
    pub text: Option<TextConfig>,
    pub truncation: Option<String>,
    pub store: Option<bool>,
    pub metadata: Option<Value>,
    pub previous_response_id: Option<String>,
    #[serde(skip_deserializing, default)]
    pub session_id: Option<String>,
    pub user: Option<String>,
    pub service_tier: Option<String>,
    #[serde(rename = "parallel_tool_calls")]
    pub parallel_tool_calls: Option<bool>,
    pub top_logprobs: Option<u32>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningConfig {
    pub effort: Option<String>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    pub format: Option<TextFormat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum TextFormat {
    Text,
    JsonObject,
    JsonSchema {
        name: String,
        schema: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        strict: Option<bool>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    // Nested format: {"type": "function", "function": {"name": "...", ...}}
    #[serde(default)]
    pub function: Option<FunctionDefinition>,
    // Flat format: {"type": "function", "name": "...", "description": "...", "parameters": {...}}
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Option<Value>,
    #[serde(default)]
    pub strict: Option<bool>,
}

impl ResponsesTool {
    /// Get FunctionDefinition regardless of whether the tool uses nested or flat format
    pub fn get_function(&self) -> Option<FunctionDefinition> {
        if let Some(ref f) = self.function {
            return Some(f.clone());
        }
        // Flat format: name/description/parameters are at top level
        self.name.as_ref().map(|n| FunctionDefinition {
            name: n.clone(),
            description: self.description.clone(),
            parameters: self.parameters.clone(),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<Value>,
}

/// Input items in a Responses request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponsesInputItem {
    Message {
        role: String,
        content: Option<Vec<ResponsesContentPart>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    #[serde(rename = "function_call_output")]
    FunctionCallOutput {
        call_id: String,
        #[serde(deserialize_with = "deser_output_or_content")]
        output: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponsesContentPart {
    #[serde(rename = "input_text")]
    InputText {
        text: String,
    },
    #[serde(rename = "input_image")]
    InputImage {
        #[serde(skip_serializing_if = "Option::is_none")]
        image_url: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    #[serde(rename = "input_file")]
    InputFile {
        #[serde(skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        filename: Option<String>,
    },
    #[serde(rename = "output_text")]
    OutputText {
        text: String,
    },
    Text {
        text: String,
    },
    #[serde(rename = "image_url")]
    ImageUrl {
        url: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        detail: Option<String>,
    },
    ToolResult {
        call_id: String,
        output: String,
    },
}

/// Responses API response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: String,
    pub output: Vec<ResponsesOutputItem>,
    pub status: Option<String>,
    pub usage: Option<ResponsesUsage>,
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub incomplete_details: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponsesOutputItem {
    Message {
        id: String,
        role: Option<String>,
        content: Vec<ResponsesContentPart>,
        status: Option<String>,
    },
    #[serde(rename = "function_call")]
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: Option<String>,
    },
    Reasoning {
        id: String,
        content: Option<Vec<ResponsesContentPart>>,
        summary: Option<Vec<ResponsesContentPart>>,
        status: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponsesUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<InputTokensDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<OutputTokensDetails>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputTokensDetails {
    pub cached_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: Option<u32>,
}

/// SSE streaming events for Responses API
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponsesStreamEvent {
    #[serde(rename = "response.created")]
    ResponseCreated {
        response: ResponsesResponse,
        sequence_number: u32,
    },
    #[serde(rename = "response.in_progress")]
    ResponseInProgress {
        response: ResponsesResponse,
        sequence_number: u32,
    },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        output_index: u32,
        item: ResponsesOutputItem,
        sequence_number: u32,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        output_index: u32,
        content_index: u32,
        part: ResponsesContentPart,
        sequence_number: u32,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        output_index: u32,
        content_index: u32,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        item_id: Option<String>,
        sequence_number: u32,
    },
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        output_index: u32,
        content_index: u32,
        delta: String,
        sequence_number: u32,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgsDelta {
        output_index: u32,
        delta: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        item_id: Option<String>,
        sequence_number: u32,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgsDone {
        output_index: u32,
        arguments: String,
        name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        item_id: Option<String>,
        sequence_number: u32,
    },
    #[serde(rename = "response.reasoning_text.delta")]
    ReasoningTextDelta {
        output_index: u32,
        content_index: u32,
        delta: String,
        sequence_number: u32,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        output_index: u32,
        item: ResponsesOutputItem,
        sequence_number: u32,
    },
    #[serde(rename = "response.content_part.done")]
    ContentPartDone {
        output_index: u32,
        content_index: u32,
        part: ResponsesContentPart,
        sequence_number: u32,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        output_index: u32,
        content_index: u32,
        text: String,
        sequence_number: u32,
    },
    #[serde(rename = "response.completed")]
    ResponseCompleted {
        response: ResponsesResponse,
        sequence_number: u32,
    },
    #[serde(rename = "error")]
    Error {
        code: Option<String>,
        message: String,
        sequence_number: u32,
    },
}

/// Helper: extract text content from content parts
pub fn content_parts_to_text(parts: &[ResponsesContentPart]) -> String {
    parts
        .iter()
        .filter_map(|p| match p {
            ResponsesContentPart::InputText { text }
            | ResponsesContentPart::OutputText { text }
            | ResponsesContentPart::Text { text } => Some(text.as_str()),
            ResponsesContentPart::ToolResult { ref output, .. } => Some(output.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// POST /v1/responses/compact request — sent by Codex CLI to compress conversation history
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactRequest {
    pub input: Vec<ResponsesInputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<Instructions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// POST /v1/responses/compact response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactResponse {
    pub id: String,
    pub object: String,
    pub output: Vec<ResponsesOutputCompactItem>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ResponsesOutputCompactItem {
    Message {
        id: String,
        role: String,
        content: Vec<ResponsesContentPart>,
        status: String,
    },
}
