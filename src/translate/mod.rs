pub mod anthropic;
pub mod deepseek;
pub mod openai;

use crate::config::{UpstreamFormat, UpstreamVendor};
use crate::types::chat::ChatCompletionsRequest;
use crate::types::responses::ResponsesRequest;

/// Dispatch Responses→Chat conversion to the correct vendor module
#[allow(dead_code)]
pub fn convert_responses_to_chat(
    responses: &ResponsesRequest,
    format: &UpstreamFormat,
    vendor: &UpstreamVendor,
) -> ChatCompletionsRequest {
    match vendor {
        UpstreamVendor::DeepSeek | UpstreamVendor::Auto => {
            match format {
                UpstreamFormat::Anthropic => {
                    // B方案: Responses→Anthropic (uses different path, not chat)
                    // Return via deepseek chat as fallback
                    deepseek::chat::convert_responses_to_chat(responses, None)
                }
                _ => deepseek::chat::convert_responses_to_chat(responses, None),
            }
        }
        _ => openai::chat::convert_responses_to_chat(responses),
    }
}

/// Dispatch for DeepSeek with reasoning cache support
pub use deepseek::chat::convert_responses_to_chat as convert_for_deepseek;

pub use deepseek::anthropic::{convert_anthropic_to_responses, convert_responses_to_anthropic};
/// Re-export common functions for server.rs compatibility
pub use deepseek::chat::{
    convert_chat_to_responses, convert_chat_to_responses_response,
    convert_responses_to_chat_response as chat_resp_to_responses,
};
