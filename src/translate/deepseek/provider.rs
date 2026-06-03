use crate::config::UpstreamVendor;
use crate::translate::compatibility::{ProviderCapabilities, ReasoningSupport};
use crate::translate::deepseek::anthropic::convert_responses_to_anthropic as ds_responses_to_anthropic;
use crate::translate::deepseek::chat::convert_responses_to_chat as ds_responses_to_chat;
use crate::translate::provider::Provider;
use crate::types::anthropic::AnthropicRequest;
use crate::types::chat::ChatCompletionsRequest;
use crate::types::responses::ResponsesRequest;

pub struct DeepSeekProvider;

impl Provider for DeepSeekProvider {
    fn vendor(&self) -> UpstreamVendor {
        UpstreamVendor::DeepSeek
    }

    fn capabilities(&self) -> ProviderCapabilities {
        let mut caps = ProviderCapabilities::default();
        caps.parameters.insert("reasoning".into());
        caps.reasoning = ReasoningSupport::Native;
        #[allow(clippy::field_reassign_with_default)]
        caps
    }

    fn responses_to_chat(
        &self,
        req: &ResponsesRequest,
        previous_reasoning: Option<String>,
    ) -> ChatCompletionsRequest {
        ds_responses_to_chat(req, previous_reasoning)
    }

    fn responses_to_anthropic(&self, req: &ResponsesRequest) -> AnthropicRequest {
        ds_responses_to_anthropic(req)
    }
}
