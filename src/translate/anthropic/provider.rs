use crate::config::UpstreamVendor;
use crate::translate::compatibility::ProviderCapabilities;
use crate::translate::deepseek::anthropic::convert_responses_to_anthropic;
use crate::translate::provider::Provider;
use crate::types::anthropic::AnthropicRequest;
use crate::types::chat::ChatCompletionsRequest;
use crate::types::responses::ResponsesRequest;

pub struct AnthropicProvider;

impl Provider for AnthropicProvider {
    fn vendor(&self) -> UpstreamVendor {
        UpstreamVendor::Anthropic
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    fn responses_to_chat(
        &self,
        _req: &ResponsesRequest,
        _previous_reasoning: Option<String>,
    ) -> ChatCompletionsRequest {
        unimplemented!("Anthropic provider uses native Anthropic protocol, not Chat")
    }

    fn responses_to_anthropic(&self, req: &ResponsesRequest) -> AnthropicRequest {
        convert_responses_to_anthropic(req, None)
    }
}
