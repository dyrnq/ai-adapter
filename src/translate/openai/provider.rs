use crate::config::UpstreamVendor;
use crate::translate::compatibility::ProviderCapabilities;
use crate::translate::openai::chat::convert_responses_to_chat;
use crate::translate::provider::Provider;
use crate::types::anthropic::AnthropicRequest;
use crate::types::chat::ChatCompletionsRequest;
use crate::types::responses::ResponsesRequest;

pub struct OpenAiProvider;

impl Provider for OpenAiProvider {
    fn vendor(&self) -> UpstreamVendor {
        UpstreamVendor::OpenAI
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    fn responses_to_chat(
        &self,
        req: &ResponsesRequest,
        _previous_reasoning: Option<String>,
    ) -> ChatCompletionsRequest {
        convert_responses_to_chat(req)
    }

    fn responses_to_anthropic(&self, _req: &ResponsesRequest) -> AnthropicRequest {
        unimplemented!("OpenAI provider does not support Anthropic protocol")
    }
}
