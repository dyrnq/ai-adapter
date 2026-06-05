use crate::config::UpstreamVendor;
use crate::translate::compatibility::ProviderCapabilities;
use crate::translate::deepseek::provider::DeepSeekProvider;
use crate::translate::openai::provider::OpenAiProvider;
use crate::translate::xiaomimimo::provider::XiaomiMimoProvider;
use crate::types::anthropic::AnthropicRequest;
use crate::types::chat::ChatCompletionsRequest;
use crate::types::responses::ResponsesRequest;

/// A provider knows how to convert Responses API requests into
/// upstream-native requests, what capabilities it has, and how to
/// create streaming translators.
#[allow(dead_code)]
pub trait Provider: Send + Sync {
    fn vendor(&self) -> UpstreamVendor;

    /// Declare this provider's capabilities (tool support, parameters, reasoning, etc).
    fn capabilities(&self) -> ProviderCapabilities;

    fn responses_to_chat(
        &self,
        req: &ResponsesRequest,
        previous_reasoning: Option<String>,
    ) -> ChatCompletionsRequest;

    fn responses_to_anthropic(&self, req: &ResponsesRequest) -> AnthropicRequest;
}

/// Return the chat provider for a given vendor.
#[allow(dead_code)]
pub fn chat_provider_for(vendor: &UpstreamVendor) -> Box<dyn Provider> {
    match vendor {
        UpstreamVendor::DeepSeek | UpstreamVendor::Auto => Box::new(DeepSeekProvider),
        UpstreamVendor::XiaomiMimo => Box::new(XiaomiMimoProvider),
        UpstreamVendor::MiniMax => Box::new(DeepSeekProvider),
        UpstreamVendor::OpenAI | UpstreamVendor::Anthropic => Box::new(OpenAiProvider),
    }
}

/// Return the Anthropic-native provider for a given vendor.
#[allow(dead_code)]
pub fn anthropic_provider_for(vendor: &UpstreamVendor) -> Box<dyn Provider> {
    match vendor {
        UpstreamVendor::XiaomiMimo => Box::new(XiaomiMimoProvider),
        UpstreamVendor::MiniMax => Box::new(DeepSeekProvider),
        _ => Box::new(DeepSeekProvider),
    }
}
