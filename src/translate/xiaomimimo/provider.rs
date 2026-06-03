use crate::config::UpstreamVendor;
use crate::translate::compatibility::{ProviderCapabilities, ReasoningSupport};
use crate::translate::provider::Provider;
use crate::translate::xiaomimimo::chat::convert_responses_to_chat as xm_chat;
use crate::translate::xiaomimimo::convert_responses_to_anthropic as xm_anthropic;
use crate::types::anthropic::AnthropicRequest;
use crate::types::chat::ChatCompletionsRequest;
use crate::types::responses::ResponsesRequest;

pub struct XiaomiMimoProvider;

impl Provider for XiaomiMimoProvider {
    fn vendor(&self) -> UpstreamVendor {
        UpstreamVendor::XiaomiMimo
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            reasoning: ReasoningSupport::Boolean,
            ..ProviderCapabilities::default()
        }
    }

    fn responses_to_chat(
        &self,
        req: &ResponsesRequest,
        previous_reasoning: Option<String>,
    ) -> ChatCompletionsRequest {
        xm_chat(req, previous_reasoning)
    }

    fn responses_to_anthropic(&self, req: &ResponsesRequest) -> AnthropicRequest {
        xm_anthropic(req)
    }
}
