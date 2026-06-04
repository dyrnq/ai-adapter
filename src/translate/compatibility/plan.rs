use super::diagnostic::{CompatibilityAction, Diagnostic, DiagnosticSeverity};
use crate::types::responses::ResponsesRequest;
use std::collections::{HashMap, HashSet};

/// Level of reasoning support by the provider.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReasoningSupport {
    None,
    Boolean,
    Native,
}

/// Describes what an upstream provider supports.
#[derive(Debug, Clone)]
pub struct ProviderCapabilities {
    /// Supported top-level parameters.
    pub parameters: HashSet<String>,
    /// Supported tool types.
    pub tools: HashSet<String>,
    /// Tool names that need degradation (original -> degraded).
    pub tool_degraded: HashMap<String, String>,
    /// Max tools count.
    pub max_tools: Option<usize>,
    /// Supported tool_choice values.
    pub tool_choice: HashSet<String>,
    /// Supported response_format types.
    pub response_formats: HashSet<String>,
    /// Reasoning support.
    pub reasoning: ReasoningSupport,
    /// Supported metadata fields.
    pub metadata_fields: HashSet<String>,
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self {
            parameters: HashSet::from([
                "stream".into(),
                "temperature".into(),
                "top_p".into(),
                "max_output_tokens".into(),
                "frequency_penalty".into(),
                "presence_penalty".into(),
                "stop".into(),
                "user".into(),
            ]),
            tools: HashSet::from(["function".into()]),
            tool_degraded: HashMap::new(),
            max_tools: Some(128),
            tool_choice: HashSet::from(["auto".into(), "none".into(), "required".into()]),
            response_formats: HashSet::from(["text".into(), "json_object".into()]),
            reasoning: ReasoningSupport::None,
            metadata_fields: HashSet::new(),
        }
    }
}

/// Result of planning compatibility for a request.
#[derive(Debug, Clone, Default)]
pub struct CompatibilityPlan {
    /// Collected diagnostics.
    pub diagnostics: Vec<Diagnostic>,
}

impl CompatibilityPlan {
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    /// Run the full compatibility plan against a request.
    pub fn plan(
        &mut self,
        request: &ResponsesRequest,
        capabilities: &ProviderCapabilities,
        provider: &str,
        model: &str,
    ) {
        self.plan_response_format(request, capabilities, provider, model);
        self.plan_tools(request, capabilities, provider, model);
        self.plan_streaming(request, capabilities, provider, model);
        self.plan_reasoning(request, capabilities, provider, model);
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        code: &str,
        severity: DiagnosticSeverity,
        action: CompatibilityAction,
        message: String,
        path: Option<&str>,
        provider: &str,
        model: &str,
        metadata: Option<serde_json::Value>,
    ) {
        let mut d = Diagnostic::new(code, severity, action, message, provider, model);
        if let Some(p) = path {
            d = d.with_path(p);
        }
        if let Some(m) = metadata {
            d = d.with_metadata(m);
        }
        self.diagnostics.push(d);
    }

    fn plan_response_format(
        &mut self,
        request: &ResponsesRequest,
        caps: &ProviderCapabilities,
        provider: &str,
        model: &str,
    ) {
        let fmt_type = request.text.as_ref().and_then(|t| match &t.format {
            Some(crate::types::responses::TextFormat::Text) => Some("text"),
            Some(crate::types::responses::TextFormat::JsonObject) => Some("json_object"),
            Some(crate::types::responses::TextFormat::JsonSchema { .. }) => Some("json_schema"),
            None => None,
        });
        let Some(fmt_type) = fmt_type else { return };

        if caps.response_formats.contains(fmt_type) {
            return;
        }

        if fmt_type == "json_schema" && caps.response_formats.contains("json_object") {
            self.add(
                "bridge.param.degraded",
                DiagnosticSeverity::Warn,
                CompatibilityAction::Degraded,
                format!("json_schema degraded to json_object for {provider}"),
                Some("text.format"),
                provider,
                model,
                Some(serde_json::json!({"requested": "json_schema", "effective": "json_object"})),
            );
            return;
        }

        self.add(
            "bridge.param.unsupported",
            DiagnosticSeverity::Error,
            CompatibilityAction::Rejected,
            format!("response_format '{fmt_type}' not supported by {provider}"),
            Some("text.format"),
            provider,
            model,
            None,
        );
    }

    fn plan_tools(
        &mut self,
        request: &ResponsesRequest,
        caps: &ProviderCapabilities,
        provider: &str,
        model: &str,
    ) {
        let Some(tools) = &request.tools else { return };
        if tools.is_empty() {
            return;
        }

        if caps.max_tools == Some(0) || caps.max_tools.is_none() && !tools.is_empty() {
            if caps.tools.is_empty() {
                self.add(
                    "bridge.param.unsupported",
                    DiagnosticSeverity::Error,
                    CompatibilityAction::Rejected,
                    format!("{provider} does not support tool calling"),
                    Some("tools"),
                    provider,
                    model,
                    None,
                );
            }
            return;
        }

        if let Some(max) = caps.max_tools {
            if tools.len() > max {
                self.add(
                    "bridge.param.degraded",
                    DiagnosticSeverity::Warn,
                    CompatibilityAction::Degraded,
                    format!("{} tools exceeds {} max of {}", tools.len(), provider, max),
                    Some("tools"),
                    provider,
                    model,
                    Some(serde_json::json!({"count": tools.len(), "max": max})),
                );
            }
        }
    }

    fn plan_streaming(
        &mut self,
        request: &ResponsesRequest,
        caps: &ProviderCapabilities,
        provider: &str,
        model: &str,
    ) {
        if request.stream.unwrap_or(false) && !caps.parameters.contains("stream") {
            self.add(
                "bridge.param.ignored",
                DiagnosticSeverity::Info,
                CompatibilityAction::Ignored,
                format!("streaming not supported by {provider}, fallback to non-stream"),
                Some("stream"),
                provider,
                model,
                None,
            );
        }
    }

    fn plan_reasoning(
        &mut self,
        request: &ResponsesRequest,
        caps: &ProviderCapabilities,
        provider: &str,
        model: &str,
    ) {
        if request.reasoning.is_some() && caps.reasoning == ReasoningSupport::None {
            self.add(
                "bridge.param.ignored",
                DiagnosticSeverity::Info,
                CompatibilityAction::Ignored,
                format!("reasoning not supported by {provider}, ignored"),
                Some("reasoning"),
                provider,
                model,
                None,
            );
        }
    }
}
