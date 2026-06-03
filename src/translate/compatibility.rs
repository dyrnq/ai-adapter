use std::collections::{HashMap, HashSet};

/// Compatibility diagnostic severity
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum DiagnosticSeverity {
    Info,
    Warn,
    Error,
}

/// A single compatibility diagnostic message
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct CompatibilityDiagnostic {
    pub code: String,
    pub severity: DiagnosticSeverity,
    pub path: Option<String>,
    pub action: String,
    pub message: String,
}

/// Provider capabilities declaration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ProviderCapabilities {
    /// Supported top-level parameters
    pub parameters: HashSet<String>,
    /// Supported tool types
    pub tools: HashSet<String>,
    /// Tool names that need degradation (original → degraded)
    pub tool_degraded: HashMap<String, String>,
    /// Max tools count
    pub max_tools: Option<usize>,
    /// Supported tool_choice values
    pub tool_choice: HashSet<String>,
    /// Supported response_format types
    pub response_formats: HashSet<String>,
    /// Reasoning support
    pub reasoning: ReasoningSupport,
    /// Supported metadata fields
    pub metadata_fields: HashSet<String>,
}

#[derive(Debug, Clone)]
pub enum ReasoningSupport {
    None,
    Boolean,
    Native,
}

/// Plan for adapting a request to provider capabilities
#[derive(Debug, Default)]
#[allow(dead_code)]
pub struct CompatibilityPlan {
    pub dropped_parameters: Vec<String>,
    pub degraded_tools: Vec<(String, String)>,
    pub effective_tool_choice: Option<String>,
    pub diagnostics: Vec<CompatibilityDiagnostic>,
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

impl ProviderCapabilities {
    /// Plan request compatibility: detect unsupported features and plan degradation.
    #[allow(dead_code)]
    pub fn plan_for_request(
        &self,
        has_tools: bool,
        user_tool_choice: Option<&str>,
        user_reasoning: bool,
        response_format: Option<&str>,
    ) -> CompatibilityPlan {
        let mut plan = CompatibilityPlan::default();
        let provider_name = "unknown"; // caller should set this

        // --- tool_choice ---
        let mut effective_tc = user_tool_choice.map(|s| s.to_string());
        if let Some(ref tc) = effective_tc {
            if !self.tool_choice.contains(tc.as_str()) {
                plan.diagnostics.push(CompatibilityDiagnostic {
                    code: "compatibility.unsupported_tool_choice".into(),
                    severity: DiagnosticSeverity::Warn,
                    path: Some("tool_choice".into()),
                    action: "degraded".into(),
                    message: format!(
                        "Provider {} does not support tool_choice '{}', defaulting to 'auto'",
                        provider_name, tc
                    ),
                });
                if has_tools {
                    effective_tc = Some("auto".into());
                } else {
                    effective_tc = None;
                }
            }
        } else if has_tools && !self.tool_choice.contains("auto") {
            effective_tc = Some("required".into());
        }
        plan.effective_tool_choice = effective_tc;

        // --- reasoning ---
        if user_reasoning && matches!(self.reasoning, ReasoningSupport::None) {
            plan.dropped_parameters.push("reasoning".into());
            plan.diagnostics.push(CompatibilityDiagnostic {
                code: "compatibility.unsupported_reasoning".into(),
                severity: DiagnosticSeverity::Info,
                path: Some("reasoning".into()),
                action: "ignored".into(),
                message: format!(
                    "Provider {} does not support reasoning, dropped from request",
                    provider_name
                ),
            });
        }

        // --- response_format ---
        if let Some(rf) = response_format {
            if !self.response_formats.contains(rf) {
                plan.diagnostics.push(CompatibilityDiagnostic {
                    code: "compatibility.unsupported_response_format".into(),
                    severity: DiagnosticSeverity::Warn,
                    path: Some("response_format".into()),
                    action: "degraded".into(),
                    message: format!(
                        "Provider {} does not support response_format '{}', may degrade to text",
                        provider_name, rf
                    ),
                });
            }
        }

        plan
    }
}

/// Log diagnostics at appropriate level
#[allow(dead_code)]
pub fn log_diagnostics(diagnostics: &[CompatibilityDiagnostic]) {
    if diagnostics.is_empty() {
        return;
    }
    for d in diagnostics {
        let msg = format!(
            "[{}] {}: {} (action: {})",
            d.code,
            d.path.as_deref().unwrap_or("?"),
            d.message,
            d.action
        );
        match d.severity {
            DiagnosticSeverity::Error => tracing::error!("{}", msg),
            DiagnosticSeverity::Warn => tracing::warn!("{}", msg),
            DiagnosticSeverity::Info => tracing::info!("{}", msg),
        }
    }
}
