/// Severity level for compatibility diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagnosticSeverity {
    Info,
    Warn,
    Error,
}

impl std::fmt::Display for DiagnosticSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosticSeverity::Info => write!(f, "info"),
            DiagnosticSeverity::Warn => write!(f, "warn"),
            DiagnosticSeverity::Error => write!(f, "error"),
        }
    }
}

/// The action taken for a parameter or feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompatibilityAction {
    /// Fully supported, no change.
    Supported,
    /// Supported but with degraded quality (e.g. json_schema → json_object).
    Degraded,
    /// Ignored (parameter not forwarded upstream).
    Ignored,
    /// Rejected (request would produce wrong results without this).
    Rejected,
}

impl std::fmt::Display for CompatibilityAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompatibilityAction::Supported => write!(f, "supported"),
            CompatibilityAction::Degraded => write!(f, "degraded"),
            CompatibilityAction::Ignored => write!(f, "ignored"),
            CompatibilityAction::Rejected => write!(f, "rejected"),
        }
    }
}

/// A single compatibility diagnostic entry describing how a request parameter
/// or feature was handled for a specific provider/model combination.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Machine-readable diagnostic code (e.g. "bridge.param.degraded").
    pub code: String,
    /// Human-readable severity.
    pub severity: DiagnosticSeverity,
    /// Path to the parameter in the request (e.g. "text.format").
    pub path: Option<String>,
    /// What action was taken.
    pub action: CompatibilityAction,
    /// Human-readable explanation.
    pub message: String,
    /// Provider name.
    pub provider: String,
    /// Model name.
    pub model: String,
    /// Optional metadata payload.
    pub metadata: Option<serde_json::Value>,
}

impl Diagnostic {
    pub fn new(
        code: impl Into<String>,
        severity: DiagnosticSeverity,
        action: CompatibilityAction,
        message: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            severity,
            path: None,
            action,
            message: message.into(),
            provider: provider.into(),
            model: model.into(),
            metadata: None,
        }
    }

    pub fn with_path(mut self, path: impl Into<String>) -> Self {
        self.path = Some(path.into());
        self
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Convert to a JSON value (for logging or attaching to responses).
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "code": self.code,
            "severity": self.severity.to_string(),
            "action": self.action.to_string(),
            "path": self.path,
            "message": self.message,
            "provider": self.provider,
            "model": self.model,
            "metadata": self.metadata,
        })
    }
}
