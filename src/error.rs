use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use std::fmt;

/// Unified error type for ai-adapter.
#[derive(Debug)]
#[allow(dead_code)]
pub enum AdapterError {
    /// Invalid client request
    BadRequest {
        message: String,
        param: Option<String>,
    },
    /// Upstream provider returned an error
    Upstream {
        status: u16,
        message: String,
        provider: String,
        model: String,
    },
    /// Bridge conversion error (protocol translation)
    Bridge {
        code: String,
        message: String,
        provider: String,
        model: String,
    },
    /// Internal server error
    Internal { message: String },
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdapterError::BadRequest { message, param } => {
                write!(f, "Bad request")?;
                if let Some(p) = param {
                    write!(f, " (param: {})", p)?;
                }
                write!(f, ": {}", message)
            }
            AdapterError::Upstream {
                status,
                message,
                provider,
                model,
            } => {
                write!(
                    f,
                    "Upstream {} error: {} on {}/{}",
                    status, message, provider, model
                )
            }
            AdapterError::Bridge {
                code,
                message,
                provider,
                model,
            } => {
                write!(
                    f,
                    "Bridge error [{}]: {} (provider={}, model={})",
                    code, message, provider, model
                )
            }
            AdapterError::Internal { message } => {
                write!(f, "Internal error: {}", message)
            }
        }
    }
}

impl std::error::Error for AdapterError {}

impl AdapterError {
    pub fn status_code(&self) -> u16 {
        match self {
            AdapterError::BadRequest { .. } => 400,
            AdapterError::Upstream { status, .. } => *status,
            AdapterError::Bridge { .. } => 502,
            AdapterError::Internal { .. } => 500,
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        let (error_type, param) = match self {
            AdapterError::BadRequest { param, .. } => ("invalid_request_error", param.clone()),
            AdapterError::Upstream { .. } => ("upstream_error", None),
            AdapterError::Bridge { .. } => ("bridge_error", None),
            AdapterError::Internal { .. } => ("server_error", None),
        };
        let mut obj = serde_json::json!({
            "error": {
                "message": self.to_string(),
                "type": error_type,
            }
        });
        if let Some(p) = param {
            obj["error"]["param"] = serde_json::Value::String(p);
        }
        obj
    }

    /// Shortcut: 400 Bad Request
    #[allow(dead_code)]
    pub fn bad_request(msg: impl Into<String>) -> Self {
        AdapterError::BadRequest {
            message: msg.into(),
            param: None,
        }
    }

    /// Shortcut: 502 Bridge error
    #[allow(dead_code)]
    pub fn bridge(
        msg: impl Into<String>,
        provider: impl Into<String>,
        model: impl Into<String>,
    ) -> Self {
        AdapterError::Bridge {
            code: "bridge_error".into(),
            message: msg.into(),
            provider: provider.into(),
            model: model.into(),
        }
    }

    /// Convert to an SSE error string for stream fallback.
    #[allow(dead_code)]
    pub fn to_sse_error(&self) -> String {
        let body = self.to_json();
        serde_json::to_string(&body).unwrap_or_else(|_| {
            r#"{"error":{"message":"internal error","type":"server_error"}}"#.to_string()
        })
    }
}

impl IntoResponse for AdapterError {
    fn into_response(self) -> Response {
        let status = self.status_code();
        let body = self.to_json();
        (
            StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR),
            axum::Json(body),
        )
            .into_response()
    }
}
