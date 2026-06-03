use crate::types::responses::TextFormat;

/// Structured output contract plan derived from `response_format` and provider capabilities.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct OutputContractPlan {
    /// Original requested format
    pub requested: Option<TextFormat>,
    /// What to send to the provider instead
    pub provider_format: Option<serde_json::Value>,
    /// If true, inject the JSON schema as instructions
    pub inject_schema_instruction: bool,
    /// If true, validate the response output is valid JSON
    pub requires_valid_json: bool,
}

/// Plan how to adapt `response_format` for a provider that may not support `json_schema`.
///
/// - Text → Text: pass through
/// - JsonObject → JsonObject: pass through
/// - JsonSchema → if supported: pass through; if not: degrade to JsonObject + inject schema
/// - JsonSchema + strict: same as above, but also validate output JSON
#[allow(dead_code)]
pub fn plan_output_contract(
    format: Option<&TextFormat>,
    supports_json_schema: bool,
) -> OutputContractPlan {
    let requested = format.cloned();

    match format {
        None | Some(TextFormat::Text) => OutputContractPlan {
            requested,
            provider_format: Some(serde_json::json!({"type": "text"})),
            inject_schema_instruction: false,
            requires_valid_json: false,
        },
        Some(TextFormat::JsonObject) => OutputContractPlan {
            requested,
            provider_format: Some(serde_json::json!({"type": "json_object"})),
            inject_schema_instruction: false,
            requires_valid_json: false,
        },
        Some(TextFormat::JsonSchema { strict, .. }) => {
            if supports_json_schema {
                // Provider supports json_schema natively — just build the JSON value
                let (name, schema, strict_val) = match format.unwrap() {
                    TextFormat::JsonSchema {
                        name,
                        schema,
                        strict,
                    } => (name.clone(), schema.clone(), *strict),
                    _ => unreachable!(),
                };
                let provider_format = serde_json::json!({
                    "type": "json_schema",
                    "json_schema": {
                        "name": name,
                        "schema": schema,
                        "strict": strict_val,
                    }
                });
                OutputContractPlan {
                    requested,
                    provider_format: Some(provider_format),
                    inject_schema_instruction: false,
                    requires_valid_json: *strict == Some(true),
                }
            } else {
                // Degrade: send json_object + inject schema as instruction
                OutputContractPlan {
                    requested,
                    provider_format: Some(serde_json::json!({"type": "json_object"})),
                    inject_schema_instruction: true,
                    requires_valid_json: *strict == Some(true),
                }
            }
        }
    }
}

/// Build a JSON schema instruction string to inject into the system prompt / instructions.
pub fn build_json_schema_instruction(format: &TextFormat) -> Option<String> {
    match format {
        TextFormat::JsonSchema {
            name: _,
            schema,
            strict,
        } => {
            let mut parts = vec![
                "You MUST respond with valid JSON only.".to_string(),
                "Do NOT include markdown, code fences, explanations, or extra text.".to_string(),
                String::new(),
                "JSON Schema:".to_string(),
                serde_json::to_string_pretty(schema).unwrap_or_default(),
            ];
            if *strict == Some(true) {
                parts.push("".to_string());
                parts.push("The response MUST strictly follow the above JSON Schema.".to_string());
            }
            Some(parts.join("\n"))
        }
        _ => None,
    }
}

/// Validate that the output text is valid JSON when required.
/// Returns an error message if validation fails.
#[allow(dead_code)]
pub fn validate_output_json(output_text: &str) -> Result<(), String> {
    serde_json::from_str::<serde_json::Value>(output_text)
        .map(|_| ())
        .map_err(|e| format!("Response is not valid JSON: {}", e))
}
