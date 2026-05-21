use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Supported upstream API formats
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamFormat {
    Anthropic,
    OpenAiChat,
    Responses,
}

/// Vendor-specific adapter behavior (defaults to auto-detect from base_url)
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum UpstreamVendor {
    DeepSeek,
    OpenAI,
    Anthropic,
    Auto,
}

impl Default for UpstreamVendor {
    fn default() -> Self {
        UpstreamVendor::Auto
    }
}

impl UpstreamVendor {
    pub fn resolve(&self, base_url: &str) -> UpstreamVendor {
        match self {
            UpstreamVendor::Auto => {
                if base_url.contains("deepseek") {
                    UpstreamVendor::DeepSeek
                } else if base_url.contains("anthropic") {
                    UpstreamVendor::Anthropic
                } else {
                    UpstreamVendor::OpenAI
                }
            }
            other => other.clone(),
        }
    }
}

impl std::fmt::Display for UpstreamFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpstreamFormat::Anthropic => write!(f, "anthropic"),
            UpstreamFormat::OpenAiChat => write!(f, "openai-chat"),
            UpstreamFormat::Responses => write!(f, "responses"),
        }
    }
}

impl UpstreamFormat {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "anthropic" => Some(UpstreamFormat::Anthropic),
            "openai-chat" | "openai_chat" | "openai" => Some(UpstreamFormat::OpenAiChat),
            "responses" => Some(UpstreamFormat::Responses),
            _ => None,
        }
    }
}

/// Main config structure (YAML/JSON config file)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub server: Option<ServerConfig>,
    pub upstreams: Option<Vec<UpstreamConfig>>,
    #[serde(rename = "currentUpstream")]
    pub current_upstream: Option<String>,
    pub headers: Option<HashMap<String, String>>,
    pub fallback: Option<FallbackConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
}

fn default_host() -> String {
    "0.0.0.0".to_string()
}
fn default_port() -> u16 {
    9090
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamConfig {
    pub name: String,
    pub format: Option<String>,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    #[serde(rename = "apiVersion")]
    pub api_version: Option<String>,
    pub apikey: Option<String>,
    pub model: Option<String>,
    #[serde(rename = "dropImages")]
    pub drop_images: Option<bool>,
    #[serde(rename = "backfillReasoning")]
    pub backfill_reasoning: Option<bool>,
    pub headers: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackConfig {
    pub enabled: Option<bool>,
    pub upstream: Option<String>,
}

/// Resolved runtime configuration
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub host: String,
    pub port: u16,
    pub base_url: String,
    pub upstream_format: UpstreamFormat,
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub api_version: Option<String>,
    pub drop_images: bool,
    pub backfill_reasoning: bool,
    pub cors: bool,
    pub log_http: bool,
    pub fallback: Option<FallbackUpstream>,
    pub extra_headers: HashMap<String, String>,
    pub vendor: UpstreamVendor,
}

#[derive(Debug, Clone)]
pub struct FallbackUpstream {
    pub base_url: String,
    pub format: UpstreamFormat,
    pub api_key: Option<String>,
    pub model: Option<String>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 9090,
            base_url: "https://api.openai.com".to_string(),
            upstream_format: UpstreamFormat::OpenAiChat,
            api_key: None,
            model: None,
            api_version: None,
            drop_images: false,
            backfill_reasoning: false,
            cors: true,
            log_http: false,
            fallback: None,
            extra_headers: HashMap::new(),
            vendor: UpstreamVendor::Auto,
        }
    }
}

/// Load config from file, env vars, and CLI overrides
pub fn load_config(
    config_path: Option<&PathBuf>,
    cli_base_url: Option<&str>,
    cli_format: Option<&str>,
    cli_api_key: Option<&str>,
    cli_model: Option<&str>,
    cli_port: Option<u16>,
    cli_host: Option<&str>,
    cli_drop_images: bool,
    cli_no_cors: bool,
    cli_log_http: bool,
    cli_vendor: Option<&str>,
) -> anyhow::Result<RuntimeConfig> {
    let mut config = RuntimeConfig::default();

    // 1. Load from config file if provided
    if let Some(path) = config_path {
        let content = std::fs::read_to_string(path)?;
        let app_config: AppConfig =
            serde_yaml::from_str(&content).or_else(|_| serde_json::from_str(&content))?;

        // Apply server config
        if let Some(server) = &app_config.server {
            config.host = server.host.clone();
            config.port = server.port;
        }

        // Apply global headers
        if let Some(headers) = &app_config.headers {
            config
                .extra_headers
                .extend(headers.iter().map(|(k, v)| (k.to_lowercase(), v.clone())));
        }

        // Apply current upstream config
        let upstream_name = app_config.current_upstream.as_deref().unwrap_or("default");
        if let Some(upstreams) = &app_config.upstreams {
            if let Some(upstream) = upstreams.iter().find(|u| u.name == upstream_name) {
                apply_upstream(&mut config, upstream, app_config.headers.as_ref());
            }

            // Apply fallback
            if let Some(fallback) = &app_config.fallback {
                if fallback.enabled.unwrap_or(false) {
                    if let Some(ref fb_name) = fallback.upstream {
                        if let Some(fb_upstream) = upstreams.iter().find(|u| &u.name == fb_name) {
                            config.fallback = Some(FallbackUpstream {
                                base_url: fb_upstream.base_url.clone(),
                                format: UpstreamFormat::from_str(
                                    fb_upstream.format.as_deref().unwrap_or("openai-chat"),
                                )
                                .unwrap_or(UpstreamFormat::OpenAiChat),
                                api_key: fb_upstream.apikey.clone(),
                                model: fb_upstream.model.clone(),
                            });
                        }
                    }
                }
            }
        }
    }

    // 2. Apply environment variables
    if let Ok(url) = std::env::var("UPSTREAM_BASE_URL") {
        config.base_url = url;
    }
    if let Ok(fmt) = std::env::var("UPSTREAM_FORMAT") {
        if let Some(f) = UpstreamFormat::from_str(&fmt) {
            config.upstream_format = f;
        }
    }
    if let Ok(key) = std::env::var("UPSTREAM_API_KEY") {
        config.api_key = Some(key);
    }
    if let Ok(model) = std::env::var("UPSTREAM_MODEL") {
        config.model = Some(model);
    }
    if let Ok(port) = std::env::var("PORT") {
        if let Ok(p) = port.parse() {
            config.port = p;
        }
    }
    if let Ok(host) = std::env::var("HOST") {
        config.host = host;
    }

    // Load .env file if exists
    let _ = dotenvy::dotenv().ok();

    // 3. Apply CLI overrides (highest priority)
    if let Some(url) = cli_base_url {
        config.base_url = url.to_string();
    }
    if let Some(fmt) = cli_format {
        if let Some(f) = UpstreamFormat::from_str(fmt) {
            config.upstream_format = f;
        }
    }
    if let Some(key) = cli_api_key {
        config.api_key = Some(key.to_string());
    }
    if let Some(model) = cli_model {
        config.model = Some(model.to_string());
    }
    if let Some(port) = cli_port {
        config.port = port;
    }
    if let Some(host) = cli_host {
        config.host = host.to_string();
    }
    config.drop_images = cli_drop_images;
    if cli_no_cors {
        config.cors = false;
    }
    config.log_http = cli_log_http || config.log_http;
    if let Some(ref v) = cli_vendor {
        config.vendor = match *v {
            "deepseek" => UpstreamVendor::DeepSeek,
            "openai" => UpstreamVendor::OpenAI,
            "anthropic" => UpstreamVendor::Anthropic,
            "auto" => UpstreamVendor::Auto,
            _ => UpstreamVendor::Auto,
        };
    }

    Ok(config)
}

fn apply_upstream(
    config: &mut RuntimeConfig,
    upstream: &UpstreamConfig,
    _global_headers: Option<&HashMap<String, String>>,
) {
    config.base_url = upstream.base_url.clone();
    if let Some(ref fmt) = upstream.format {
        if let Some(f) = UpstreamFormat::from_str(fmt) {
            config.upstream_format = f;
        }
    }
    if let Some(ref key) = upstream.apikey {
        config.api_key = Some(key.clone());
    }
    if let Some(ref model) = upstream.model {
        config.model = Some(model.clone());
    }
    if let Some(ref version) = upstream.api_version {
        config.api_version = Some(version.clone());
    }
    if let Some(drop_images) = upstream.drop_images {
        config.drop_images = drop_images;
    }
    if let Some(backfill) = upstream.backfill_reasoning {
        config.backfill_reasoning = backfill;
    }

    // Merge headers: per-upstream overrides global
    if let Some(ref upstream_headers) = upstream.headers {
        config.extra_headers.extend(
            upstream_headers
                .iter()
                .map(|(k, v)| (k.to_lowercase(), v.clone())),
        );
    }
}
