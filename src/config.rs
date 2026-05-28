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
#[derive(Default)]
pub enum UpstreamVendor {
    DeepSeek,
    OpenAI,
    Anthropic,
    XiaomiMimo,
    #[default]
    Auto,
}

#[allow(dead_code)]
impl UpstreamVendor {
    pub fn resolve(&self, base_url: &str) -> UpstreamVendor {
        match self {
            UpstreamVendor::Auto => {
                if base_url.contains("xiaomimimo") {
                    UpstreamVendor::XiaomiMimo
                } else if base_url.contains("deepseek") {
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
/// Fields mirror CLI options; use camelCase for JSON, snake_case for YAML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub server: Option<ServerConfig>,
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
    pub format: Option<String>,
    pub apikey: Option<String>,
    pub model: Option<String>,
    #[serde(rename = "dropImages")]
    pub drop_images: Option<bool>,
    pub vendor: Option<String>,
    pub headers: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_addr")]
    pub addr: String,
}

fn default_addr() -> String {
    "0.0.0.0:9090".to_string()
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub addr: String,
    pub base_url: String,
    pub upstream_format: UpstreamFormat,
    pub api_key: Option<String>,
    pub prefer_client_key: bool,
    pub model: Option<String>,
    pub api_version: Option<String>,
    pub drop_images: bool,
    pub backfill_reasoning: bool,
    pub truncate_reasoning: bool,
    pub cors: bool,
    pub log_http: bool,
    pub access_log_dir: Option<String>,
    pub extra_headers: HashMap<String, String>,
    pub vendor: UpstreamVendor,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FallbackUpstream {
    pub base_url: String,
    pub format: UpstreamFormat,
    pub api_key: Option<String>,
    pub model: Option<String>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            addr: "0.0.0.0:9090".to_string(),
            base_url: "https://api.openai.com".to_string(),
            upstream_format: UpstreamFormat::OpenAiChat,
            api_key: None,
            prefer_client_key: false,
            model: None,
            api_version: None,
            drop_images: false,
            backfill_reasoning: false,
            truncate_reasoning: false,
            cors: true,
            log_http: false,
            access_log_dir: None,
            extra_headers: HashMap::new(),
            vendor: UpstreamVendor::Auto,
        }
    }
}

impl RuntimeConfig {
    /// Pretty-print the resolved config (masks api_key)
    pub fn print(&self) {
        let masked_key = self
            .api_key
            .as_deref()
            .map(|k| {
                if k.len() > 8 {
                    format!("{}****{}", &k[..4], &k[k.len() - 4..])
                } else {
                    "****".to_string()
                }
            })
            .unwrap_or_else(|| "-".to_string());

        println!("addr:             {}", self.addr);
        println!("base_url:         {}", self.base_url);
        println!("upstream_format:  {}", self.upstream_format);
        println!("vendor:           {:?}", self.vendor);
        println!("model:            {}", self.model.as_deref().unwrap_or("-"));
        println!(
            "api_version:      {}",
            self.api_version.as_deref().unwrap_or("-")
        );
        println!("api_key:          {}", masked_key);
        println!("drop_images:      {}", self.drop_images);
        println!("backfill_reason:  {}", self.backfill_reasoning);
        println!("truncate_reasoning:   {}", self.truncate_reasoning);
        println!("cors:             {}", self.cors);
        println!("log_http:         {}", self.log_http);
        if let Some(ref d) = self.access_log_dir {
            println!("access_log_dir:   {}", d);
        }
        if !self.extra_headers.is_empty() {
            println!("extra_headers:    {} entries", self.extra_headers.len());
        }
    }
}

/// Load config from file, env vars, and CLI overrides
#[allow(clippy::too_many_arguments)]
pub fn load_config(
    config_path: Option<&PathBuf>,
    cli_base_url: Option<&str>,
    cli_format: Option<&str>,
    cli_api_key: Option<&str>,
    cli_model: Option<&str>,
    cli_addr: Option<&str>,
    cli_drop_images: bool,
    cli_no_cors: bool,
    cli_log_http: bool,
    cli_vendor: Option<&str>,
    cli_access_log_dir: Option<&str>,
    cli_prefer_client_key: bool,
) -> anyhow::Result<RuntimeConfig> {
    let mut config = RuntimeConfig::default();

    // 1. Load from config file if provided
    if let Some(path) = config_path {
        let content = std::fs::read_to_string(path)?;
        let app_config: AppConfig =
            serde_yaml::from_str(&content).or_else(|_| serde_json::from_str(&content))?;

        // Apply server config
        if let Some(server) = &app_config.server {
            config.addr = server.addr.clone();
        }

        // Apply flat upstream config (mirrors CLI)
        if let Some(ref url) = app_config.base_url {
            config.base_url = url.clone();
        }
        if let Some(ref fmt) = app_config.format {
            if let Some(f) = UpstreamFormat::from_str(fmt) {
                config.upstream_format = f;
            }
        }
        if let Some(ref key) = app_config.apikey {
            config.api_key = Some(key.clone());
        }
        if let Some(ref model) = app_config.model {
            config.model = Some(model.clone());
        }
        if let Some(d) = app_config.drop_images {
            config.drop_images = d;
        }
        if let Some(ref v) = app_config.vendor {
            config.vendor = match v.as_str() {
                "deepseek" => UpstreamVendor::DeepSeek,
                "openai" => UpstreamVendor::OpenAI,
                "anthropic" => UpstreamVendor::Anthropic,
                "xiaomimimo" => UpstreamVendor::XiaomiMimo,
                "auto" => UpstreamVendor::Auto,
                _ => UpstreamVendor::Auto,
            };
        }

        // Apply global headers
        if let Some(headers) = &app_config.headers {
            config
                .extra_headers
                .extend(headers.iter().map(|(k, v)| (k.to_lowercase(), v.clone())));
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
    if let Ok(addr) = std::env::var("ADDR") {
        config.addr = addr;
    }

    // Load .env file if exists
    let _ = dotenvy::dotenv().ok();

    // Apply env var for reasoning truncation
    if std::env::var("TRUNCATE_REASONING").as_deref() == Ok("true") {
        config.truncate_reasoning = true;
    }

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
    if let Some(addr) = cli_addr {
        config.addr = addr.to_string();
    }
    config.drop_images = cli_drop_images;
    config.prefer_client_key = cli_prefer_client_key;
    if cli_no_cors {
        config.cors = false;
    }
    config.log_http = cli_log_http || config.log_http;
    if let Some(d) = cli_access_log_dir {
        config.access_log_dir = Some(d.to_string());
    } else if config.access_log_dir.is_none() {
        if let Ok(d) = std::env::var("LOG_DIR") {
            config.access_log_dir = Some(d);
        }
    }
    if let Some(v) = cli_vendor {
        config.vendor = match v {
            "deepseek" => UpstreamVendor::DeepSeek,
            "openai" => UpstreamVendor::OpenAI,
            "anthropic" => UpstreamVendor::Anthropic,
            "auto" => UpstreamVendor::Auto,
            _ => UpstreamVendor::Auto,
        };
    }

    Ok(config)
}
