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

impl std::fmt::Display for UpstreamVendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UpstreamVendor::DeepSeek => write!(f, "deepseek"),
            UpstreamVendor::OpenAI => write!(f, "openai"),
            UpstreamVendor::Anthropic => write!(f, "anthropic"),
            UpstreamVendor::XiaomiMimo => write!(f, "xiaomimimo"),
            UpstreamVendor::Auto => write!(f, "auto"),
        }
    }
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

// ---------------------------------------------------------------------------
// Provider configuration — multi-provider support
// ---------------------------------------------------------------------------

/// A single upstream provider entry in the YAML config
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Provider name identifier (used in model aliases: provider/model)
    pub name: String,
    /// Upstream base URL
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    /// Upstream API format
    pub format: String,
    /// API key
    pub apikey: Option<String>,
    /// Optional list of model names this provider serves
    #[serde(default)]
    pub models: Vec<String>,
    /// Provider-specific adapter behavior hint
    pub vendor: Option<String>,
    /// Extra headers to add to upstream requests
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Drop images from requests (for text-only upstreams)
    #[serde(rename = "dropImages", default)]
    pub drop_images: bool,
}

/// Main config structure (YAML/JSON config file)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub server: Option<ServerConfig>,
    /// List of upstream providers
    #[serde(default)]
    pub providers: Vec<ProviderConfig>,
    /// Model alias map: alias -> provider/model
    #[serde(rename = "modelAliases", default)]
    pub model_aliases: HashMap<String, String>,
    /// Global fallback base URL (legacy, overridden by providers)
    #[serde(rename = "baseUrl")]
    pub base_url: Option<String>,
    /// Global format (legacy)
    pub format: Option<String>,
    /// Global API key (legacy)
    pub apikey: Option<String>,
    /// Global model (legacy)
    pub model: Option<String>,
    #[serde(rename = "dropImages")]
    pub drop_images: Option<bool>,
    pub vendor: Option<String>,
    /// Global extra headers
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Enable request logging to SQLite
    #[serde(default)]
    pub reqlog: bool,
    /// Access log directory
    #[serde(rename = "accessLogDir")]
    pub access_log_dir: Option<String>,
    /// Log HTTP bodies
    #[serde(rename = "logHttp", default)]
    pub log_http: bool,
    /// Disable CORS
    #[serde(rename = "noCors", default)]
    pub no_cors: bool,
    /// Prefer client-supplied API key
    #[serde(rename = "preferClientKey", default)]
    pub prefer_client_key: bool,
    /// Truncate reasoning to 32KB
    #[serde(rename = "truncateReasoning", default)]
    pub truncate_reasoning: bool,
    #[serde(rename = "hideModelList", default)]
    pub hide_model_list: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    #[serde(default = "default_addr")]
    pub addr: String,
    /// Access log directory (optional, overrides global)
    #[serde(rename = "accessLogDir")]
    pub access_log_dir: Option<String>,
    /// Log HTTP body
    #[serde(rename = "logHttp", default)]
    pub log_http: bool,
    /// Disable CORS
    #[serde(rename = "noCors", default)]
    pub no_cors: bool,
    /// Prefer client-supplied API key
    #[serde(rename = "preferClientKey", default)]
    pub prefer_client_key: bool,
    /// Truncate reasoning to 32KB
    #[serde(rename = "truncateReasoning", default)]
    pub truncate_reasoning: bool,
    /// Enable request logging to SQLite
    #[serde(default)]
    pub reqlog: bool,
    /// Drop images from requests
    #[serde(rename = "dropImages", default)]
    pub drop_images: bool,
}

fn default_addr() -> String {
    "0.0.0.0:9090".to_string()
}

// ---------------------------------------------------------------------------
// Runtime config (resolved from AppConfig + CLI + env)
// ---------------------------------------------------------------------------

/// Resolved provider entry used at runtime
#[derive(Debug, Clone)]
pub struct ResolvedProvider {
    pub name: String,
    pub base_url: String,
    pub format: UpstreamFormat,
    pub api_key: Option<String>,
    pub vendor: UpstreamVendor,
    pub drop_images: bool,
    pub extra_headers: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub addr: String,
    pub cors: bool,
    pub log_http: bool,
    pub access_log_dir: Option<String>,
    pub enable_reqlog: bool,
    pub prefer_client_key: bool,
    pub truncate_reasoning: bool,
    pub drop_images: bool,
    pub hide_model_list: bool,
    /// Resolved alias map: alias -> (provider_name, upstream_model)
    pub alias_map: crate::translate::aliases::ModelAliasMap,
    /// All configured providers
    pub providers: Vec<ResolvedProvider>,
    /// Default provider name (first provider, or "default")
    pub default_provider: String,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            addr: "0.0.0.0:9090".to_string(),
            cors: true,
            log_http: false,
            access_log_dir: None,
            enable_reqlog: false,
            prefer_client_key: false,
            truncate_reasoning: false,
            drop_images: false,
            hide_model_list: false,
            alias_map: crate::translate::aliases::ModelAliasMap::new(),
            providers: Vec::new(),
            default_provider: "default".to_string(),
        }
    }
}

impl RuntimeConfig {
    /// Pretty-print the resolved config (masks api_key)
    pub fn print(&self) {
        fn mask(key: &Option<String>) -> String {
            key.as_deref()
                .map(|k| {
                    if k.len() > 8 {
                        format!("{}****{}", &k[..4], &k[k.len() - 4..])
                    } else {
                        "***".to_string()
                    }
                })
                .unwrap_or_else(|| "-".to_string())
        }

        let w = |label: &str, value: &dyn std::fmt::Display| {
            println!("  {:<18} {}", label, value);
        };

        println!("── server ──");
        w("addr:", &self.addr);
        w("cors:", &self.cors);
        w("reqlog:", &self.enable_reqlog);

        for p in &self.providers {
            println!("── {} ──", p.name);
            w("base_url:", &p.base_url);
            w("format:", &p.format);
            w("vendor:", &p.vendor);
            w("api_key:", &mask(&p.api_key));
            w("drop_images:", &p.drop_images);
        }

        if !self.alias_map.is_empty() {
            println!("── aliases ──");
            for (alias, target) in self.alias_map.iter() {
                if alias != "*" {
                    println!(
                        "  {:<18} {}/{}",
                        format!("{} →", alias),
                        target.provider,
                        target.model
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn load_config(
    config_path: Option<&PathBuf>,
    cli_base_url: Option<&str>,
    cli_format: Option<&str>,
    cli_api_key: Option<&str>,
    _cli_model: Option<&str>,
    _cli_addr: Option<&str>,
    cli_drop_images: bool,
    cli_no_cors: bool,
    cli_log_http: bool,
    cli_vendor: Option<&str>,
    cli_access_log_dir: Option<&str>,
    cli_prefer_client_key: bool,
    cli_model_alias: &[String],
    cli_enable_reqlog: bool,
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
            if let Some(d) = &server.access_log_dir {
                config.access_log_dir = Some(d.clone());
            }
            config.log_http = server.log_http || app_config.log_http;
            if server.no_cors {
                config.cors = false;
            }
            if app_config.no_cors {
                config.cors = false;
            }
            config.prefer_client_key = server.prefer_client_key || app_config.prefer_client_key;
            config.truncate_reasoning = server.truncate_reasoning || app_config.truncate_reasoning;
            config.enable_reqlog = server.reqlog || app_config.reqlog;
            config.drop_images = server.drop_images || app_config.drop_images.unwrap_or(false);
        } else {
            config.log_http = app_config.log_http;
            if app_config.no_cors {
                config.cors = false;
            }
            config.prefer_client_key = app_config.prefer_client_key;
            config.truncate_reasoning = app_config.truncate_reasoning;
            config.enable_reqlog = app_config.reqlog;
            if let Some(d) = app_config.drop_images {
                config.drop_images = d;
            }
        }

        // Apply access_log_dir from AppConfig top-level fallback
        if let Some(d) = &app_config.access_log_dir {
            if config.access_log_dir.is_none() {
                config.access_log_dir = Some(d.clone());
            }
        }

        // Load providers
        if !app_config.providers.is_empty() {
            for p in &app_config.providers {
                let vendor = match p.vendor.as_deref() {
                    Some("deepseek") => UpstreamVendor::DeepSeek,
                    Some("openai") => UpstreamVendor::OpenAI,
                    Some("anthropic") => UpstreamVendor::Anthropic,
                    Some("xiaomimimo") => UpstreamVendor::XiaomiMimo,
                    _ => UpstreamVendor::Auto.resolve(&p.base_url),
                };
                let format =
                    UpstreamFormat::from_str(&p.format).unwrap_or(UpstreamFormat::OpenAiChat);
                config.providers.push(ResolvedProvider {
                    name: p.name.clone(),
                    base_url: p.base_url.clone(),
                    format,
                    api_key: p.apikey.clone(),
                    vendor,
                    drop_images: p.drop_images,
                    extra_headers: p.headers.clone(),
                });
            }
        }

        // Legacy flat config: create a single "default" provider
        if config.providers.is_empty() && app_config.base_url.is_some() {
            let vendor = match app_config.vendor.as_deref() {
                Some("deepseek") => UpstreamVendor::DeepSeek,
                Some("openai") => UpstreamVendor::OpenAI,
                Some("anthropic") => UpstreamVendor::Anthropic,
                Some("xiaomimimo") => UpstreamVendor::XiaomiMimo,
                _ => UpstreamVendor::Auto,
            };
            let base_url = app_config.base_url.clone().unwrap_or_default();
            // Resolve auto vendor
            let vendor = vendor.resolve(&base_url);
            let format = app_config
                .format
                .as_deref()
                .and_then(UpstreamFormat::from_str)
                .unwrap_or(UpstreamFormat::OpenAiChat);
            config.providers.push(ResolvedProvider {
                name: "default".to_string(),
                base_url,
                format,
                api_key: app_config.apikey.clone(),
                vendor,
                drop_images: app_config.drop_images.unwrap_or(false),
                extra_headers: app_config.headers.clone(),
            });
        }

        // Load model aliases
        for (alias, target) in &app_config.model_aliases {
            if let Some((provider, model)) = target.split_once('/') {
                config.alias_map.insert(
                    alias,
                    crate::translate::aliases::ModelAlias {
                        provider: provider.to_string(),
                        model: model.to_string(),
                    },
                );
                tracing::info!("model alias: {} → {}", alias, target);
            } else {
                tracing::warn!(
                    "Invalid model alias value: {} (expected provider/model)",
                    target
                );
            }
        }
    }

    // 2. Apply environment variables
    if let Ok(url) = std::env::var("UPSTREAM_BASE_URL") {
        if config.providers.is_empty() {
            let vendor = UpstreamVendor::Auto.resolve(&url);
            let format = std::env::var("UPSTREAM_FORMAT")
                .ok()
                .and_then(|f| UpstreamFormat::from_str(&f))
                .unwrap_or(UpstreamFormat::OpenAiChat);
            config.providers.push(ResolvedProvider {
                name: "default".to_string(),
                base_url: url,
                format,
                api_key: std::env::var("UPSTREAM_API_KEY").ok(),
                vendor,
                drop_images: false,
                extra_headers: HashMap::new(),
            });
        }
    }

    // Load .env file if exists
    let _ = dotenvy::dotenv().ok();

    // 3. Apply CLI overrides (highest priority)
    // If CLI provides base_url, create/update a "cli" provider
    if let Some(url) = cli_base_url {
        let vendor = if let Some(v) = cli_vendor {
            match v {
                "deepseek" => UpstreamVendor::DeepSeek,
                "openai" => UpstreamVendor::OpenAI,
                "anthropic" => UpstreamVendor::Anthropic,
                "xiaomimimo" => UpstreamVendor::XiaomiMimo,
                _ => UpstreamVendor::Auto.resolve(url),
            }
        } else {
            UpstreamVendor::Auto.resolve(url)
        };
        let format = cli_format
            .and_then(UpstreamFormat::from_str)
            .unwrap_or(UpstreamFormat::OpenAiChat);

        // Replace or prepend "cli" provider
        config
            .providers
            .retain(|p| p.name != "cli" && p.name != "default");
        config.providers.insert(
            0,
            ResolvedProvider {
                name: "cli".to_string(),
                base_url: url.to_string(),
                format,
                api_key: cli_api_key.map(|s| s.to_string()),
                vendor,
                drop_images: cli_drop_images,
                extra_headers: HashMap::new(),
            },
        );
    }

    if cli_no_cors {
        config.cors = false;
    }
    config.log_http = cli_log_http || config.log_http;
    if let Some(d) = cli_access_log_dir {
        config.access_log_dir = Some(d.to_string());
    } else if config.access_log_dir.is_none() {
        if let Ok(d) = std::env::var("LOG_DIR") {
            config.access_log_dir = Some(d);
        } else if let Ok(data_dir) = std::env::var("DATA_DIR") {
            config.access_log_dir = Some(format!("{}/logs", data_dir));
        }
    }
    config.prefer_client_key = cli_prefer_client_key || config.prefer_client_key;
    config.enable_reqlog = cli_enable_reqlog || config.enable_reqlog;

    // Parse model aliases from CLI
    for raw in cli_model_alias {
        if let Some((alias, target)) = crate::translate::aliases::ModelAlias::parse(raw) {
            tracing::info!(
                "model alias: {} → {}/{}",
                alias,
                target.provider,
                target.model
            );
            config.alias_map.insert(&alias, target);
        } else {
            tracing::warn!(
                "Invalid model alias format: {} (expected alias=provider/model)",
                raw
            );
        }
    }

    // Ensure at least one provider exists
    if config.providers.is_empty() {
        let base_url = "https://api.openai.com".to_string();
        config.providers.push(ResolvedProvider {
            name: "default".to_string(),
            base_url,
            format: UpstreamFormat::OpenAiChat,
            api_key: None,
            vendor: UpstreamVendor::OpenAI,
            drop_images: false,
            extra_headers: HashMap::new(),
        });
    }

    // Set default provider to first one
    config.default_provider = config.providers[0].name.clone();

    Ok(config)
}

impl RuntimeConfig {
    #[allow(dead_code)]
    /// Resolve a model name to a provider.
    ///   - "provider/model" -> exact match
    ///   - "alias" -> alias_map lookup -> provider/model
    ///   - no match -> default_provider
    pub fn resolve_provider(&self, model: Option<&str>) -> Option<&ResolvedProvider> {
        let model = model?;
        // Direct provider/model reference
        if let Some((provider_name, _upstream_model)) = model.split_once('/') {
            return self
                .providers
                .iter()
                .find(|p| p.name == provider_name)
                .or_else(|| self.providers.first());
        }
        // Alias lookup
        if let Some(target) = self.alias_map.resolve(model) {
            return self
                .providers
                .iter()
                .find(|p| p.name == target.provider)
                .or_else(|| self.providers.first());
        }
        self.providers.first()
    }

    /// Resolve model name to (provider, upstream_model_name) in one lookup.
    /// Falls back to first provider and raw model name if no match.
    pub fn resolve_model_and_provider(&self, model: &str) -> (String, &ResolvedProvider) {
        // Direct "provider/model" reference
        if let Some((provider_name, upstream_model)) = model.split_once('/') {
            let provider = self
                .providers
                .iter()
                .find(|p| p.name == provider_name)
                .unwrap_or_else(|| &self.providers[0]);
            return (upstream_model.to_string(), provider);
        }
        // Alias lookup
        if let Some(target) = self.alias_map.resolve(model) {
            let provider = self
                .providers
                .iter()
                .find(|p| p.name == target.provider)
                .unwrap_or_else(|| &self.providers[0]);
            return (target.model.clone(), provider);
        }
        // Fallback: use model as-is, first provider
        (model.to_string(), &self.providers[0])
    }
}

// Legacy accessors for RuntimeConfig to minimize server.rs changes
impl RuntimeConfig {
    pub fn base_url(&self) -> &str {
        self.providers
            .first()
            .map(|p| p.base_url.as_str())
            .unwrap_or("https://api.openai.com")
    }
    pub fn upstream_format(&self) -> &UpstreamFormat {
        self.providers
            .first()
            .map(|p| &p.format)
            .unwrap_or(&UpstreamFormat::OpenAiChat)
    }
    pub fn api_key(&self) -> Option<&str> {
        self.providers.first().and_then(|p| p.api_key.as_deref())
    }
    pub fn vendor(&self) -> &UpstreamVendor {
        self.providers
            .first()
            .map(|p| &p.vendor)
            .unwrap_or(&UpstreamVendor::OpenAI)
    }
    pub fn drop_images(&self) -> bool {
        self.drop_images
            || self
                .providers
                .first()
                .map(|p| p.drop_images)
                .unwrap_or(false)
    }
    #[allow(dead_code)]
    pub fn extra_headers(&self) -> Option<&HashMap<String, String>> {
        self.providers.first().map(|p| &p.extra_headers)
    }
    /// Resolve model alias to upstream model name.
    /// "gpt-5.5" -> alias lookup -> "deepseek-v4-pro"
    /// "deepseek/deepseek-v4-pro" -> "deepseek-v4-pro"
    /// "" or None -> empty string
    #[allow(dead_code)]
    pub fn resolve_upstream_model(&self, model: Option<&str>) -> String {
        let model = match model {
            Some(m) if !m.is_empty() => m,
            _ => return String::new(),
        };
        // Direct provider/model reference
        if let Some((_provider, upstream_model)) = model.split_once('/') {
            return upstream_model.to_string();
        }
        // Alias lookup
        if let Some(target) = self.alias_map.resolve(model) {
            return target.model.clone();
        }
        // Return as-is
        model.to_string()
    }

    /// Legacy: return default model name from first provider.
    /// Server handlers should use resolve_provider instead.
    pub fn model(&self) -> &str {
        ""
    }

    pub fn hide_model_list(&self) -> bool {
        self.hide_model_list
    }
}
