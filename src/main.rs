mod commands;
mod config;
mod server;
mod state;
mod stream;
mod translate;
mod types;

use clap::Parser;
use std::path::PathBuf;

/// AI API Adapter - bidirectional proxy translating between OpenAI Responses, Chat Completions, and Anthropic Messages APIs
#[derive(Parser, Debug)]
#[command(name = "ai-adapter", version, about, long_about = None)]
struct Cli {
    /// Path to config file (YAML or JSON)
    #[arg(short = 'c', long = "config")]
    config: Option<PathBuf>,

    /// Upstream base URL
    #[arg(long = "base-url")]
    base_url: Option<String>,

    /// Upstream API format (anthropic, openai-chat, responses)
    #[arg(long = "upstream-format")]
    upstream_format: Option<String>,

    /// Upstream API key
    #[arg(long = "apikey")]
    api_key: Option<String>,

    /// Default model to use
    #[arg(long = "model")]
    model: Option<String>,

    /// Server listen address (e.g. 0.0.0.0:9090)
    #[arg(short = 'a', long = "addr", env = "ADDR")]
    addr: Option<String>,

    /// Log level (trace, debug, info, warn, error)
    #[arg(long = "log-level", env = "RUST_LOG")]
    log_level: Option<String>,

    /// If true, log to stderr instead of files (default true)
    #[arg(long = "logtostderr", env = "LOGTOSTDERROR")]
    logtostderr: Option<bool>,

    /// If true, log to stderr as well as files (no effect when logtostderr=true)
    #[arg(long = "alsologtostderr", env = "ALSOLOGTOSTDERROR")]
    alsologtostderr: bool,

    /// If non-empty, write log files in this directory (default: DATA_DIR/logs)
    #[arg(long = "log-dir", env = "LOG_DIR")]
    log_dir: Option<String>,

    /// Drop images from requests (for text-only upstreams)
    #[arg(long = "drop-images")]
    drop_images: bool,

    /// Vendor-specific behavior: deepseek, openai, anthropic, or auto (detect from base_url)
    #[arg(long = "vendor")]
    vendor: Option<String>,

    /// Disable CORS headers
    #[arg(long = "no-cors")]
    no_cors: bool,

    /// Log HTTP request/response bodies (may contain sensitive data)
    #[arg(long = "access-log")]
    log_http: bool,

    /// If set, write HTTP access logs to this directory (JSON, daily rotate)
    #[arg(long = "access-log-dir", env = "ACCESS_LOG_DIR")]
    access_log_dir: Option<String>,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(clap::Subcommand, Debug)]
enum Commands {
    /// Show detailed version info
    Version,
    /// Print default config template
    Config {
        /// Output format (yaml, json)
        #[arg(short = 'f', long = "format", default_value = "yaml")]
        format: String,
    },
    /// Session management
    Session {
        #[command(subcommand)]
        action: SessionAction,
    },
}

#[derive(clap::Subcommand, Debug)]
enum SessionAction {
    /// List all session entries
    Ls {
        /// Adapter endpoint (default: http://127.0.0.1:9090)
        #[arg(long, default_value = "http://127.0.0.1:9090")]
        endpoint: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    if let Some(ref cmd) = &cli.command {
        match cmd {
            Commands::Version => {
                commands::version::run();
                return Ok(());
            }
            Commands::Config { format } => {
                commands::config::run(format);
                return Ok(());
            }
            Commands::Session { action } => {
                match action {
                    SessionAction::Ls { endpoint } => {
                        commands::session::run(endpoint, "ls").await?;
                    }
                }
                return Ok(());
            }
        }
    }

    // Resolve data directory (used as default log-dir)
    let data_dir = std::env::var("DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .join(".ai-adapter")
        });
    std::fs::create_dir_all(&data_dir).ok();
    let default_log_dir = data_dir.join("logs");
    std::fs::create_dir_all(&default_log_dir).ok();

    // Initialize tracing
    let log_level = cli.log_level.as_deref().unwrap_or("info");
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level));

    let logtostderr = cli.logtostderr.unwrap_or(true);
    let alsologtostderr = cli.alsologtostderr;
    let log_dir = cli
        .log_dir
        .as_deref()
        .unwrap_or_else(|| default_log_dir.to_str().unwrap_or("/tmp/ai-adapter-logs"));

    let _log_guard: Option<tracing_appender::non_blocking::WorkerGuard>;

    // alsologtostderr only takes effect when file logging is active
    let write_file = !logtostderr || alsologtostderr;
    // Stderr is always written unless explicitly disabled
    let write_stderr = logtostderr || alsologtostderr;

    if write_file {
        let file_appender = tracing_appender::rolling::daily(log_dir, "ai-adapter");
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);
        _log_guard = Some(guard);
        let file_layer = tracing_subscriber::fmt::layer()
            .json()
            .with_target(false)
            .with_writer(non_blocking)
            .with_file(false)
            .with_line_number(false);

        if write_stderr {
            use tracing_subscriber::layer::SubscriberExt;
            use tracing_subscriber::util::SubscriberInitExt;
            let stderr_layer = tracing_subscriber::fmt::layer()
                .with_target(false)
                .with_writer(std::io::stderr);
            tracing_subscriber::registry()
                .with(env_filter)
                .with(file_layer)
                .with(stderr_layer)
                .init();
        } else {
            use tracing_subscriber::layer::SubscriberExt;
            use tracing_subscriber::util::SubscriberInitExt;
            tracing_subscriber::registry()
                .with(env_filter)
                .with(file_layer)
                .init();
        }
    } else {
        _log_guard = None;
        tracing_subscriber::fmt()
            .with_env_filter(env_filter)
            .with_target(false)
            .init();
    }

    // Log panics from spawned tasks to help debug crashes
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        tracing::error!("PANIC: {}", info);
        default_hook(info);
    }));

    // Load config
    let config = config::load_config(
        cli.config.as_ref(),
        cli.base_url.as_deref(),
        cli.upstream_format.as_deref(),
        cli.api_key.as_deref(),
        cli.model.as_deref(),
        cli.addr.as_deref(),
        cli.drop_images,
        cli.no_cors,
        cli.log_http,
        cli.vendor.as_deref(),
        cli.access_log_dir.as_deref(),
    )?;

    config.print();

    // Build router
    let data_dir = std::env::var("DATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .join(".ai-adapter")
        });
    std::fs::create_dir_all(&data_dir)?;
    std::fs::create_dir_all(data_dir.join("logs"))?;
    let cache_path = data_dir.join("state.redb");
    let db = std::sync::Arc::new(tokio::sync::RwLock::new(redb::Database::create(
        &cache_path,
    )?));
    // Ensure tables exist
    {
        let write_txn = db.read().await.begin_write()?;
        let _ = write_txn.open_table(redb::TableDefinition::<&str, &str>::new("reasoning"));
        let _ = write_txn.open_table(redb::TableDefinition::<&str, &str>::new("session"));
        write_txn.commit()?;
    }
    let reason_cache = state::ReasoningCache::new(db.clone());
    let session_store = state::SessionStore::new(db.clone());
    let router = server::build_router(
        config.clone(),
        reason_cache,
        session_store,
        config.access_log_dir.clone(),
    );

    // Bind and serve
    let listener = tokio::net::TcpListener::bind(&config.addr).await?;
    tracing::info!("Listening on {}", config.addr);

    axum::serve(listener, router)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    Ok(())
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c()
        .await
        .expect("Failed to install Ctrl+C handler");
    tracing::info!("Shutting down...");
}
