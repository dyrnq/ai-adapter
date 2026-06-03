use anyhow::Result;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Request log entry.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct ReqlogEntry {
    pub ts: String,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub latency_ms: u128,
    pub provider: String,
    pub model: String,
    pub req_id: Option<String>,
    pub in_tokens: Option<u32>,
    pub out_tokens: Option<u32>,
    pub cache_hit_tokens: Option<u32>,
    pub cache_miss_tokens: Option<u32>,
    pub err_code: Option<String>,
    pub err_msg: Option<String>,
    pub req_body_preview: Option<String>,
    pub resp_body_preview: Option<String>,
    pub req_headers: Option<String>,
}

/// Unified state store trait.
/// Implementations: RedbStore (legacy), SqliteStore (new hot-swappable).
pub trait StateStore: Send + Sync {
    // -- sessions --
    fn session_record(&self, session_id: &str) -> Result<()>;
    fn session_list(&self) -> Result<Vec<String>>;

    // -- reasoning cache --
    fn reasoning_save(&self, session_id: &str, response_id: &str, reasoning: &str) -> Result<()>;
    fn reasoning_get(&self, session_id: &str, response_id: &str) -> Result<Option<String>>;
    fn reasoning_remove(&self, session_id: &str, response_id: &str) -> Result<()>;

    // -- request logging --
    #[allow(dead_code)]
    fn reqlog(&self, entry: &ReqlogEntry) -> Result<()>;

    // -- migration --
    fn import_from_redb(&self, redb_path: &Path) -> Result<()>;
}

/// Detect and return the appropriate state store.
/// If SQLITE_PATH env is set → SqliteStore.
/// Otherwise → RedbStore (legacy default).
pub fn detect_store() -> Arc<dyn StateStore> {
    if let Ok(path) = std::env::var("AI_ADAPTER_DB") {
        tracing::info!("Using SQLite database: {}", path);
        match crate::state::sqlite::SqliteStore::open(path) {
            Ok(store) => {
                // Auto-migrate from redb if legacy file exists
                let redb_path = PathBuf::from(
                    std::env::var("STATE_REDB_PATH").unwrap_or_else(|_| "state.redb".to_string()),
                );
                if redb_path.exists() {
                    if let Err(e) = store.import_from_redb(&redb_path) {
                        tracing::warn!("Failed to import redb data: {}", e);
                    }
                }
                Arc::new(store)
            }
            Err(e) => {
                tracing::error!("Failed to open SQLite store: {}, falling back to redb", e);
                Arc::new(super::redb_store::RedbStore::new(PathBuf::from(
                    "state.redb",
                )))
            }
        }
    } else {
        let path = PathBuf::from(
            std::env::var("STATE_REDB_PATH").unwrap_or_else(|_| "state.redb".to_string()),
        );
        tracing::info!("Using Redb state store: {}", path.display());
        Arc::new(super::redb_store::RedbStore::new(path))
    }
}
