pub mod redb_store;
pub mod sqlite;
pub mod store;

pub use store::StateStore;

use std::sync::Arc;

/// Thin wrappers that delegate to the unified StateStore.
/// These maintain the existing public API while the backing store
/// can be swapped between redb and sqlite via STATE_SQLITE_PATH.

#[derive(Clone)]
pub struct SessionStore {
    store: Arc<dyn StateStore>,
}

impl SessionStore {
    pub fn new(store: Arc<dyn StateStore>) -> Self {
        Self { store }
    }

    pub async fn record(&self, session_id: &str) -> anyhow::Result<()> {
        let store = self.store.clone();
        let sid = session_id.to_string();
        tokio::task::spawn_blocking(move || store.session_record(&sid)).await?
    }

    pub async fn list(&self) -> Vec<String> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.session_list().unwrap_or_default())
            .await
            .unwrap_or_default()
    }

    /// Log an upstream request to the persistent store (SQLite).
    pub async fn log_request(&self, entry: store::ReqlogEntry) -> anyhow::Result<()> {
        let store = self.store.clone();
        tokio::task::spawn_blocking(move || store.reqlog(&entry)).await?
    }
}

#[derive(Clone)]
pub struct ReasoningCache {
    store: Arc<dyn StateStore>,
}

impl ReasoningCache {
    pub fn new(store: Arc<dyn StateStore>) -> Self {
        Self { store }
    }

    pub async fn save(
        &self,
        session_id: &str,
        response_id: &str,
        reasoning: &str,
    ) -> anyhow::Result<()> {
        let store = self.store.clone();
        let sid = session_id.to_string();
        let rid = response_id.to_string();
        let r = reasoning.to_string();
        tokio::task::spawn_blocking(move || store.reasoning_save(&sid, &rid, &r)).await?
    }

    pub async fn get(&self, session_id: &str, response_id: &str) -> anyhow::Result<Option<String>> {
        let store = self.store.clone();
        let sid = session_id.to_string();
        let rid = response_id.to_string();
        tokio::task::spawn_blocking(move || store.reasoning_get(&sid, &rid)).await?
    }

    #[allow(dead_code)]
    pub async fn remove(&self, session_id: &str, response_id: &str) -> anyhow::Result<()> {
        let store = self.store.clone();
        let sid = session_id.to_string();
        let rid = response_id.to_string();
        tokio::task::spawn_blocking(move || store.reasoning_remove(&sid, &rid)).await?
    }
}
