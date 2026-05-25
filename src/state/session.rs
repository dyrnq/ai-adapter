use redb::{Database, TableDefinition};
use std::sync::Arc;
use tokio::sync::RwLock;

const TABLE: TableDefinition<&str, &str> = TableDefinition::new("session");

/// Thread-safe session store.
#[derive(Clone)]
pub struct SessionStore {
    db: Arc<RwLock<Database>>,
}

impl SessionStore {
    pub fn new(db: Arc<RwLock<Database>>) -> Self {
        Self { db }
    }

    pub async fn list(&self) -> Vec<serde_json::Value> {
        let db = self.db.read().await;
        let read_txn = match db.begin_read() {
            Ok(tx) => tx,
            Err(_) => return vec![],
        };
        let table = match read_txn.open_table(TABLE) {
            Ok(t) => t,
            Err(_) => return vec![],
        };
        table
            .range::<&str>(..)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|r| r.ok())
            .map(|(k, v)| {
                serde_json::json!({
                    "id": k.value(),
                    "size": v.value().len(),
                })
            })
            .collect()
    }
}
