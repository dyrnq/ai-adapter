use redb::{Database, TableDefinition};
use std::sync::Arc;
use tokio::sync::RwLock;

const TABLE: TableDefinition<&str, &str> = TableDefinition::new("session");

/// Records session_ids seen by the proxy.
/// No longer stores request bodies — just tracks which sessions exist
/// for the `session ls` command.
#[derive(Clone)]
pub struct SessionStore {
    db: Arc<RwLock<Database>>,
}

impl SessionStore {
    pub fn new(db: Arc<RwLock<Database>>) -> Self {
        Self { db }
    }

    /// Record a session_id (value is timestamp, overwrites on repeat).
    pub async fn record(&self, session_id: &str) -> anyhow::Result<()> {
        let db = self.db.read().await;
        let write_txn = db.begin_write()?;
        {
            let mut table = write_txn.open_table(TABLE)?;
            table.insert(session_id, "")?;
        }
        write_txn.commit()?;
        Ok(())
    }

    /// List all session ids.
    pub async fn list(&self) -> Vec<String> {
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
            .map(|(k, _)| k.value().to_string())
            .collect()
    }
}
