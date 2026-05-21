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

    pub fn db_ref(&self) -> &Arc<RwLock<Database>> {
        &self.db
    }

    pub async fn save(&self, session_id: &str, content: &str) -> anyhow::Result<()> {
        let db = self.db.read().await;
        let write_txn = db.begin_write()?;
        {
            let mut table = write_txn.open_table(TABLE)?;
            table.insert(session_id, content)?;
        }
        write_txn.commit()?;
        tracing::debug!(
            "SessionStore: saved {} bytes for session_id={}",
            content.len(),
            session_id
        );
        Ok(())
    }

    pub async fn get(&self, session_id: &str) -> anyhow::Result<Option<String>> {
        let db = self.db.read().await;
        let read_txn = db.begin_read()?;
        let result = {
            let table = read_txn.open_table(TABLE)?;
            table.get(session_id)?.map(|v| v.value().to_string())
        };
        Ok(result)
    }

    pub async fn remove(&self, session_id: &str) -> anyhow::Result<()> {
        let db = self.db.read().await;
        let write_txn = db.begin_write()?;
        {
            let mut table = write_txn.open_table(TABLE)?;
            table.remove(session_id)?;
        }
        write_txn.commit()?;
        Ok(())
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
