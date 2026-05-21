use redb::{Database, TableDefinition};
use std::sync::Arc;
use tokio::sync::RwLock;

const TABLE: TableDefinition<&str, &str> = TableDefinition::new("reasoning");

fn compose_key(session_id: &str, response_id: &str) -> String {
    format!("{}\0{}", session_id, response_id)
}

/// Thread-safe reasoning cache: (session_id, response_id) → reasoning_content
#[derive(Clone)]
pub struct ReasoningCache {
    db: Arc<RwLock<Database>>,
}

impl ReasoningCache {
    pub fn new(db: Arc<RwLock<Database>>) -> Self {
        Self { db }
    }

    pub async fn save(
        &self,
        session_id: &str,
        response_id: &str,
        reasoning: &str,
    ) -> anyhow::Result<()> {
        let key = compose_key(session_id, response_id);
        let db = self.db.read().await;
        let write_txn = db.begin_write()?;
        {
            let mut table = write_txn.open_table(TABLE)?;
            table.insert(key.as_str(), reasoning)?;
        }
        write_txn.commit()?;
        tracing::debug!(
            "ReasoningCache: saved {} bytes for session_id={} response_id={}",
            reasoning.len(),
            session_id,
            response_id
        );
        Ok(())
    }

    pub async fn get(&self, session_id: &str, response_id: &str) -> anyhow::Result<Option<String>> {
        let key = compose_key(session_id, response_id);
        let db = self.db.read().await;
        let read_txn = db.begin_read()?;
        let result = {
            let table = read_txn.open_table(TABLE)?;
            table.get(key.as_str())?.map(|v| v.value().to_string())
        };
        Ok(result)
    }

    pub async fn remove(&self, session_id: &str, response_id: &str) -> anyhow::Result<()> {
        let key = compose_key(session_id, response_id);
        let db = self.db.read().await;
        let write_txn = db.begin_write()?;
        {
            let mut table = write_txn.open_table(TABLE)?;
            table.remove(key.as_str())?;
        }
        write_txn.commit()?;
        Ok(())
    }
}
