use anyhow::Result;
use redb::{Database, ReadableTable, TableDefinition};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::task::block_in_place;

use super::store::StateStore;

const SESSION_TABLE: TableDefinition<&str, &str> = TableDefinition::new("session");
const REASONING_TABLE: TableDefinition<&str, &str> = TableDefinition::new("reasoning");

pub struct RedbStore {
    db: Arc<Mutex<Database>>,
}

impl RedbStore {
    /// Lock the redb database, recovering from a poisoned mutex.
    fn locked(&self) -> std::sync::MutexGuard<'_, Database> {
        self.db.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn new(path: PathBuf) -> Self {
        let db = Arc::new(Mutex::new(
            Database::create(&path).expect("Failed to open redb database"),
        ));
        let db_clone = db.clone();
        block_in_place(move || {
            let db_guard = db_clone.lock().unwrap_or_else(|e| e.into_inner());
            let write_txn = db_guard.begin_write().expect("redb write txn failed");
            let _ = write_txn.open_table(SESSION_TABLE);
            let _ = write_txn.open_table(REASONING_TABLE);
            write_txn.commit().expect("redb commit failed");
        });
        Self { db }
    }
}

impl StateStore for RedbStore {
    fn session_record(&self, session_id: &str) -> Result<()> {
        let db = self.locked();
        let write_txn = db.begin_write()?;
        {
            let mut table = write_txn.open_table(SESSION_TABLE)?;
            table.insert(session_id, "")?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn session_list(&self) -> Result<Vec<String>> {
        let db = self.locked();
        let read_txn = db.begin_read()?;
        let table = read_txn.open_table(SESSION_TABLE)?;
        let result: Vec<String> = table
            .iter()?
            .filter_map(|r| r.ok())
            .map(|(k, _)| k.value().to_string())
            .collect();
        Ok(result)
    }

    fn reasoning_save(&self, session_id: &str, response_id: &str, reasoning: &str) -> Result<()> {
        let key = compose_key(session_id, response_id);
        let db = self.locked();
        let write_txn = db.begin_write()?;
        {
            let mut table = write_txn.open_table(REASONING_TABLE)?;
            table.insert(key.as_str(), reasoning)?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn reasoning_get(&self, session_id: &str, response_id: &str) -> Result<Option<String>> {
        let key = compose_key(session_id, response_id);
        let db = self.locked();
        let read_txn = db.begin_read()?;
        let table = read_txn.open_table(REASONING_TABLE)?;
        Ok(table.get(key.as_str())?.map(|v| v.value().to_string()))
    }

    fn reasoning_remove(&self, session_id: &str, response_id: &str) -> Result<()> {
        let key = compose_key(session_id, response_id);
        let db = self.locked();
        let write_txn = db.begin_write()?;
        {
            let mut table = write_txn.open_table(REASONING_TABLE)?;
            table.remove(key.as_str())?;
        }
        write_txn.commit()?;
        Ok(())
    }

    fn reqlog(&self, _entry: &super::store::ReqlogEntry) -> Result<()> {
        Ok(())
    }

    fn import_from_redb(&self, _redb_path: &Path) -> Result<()> {
        Ok(())
    }
}

fn compose_key(session_id: &str, response_id: &str) -> String {
    format!("{}\0{}", session_id, response_id)
}
