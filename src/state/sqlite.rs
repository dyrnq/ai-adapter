use anyhow::{anyhow, Result};
use redb::ReadableTable;
use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::task::block_in_place;

use super::store::StateStore;

pub struct SqliteStore {
    conn: Arc<Mutex<Connection>>,
    db_path: PathBuf,
}

impl SqliteStore {
    pub fn open<P: Into<PathBuf>>(path: P) -> Result<Self> {
        let path: PathBuf = path.into();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(&path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS reqlog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                method TEXT NOT NULL,
                path TEXT NOT NULL,
                status INTEGER NOT NULL,
                latency_ms INTEGER NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                req_id TEXT,
                in_tokens INTEGER,
                out_tokens INTEGER,
                cache_hit_tokens INTEGER,
                cache_miss_tokens INTEGER,
                err_code TEXT,
                err_msg TEXT,
                req_body_preview TEXT,
                resp_body_preview TEXT,
                req_headers TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_reqlog_ts ON reqlog(ts);
            CREATE INDEX IF NOT EXISTS idx_reqlog_model ON reqlog(model);
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS reasoning (
                session_id TEXT NOT NULL,
                response_id TEXT NOT NULL,
                reasoning TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (session_id, response_id)
            );",
        )?;

        // Idempotent migration: add cache columns if missing
        // Idempotent migration: add indexes (safe to re-run)
        conn.execute_batch("CREATE INDEX IF NOT EXISTS idx_reqlog_model ON reqlog(model)")
            .ok();
        for col in &["cache_hit_tokens INTEGER", "cache_miss_tokens INTEGER"] {
            let sql = format!("ALTER TABLE reqlog ADD COLUMN {}", col);
            if let Err(e) = conn.execute_batch(&sql) {
                tracing::debug!("Migration (ignored): {}", e);
            }
        }

        let db_path = path;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            db_path,
        })
    }

    /// Lock the connection, recovering from a poisoned mutex.
    fn locked(&self) -> std::sync::MutexGuard<'_, Connection> {
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl StateStore for SqliteStore {
    fn session_record(&self, session_id: &str) -> Result<()> {
        let conn = self.locked();
        conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id) VALUES (?1)",
            [session_id],
        )?;
        Ok(())
    }

    fn session_list(&self) -> Result<Vec<String>> {
        let conn = self.locked();
        let mut stmt = conn.prepare("SELECT session_id FROM sessions ORDER BY created_at DESC")?;
        let ids: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(ids)
    }

    fn reasoning_save(&self, session_id: &str, response_id: &str, reasoning: &str) -> Result<()> {
        let conn = self.locked();
        conn.execute(
            "INSERT OR REPLACE INTO reasoning (session_id, response_id, reasoning) VALUES (?1, ?2, ?3)",
            rusqlite::params![session_id, response_id, reasoning],
        )?;
        Ok(())
    }

    fn reasoning_get(&self, session_id: &str, response_id: &str) -> Result<Option<String>> {
        let conn = self.locked();
        let mut stmt = conn.prepare(
            "SELECT reasoning FROM reasoning WHERE session_id = ?1 AND response_id = ?2",
        )?;
        let result: Option<String> = stmt
            .query_row(rusqlite::params![session_id, response_id], |row| row.get(0))
            .ok();
        Ok(result)
    }

    fn reasoning_remove(&self, session_id: &str, response_id: &str) -> Result<()> {
        let conn = self.locked();
        conn.execute(
            "DELETE FROM reasoning WHERE session_id = ?1 AND response_id = ?2",
            rusqlite::params![session_id, response_id],
        )?;
        Ok(())
    }

    fn reqlog(&self, entry: &super::store::ReqlogEntry) -> Result<()> {
        let conn = self.locked();
        conn.execute(
            "INSERT INTO reqlog (ts, method, path, status, latency_ms, provider, model, req_id, in_tokens, out_tokens, cache_hit_tokens, cache_miss_tokens, err_code, err_msg, req_body_preview, resp_body_preview, req_headers)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            rusqlite::params![
                entry.ts,
                entry.method,
                entry.path,
                entry.status,
                entry.latency_ms as i64,
                entry.provider,
                entry.model,
                entry.req_id,
                entry.in_tokens,
                entry.out_tokens,
                entry.cache_hit_tokens,
                entry.cache_miss_tokens,
                entry.err_code,
                entry.err_msg,
                entry.req_body_preview,
                entry.resp_body_preview,
                entry.req_headers,
            ],
        )?;
        Ok(())
    }

    fn import_from_redb(&self, redb_path: &Path) -> Result<()> {
        if !redb_path.exists() {
            return Ok(());
        }
        tracing::info!(
            "Migrating redb data from {} to SQLite...",
            redb_path.display()
        );
        let redb_db = block_in_place(|| {
            redb::Database::open(redb_path)
                .map_err(|e| anyhow!("Failed to open redb for migration: {}", e))
        })?;

        let read_txn = redb_db
            .begin_read()
            .map_err(|e| anyhow!("Redb read txn failed: {}", e))?;

        // Migrate sessions
        let session_table_def: redb::TableDefinition<&str, &str> =
            redb::TableDefinition::new("session");
        if let Ok(table) = read_txn.open_table(session_table_def) {
            let conn = self.locked();
            for entry in table.iter().map_err(|e| anyhow!("Redb iter: {}", e))? {
                let (k, _) = entry.map_err(|e| anyhow!("Redb entry: {}", e))?;
                conn.execute(
                    "INSERT OR IGNORE INTO sessions (session_id) VALUES (?1)",
                    [k.value()],
                )?;
            }
            let count: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
            tracing::info!("Migrated {} sessions from redb", count);
        }

        // Migrate reasoning
        let reasoning_table_def: redb::TableDefinition<&str, &str> =
            redb::TableDefinition::new("reasoning");
        if let Ok(table) = read_txn.open_table(reasoning_table_def) {
            let conn = self.locked();
            let mut reasoning_count = 0i64;
            for entry in table.iter().map_err(|e| anyhow!("Redb iter: {}", e))? {
                let (k, v) = entry.map_err(|e| anyhow!("Redb entry: {}", e))?;
                let key = k.value();
                // Key format: "session_id\0response_id"
                if let Some(null_pos) = key.find('\0') {
                    let sid = &key[..null_pos];
                    let rid = &key[null_pos + 1..];
                    conn.execute(
                        "INSERT OR IGNORE INTO reasoning (session_id, response_id, reasoning) VALUES (?1, ?2, ?3)",
                        rusqlite::params![sid, rid, v.value()],
                    )?;
                    reasoning_count += 1;
                }
            }
            tracing::info!("Migrated {} reasoning entries from redb", reasoning_count);
        }

        let migrated_path = self.db_path.with_extension("redb.migrated");
        if let Err(e) = std::fs::rename(redb_path, &migrated_path) {
            tracing::warn!(
                "Failed to rename {} -> {}: {}",
                redb_path.display(),
                migrated_path.display(),
                e
            );
        } else {
            tracing::info!(
                "Renamed {} -> {}",
                redb_path.display(),
                migrated_path.display()
            );
        }
        Ok(())
    }
}
