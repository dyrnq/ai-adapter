use anyhow::Result;
use std::path::PathBuf;

pub async fn run_cache(
    db_path: Option<String>,
    model_filter: Option<String>,
    daily: bool,
) -> Result<()> {
    let path = resolve_db_path(db_path)?;
    let conn = rusqlite::Connection::open(&path)?;

    let where_clause = model_filter
        .as_ref()
        .map(|m| format!("WHERE model = '{}'", m.replace('\'', "''")))
        .unwrap_or_default();

    if daily {
        let sql = format!(
            "SELECT SUBSTR(ts, 1, 10) as day, model,
                    COUNT(*) as requests,
                    SUM(COALESCE(in_tokens,0)) as input_tokens,
                    SUM(COALESCE(out_tokens,0)) as output_tokens,
                    SUM(COALESCE(cache_hit_tokens,0)) as cache_hit,
                    SUM(COALESCE(cache_miss_tokens,0)) as cache_miss
             FROM reqlog
             {}
             GROUP BY day, model
             ORDER BY day, model",
            where_clause
        );

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            let day: String = row.get(0)?;
            let model: String = row.get(1)?;
            let requests: i64 = row.get(2)?;
            let in_tok: i64 = row.get(3)?;
            let out_tok: i64 = row.get(4)?;
            let cache_hit: i64 = row.get(5)?;
            let cache_miss: i64 = row.get(6)?;
            Ok((day, model, requests, in_tok, out_tok, cache_hit, cache_miss))
        })?;

        for row in rows {
            let (day, model, req, in_tok, out_tok, hit, miss) = row?;
            let total_input = hit + miss;
            let hit_rate = if total_input > 0 {
                hit as f64 / total_input as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "  {:>10}  {:<20}  {:>6} req  {:>10} in  {:>10} out  {:>5.1}% cache hit",
                day, model, req, in_tok, out_tok, hit_rate
            );
        }
    } else {
        let sql = format!(
            "SELECT model,
                    COUNT(*) as requests,
                    SUM(COALESCE(in_tokens,0)) as input_tokens,
                    SUM(COALESCE(out_tokens,0)) as output_tokens,
                    SUM(COALESCE(cache_hit_tokens,0)) as cache_hit,
                    SUM(COALESCE(cache_miss_tokens,0)) as cache_miss
             FROM reqlog
             {}
             GROUP BY model
             ORDER BY model",
            where_clause
        );

        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            let model: String = row.get(0)?;
            let requests: i64 = row.get(1)?;
            let in_tok: i64 = row.get(2)?;
            let out_tok: i64 = row.get(3)?;
            let cache_hit: i64 = row.get(4)?;
            let cache_miss: i64 = row.get(5)?;
            Ok((model, requests, in_tok, out_tok, cache_hit, cache_miss))
        })?;

        let mut has_data = false;
        for row in rows {
            has_data = true;
            let (model, req, in_tok, out_tok, hit, miss) = row?;
            let total_input = hit + miss;
            let hit_rate = if total_input > 0 {
                hit as f64 / total_input as f64 * 100.0
            } else {
                0.0
            };
            let avg_output = if req > 0 { out_tok / req } else { 0 };
            println!(
                "  {:<25}  {:>6} req  {:>10} in  {:>10} out  {:>6.1}% cache hit  {:>5} avg out/req",
                model, req, in_tok, out_tok, hit_rate, avg_output
            );
        }
        if !has_data {
            println!("  (no data in reqlog)");
        }
    }

    Ok(())
}

fn resolve_db_path(override_path: Option<String>) -> Result<PathBuf> {
    if let Some(p) = override_path {
        return Ok(PathBuf::from(p));
    }
    if let Ok(p) = std::env::var("AI_ADAPTER_DB") {
        return Ok(PathBuf::from(p));
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Ok(PathBuf::from(home).join(".ai-adapter").join("adapter.db"))
}
