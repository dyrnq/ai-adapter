/// Run `session ls` — list all sessions via HTTP API.
pub async fn run(endpoint: &str, subcommand: &str) -> anyhow::Result<()> {
    match subcommand {
        "ls" => {
            let resp = reqwest::get(format!("{}/__/session", endpoint)).await?;
            let body: serde_json::Value = resp.json().await?;
            let sessions = body["sessions"]
                .as_array()
                .map(|a| a.as_slice())
                .unwrap_or(&[]);
            if sessions.is_empty() {
                println!("No sessions.");
            } else {
                println!("Sessions:");
                for s in sessions {
                    println!("  {}", s.as_str().unwrap_or("?"));
                }
            }
            println!("Total: {}", sessions.len());
            Ok(())
        }
        other => anyhow::bail!("Unknown session subcommand: {}", other),
    }
}
