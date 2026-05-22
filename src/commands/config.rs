use crate::config::RuntimeConfig;

pub fn run(format: &str) {
    let config = RuntimeConfig::default();
    match format {
        "json" => {
            let json = serde_json::json!({
                "server": { "addr": "0.0.0.0:9090" },
                "baseUrl": "https://api.openai.com",
                "format": "openai-chat"
            });
            println!("{}", serde_json::to_string_pretty(&json).unwrap());
        }
        _ => {
            println!("# ai-adapter config template\n");
            println!("server:");
            println!("  addr: \"{}\"", config.addr);
            println!();
            println!("baseUrl: \"{}\"", config.base_url);
            println!("format: {}", config.upstream_format);
            println!();
            println!("# Resolved defaults:");
            config.print();
        }
    }
}
