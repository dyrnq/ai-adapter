use crate::types::responses::ResponsesTool;

#[allow(dead_code)]
const CODEX_SPECIAL_TOOLS: &[&str] = &[
    "apply_patch",
    "shell",
    "local_shell",
    "mcp",
    "web_search",
    "computer",
];

#[allow(dead_code)]
pub fn degrade_tools(tools: &mut [ResponsesTool]) -> usize {
    let mut count = 0;
    for tool in tools.iter_mut() {
        let name = tool
            .get_function()
            .map(|f| f.name.clone())
            .unwrap_or_default();
        if CODEX_SPECIAL_TOOLS.contains(&name.as_str()) {
            tool.tool_type = "function".to_string();
            count += 1;
        }
    }
    count
}
