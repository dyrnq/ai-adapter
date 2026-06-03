use std::collections::HashMap;

/// Model alias: Codex model name → (provider_name, upstream_model)
#[derive(Debug, Clone)]
pub struct ModelAlias {
    pub provider: String,
    pub model: String,
}

/// Parsed from CLI: `--model-alias gpt-5.5=deepseek/deepseek-v4-pro`
impl ModelAlias {
    pub fn parse(raw: &str) -> Option<(String, Self)> {
        let (alias, target) = raw.split_once('=')?;
        let (provider, model) = target.split_once('/')?;
        Some((
            alias.to_string(),
            ModelAlias {
                provider: provider.to_string(),
                model: model.to_string(),
            },
        ))
    }
}

#[derive(Debug, Clone, Default)]
pub struct ModelAliasMap {
    aliases: HashMap<String, ModelAlias>,
    wildcard: Option<ModelAlias>,
}

impl ModelAliasMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, alias: &str, target: ModelAlias) {
        if alias == "*" {
            self.wildcard = Some(target);
        } else {
            self.aliases.insert(alias.to_string(), target);
        }
    }

    /// Resolve a model name. If alias found → target. If wildcard exists → wildcard.
    /// Otherwise returns None (meaning: use the default model as-is).
    #[allow(dead_code)]
    pub fn resolve(&self, model: &str) -> Option<&ModelAlias> {
        self.aliases.get(model).or(self.wildcard.as_ref())
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.aliases.is_empty() && self.wildcard.is_none()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&str, &ModelAlias)> {
        self.aliases
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .chain(self.wildcard.as_ref().map(|w| ("*", w)))
    }
}
