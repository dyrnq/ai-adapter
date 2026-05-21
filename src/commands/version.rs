pub fn run() {
    println!("ai-adapter {}", env!("CARGO_PKG_VERSION"));
    println!("  commit:  {}", env!("GIT_HASH"));
    println!("  built:   {}", env!("BUILD_DATE"));
    println!("  target:  {}", env!("TARGET_TRIPLE"));
}
