BINARY_NAME := ai-adapter
TARGET_DIR := target/debug

# Default target
.PHONY: all
all: build

# Build (debug by default)
.PHONY: build
build:
	cargo build

# Release build
.PHONY: release
release:
	cargo build --release

# Check
.PHONY: fmt clippy check
fmt:
	cargo fmt --all -- --check

fmt-fix:
	cargo fmt --all

clippy:
	cargo clippy --all-targets -- -D warnings

check: fmt clippy build

# Test
.PHONY: test
test:
	cargo test

# Docker
.PHONY: docker
docker:
	docker build -t $(BINARY_NAME) .

# Clean
.PHONY: clean
clean:
	cargo clean
