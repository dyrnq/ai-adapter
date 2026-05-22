# Build stage
FROM rust:1.88-alpine AS builder

RUN apk add --no-cache musl-dev pkgconfig

WORKDIR /app

# Cache dependencies
COPY Cargo.toml Cargo.lock ./
RUN mkdir src && echo 'fn main() {}' > src/main.rs
RUN cargo build --release
RUN rm -rf src

# Build application
COPY src/ src/
RUN cargo build --release

# Runtime stage
FROM alpine:3.21

RUN apk add --no-cache ca-certificates tzdata

# Unified data directory for state database and logs
ENV DATA_DIR=/data

RUN mkdir -p /data/logs && chown -R nobody:nobody /data

COPY --from=builder /app/target/release/ai-adapter /usr/local/bin/ai-adapter

VOLUME ["/data"]

EXPOSE 9090

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:9090/health || exit 1

USER nobody

ENTRYPOINT ["/usr/local/bin/ai-adapter"]
CMD []
