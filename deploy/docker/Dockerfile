# Multi-stage build: compile Rust binary, then create a slim Python runtime that can run parquet2mbt and tools

ARG RUST_VERSION=1.90
ARG DEBIAN_FRONTEND=noninteractive
# 可在构建时覆盖的 pip 镜像配置（默认使用清华镜像，附带官方源作为备用）
ARG PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"
ARG PIP_EXTRA_INDEX_URL="https://pypi.org/simple"

FROM rust:${RUST_VERSION}-slim AS builder

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    pkg-config \
    cmake \
    clang \
    perl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先只复制清单与 benches，并用临时空 main 让 Cargo 识别目标后预取依赖
COPY Cargo.toml Cargo.lock ./
COPY benches ./benches
RUN mkdir -p src \
  && echo 'fn main() {}' > src/main.rs \
  && cargo fetch --locked \
  && rm -rf src

# 再复制全部源代码并一次性构建发布二进制
COPY . .
RUN cargo build --release && strip target/release/parquet2mbt || true


# Runtime image with Python tools
FROM python:3.11-slim AS runtime

# 传递构建参数到运行时阶段
ARG PIP_INDEX_URL
ARG PIP_EXTRA_INDEX_URL

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RUST_LOG=info

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
    ca-certificates \
    libgcc-s1 \
    libstdc++6 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 配置 pip 镜像与超时，加速安装
RUN python -m pip config set global.index-url "$PIP_INDEX_URL" \
  && python -m pip config set global.extra-index-url "$PIP_EXTRA_INDEX_URL" \
  && python -m pip config set global.timeout "120"

# Install minimal Python deps for tools (pyarrow/pandas/faker, tokenizers)
RUN pip install --no-cache-dir --upgrade pip \
  && pip install --no-cache-dir \
    pyarrow \
    pandas \
    faker \
    tokenizers

# Copy compiled binary and tools
COPY --from=builder /app/target/release/parquet2mbt /usr/local/bin/parquet2mbt
COPY tools ./tools

# Default command prints help
CMD ["parquet2mbt", "--help"]


