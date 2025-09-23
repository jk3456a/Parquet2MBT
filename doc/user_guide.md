## Parquet2MBT 使用指南

### 功能概述

- 递归扫描输入目录，按列名投影读取 Parquet（Arrow RecordBatch 流式处理）。
- 文本预处理：多列拼接、去首尾空白。
- 批量分词（HF tokenizers），写出 Megatron `.bin/.idx`（dtype: u16/i32/auto）。
- 指标（stdout）：定期输出读/写带宽、records/s、tokens/s、累计计数。
- 日志：结构化文本（tracing），支持 `RUST_LOG` 控制级别。

### 安装与构建

```bash
cargo build --release
```

可执行文件位置：`./target/release/parquet2mbt`

### 基本用法

```bash
./target/release/parquet2mbt \
  --input-dir /data/corpus \
  --pattern "*.parquet" \
  --text-cols content,meta \
  --tokenizer /path/to/tokenizer.json \
  --output-prefix /data/out/corpus \
  --batch-size 32768 --workers $(nproc) --dtype auto --doc-boundary row \
  --metrics-interval 5
```

### 参数详细说明

#### 必需参数
- `--input-dir`：输入目录（递归扫描）
- `--text-cols`：文本列名，逗号分隔；支持多列拼接
- `--tokenizer`：HF `tokenizer.json` 或 `sentencepiece.model` 路径
- `--output-prefix`：输出前缀，生成 `<prefix>.bin/.idx`

#### 基础配置
- `--pattern`：文件匹配模式（默认 `*.parquet`）
- `--batch-size`：每批处理的行数（影响吞吐与内存，默认：8192）
- `--dtype`：`auto|u16|i32`，决定 `.bin` 元素类型（默认：`auto`）
- `--doc-boundary`：`row|file`，文档边界策略（默认：`row`）
- `--concat-sep`：多列拼接分隔符（默认 `\n`）
- `--metrics-interval`：指标输出间隔（秒），`0` 关闭（默认：5）

#### 并行处理配置
- `--workers`：总工作线程数（默认：CPU核数）
- `--read-workers`：读取工作线程数（默认：4）
- `--tokenize-workers`：分词工作线程数（默认：CPU核数-6）
- `--write-workers`：写入工作线程数（默认：2）
- `--queue-cap`：内部队列容量（默认：8）

#### 高级功能
- `--no-write`：仅测试模式，不写入文件（用于性能测试）
- `--no-tokenize`：跳过分词，仅做读取和预处理（用于I/O测试）
- `--use-rayon-tokenize`：启用Rayon内部并行化（实验性功能）
- `--target-shard-size-mb`：分片文件大小限制（默认：2048MB）
- `--resume`：启用断点续传，跳过已完成的文件

### 指标说明（stdout）

定期打印如下关键字段：
- `read_mb_per_sec`：读侧带宽（近似，以文件大小增量估算）。
- `convert_mb_per_sec`：转换写出带宽（按写入 `.bin` 的实际字节估算）。
- `records_per_sec`、`tokens_per_sec`：速率。
- `input_bytes_total`、`output_bytes_total`、`files_total`、`batches_total`、`records_total`、`tokens_total`：累计值。

作业结束后，进程会优雅停止指标线程并退出。

### 常见问题

1) 没有匹配文件：确认 `--input-dir` 与 `--pattern` 是否正确。
2) tokenizer 警告（特殊 token 缺失/ID 不一致）：更换为与模型一致的 `tokenizer.json`；或后续引入 `--special-tokens-json` 与严格校验（参考 PRD）。
3) 带宽为 0：样例数据很小或列名/分词器不匹配导致 tokens 为空；换真实数据与 tokenizer 验证。
4) 指标过多：设 `--metrics-interval 0` 关闭。

### 环境变量

#### 日志控制
```bash
RUST_LOG=info ./target/release/parquet2mbt ...
RUST_LOG=debug ./target/release/parquet2mbt ...
```

#### Rayon并行控制（仅在使用 --use-rayon-tokenize 时生效）
```bash
RAYON_NUM_THREADS=4 ./target/release/parquet2mbt --use-rayon-tokenize ...
```

### 使用示例

#### 基础单列处理
```bash
./target/release/parquet2mbt \
  --input-dir /data/corpus \
  --pattern "*.snappy.parquet" \
  --text-cols content \
  --tokenizer /path/to/tokenizer.json \
  --output-prefix /data/out/corpus_content
```

#### 高性能生产配置（推荐）
```bash
# 使用优化的默认配置（推荐）
./target/release/parquet2mbt \
  --input-dir /data/corpus \
  --text-cols content \
  --tokenizer /models/tokenizer.json \
  --output-prefix /output/corpus \
  --target-shard-size-mb 2048

# 手动指定配置（高级用户）
./target/release/parquet2mbt \
  --input-dir /data/corpus \
  --text-cols content \
  --tokenizer /models/tokenizer.json \
  --output-prefix /output/corpus \
  --read-workers 4 \
  --tokenize-workers 122 \
  --write-workers 2 \
  --batch-size 8192 \
  --target-shard-size-mb 2048
```

#### 性能测试配置
```bash
# 纯I/O性能测试
./target/release/parquet2mbt \
  --input-dir /data/test \
  --text-cols content \
  --tokenizer /models/tokenizer.json \
  --output-prefix /tmp/test \
  --no-tokenize --no-write \
  --workers 8

# 完整流水线性能测试（不写文件）
./target/release/parquet2mbt \
  --input-dir /data/test \
  --text-cols content \
  --tokenizer /models/tokenizer.json \
  --output-prefix /tmp/test \
  --no-write \
  --read-workers 4 \
  --tokenize-workers 120 \
  --write-workers 2

# Rayon并行化测试（实验性）
RAYON_NUM_THREADS=4 ./target/release/parquet2mbt \
  --input-dir /data/test \
  --text-cols content \
  --tokenizer /models/tokenizer.json \
  --output-prefix /tmp/test \
  --use-rayon-tokenize \
  --workers 64
```

### 工作线程分配策略

#### 自动分配（推荐）
```bash
# 使用优化的默认配置，无需指定worker参数
./target/release/parquet2mbt [其他参数]
```

#### 手动分配（高级用户）
```bash
# 基于性能测试的最优配置
./target/release/parquet2mbt \
  --read-workers 4 \
  --tokenize-workers 122 \
  --write-workers 2 \
  [其他参数]
```

**新默认配置说明**：
- **总线程数**: CPU核数（不再是CPU核数-2）
- **Read Workers**: 4个（经过优化测试的最佳值）
- **Tokenize Workers**: CPU核数-6（为read和write预留资源）
- **Write Workers**: 2个（相比单线程提升2.1%性能）
- **批处理大小**: 8192（经过测试的最佳值）

**分配原则**：
- **Read Workers**: 4个通常足够，过多会导致I/O竞争
- **Tokenize Workers**: 分配大部分CPU核心，这是主要瓶颈
- **Write Workers**: 2个为最佳，过多会导致磁盘I/O竞争



