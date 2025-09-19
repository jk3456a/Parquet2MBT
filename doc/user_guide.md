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

参数说明（常用）：
- `--input-dir`：输入目录（递归）。
- `--pattern`：文件匹配（默认 `*.parquet`）。
- `--text-cols`：文本列名，逗号分隔；支持多列拼接。
- `--concat-sep`：多列拼接分隔符（默认 `\n`）。
- `--tokenizer`：HF `tokenizer.json` 或 `sentencepiece.model` 路径。
- `--output-prefix`：输出前缀，生成 `<prefix>.bin/.idx`。
- `--batch-size`：每批处理的行数（影响吞吐与内存）。
- `--workers`：分词工作线程数（默认 `CPU核数-2`）。
- `--dtype`：`auto|u16|i32`，决定 `.bin` 元素类型。
- `--doc-boundary`：`row|file`，文档边界策略。
- `--metrics-interval`：指标输出间隔（秒），`0` 关闭。

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

### 环境变量（日志）

```bash
RUST_LOG=info ./target/release/parquet2mbt ...
RUST_LOG=debug ./target/release/parquet2mbt ...
```

### 示例（只读单列）

```bash
./target/release/parquet2mbt \
  --input-dir /data/corpus \
  --pattern "*.snappy.parquet" \
  --text-cols content \
  --tokenizer /path/to/tokenizer.json \
  --output-prefix /data/out/corpus_content \
  --batch-size 16384 --workers $(nproc) --dtype auto --doc-boundary row \
  --metrics-interval 5
```



