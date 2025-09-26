# Parquet2MBT

一个高性能的Parquet到Megatron二进制格式 (Megatron Binary Type, MBT) 转换工具，专为大规模语言模型训练数据预处理而设计。

## 功能特性

- 🚀 **高性能**: 多线程流水线架构，充分利用CPU和I/O资源
- 📊 **批量处理**: 支持递归目录扫描，批量转换Parquet文件
- 🔧 **灵活配置**: 支持多列文本拼接、自定义分词器、可配置输出格式
- 📈 **实时监控**: 内置指标监控，实时显示转换进度和性能统计
- 🛡️ **可靠性**: 支持断点续传、原子写入、错误恢复
- 🎯 **兼容性**: 输出格式完全兼容Megatron-LM的IndexedDataset

## 性能基准

### 测试环境
- **硬件**: 128核CPU + 8×RTX4090 GPU + 高速SSD
- **数据集**: zh__CCI4.0-M2-Base-v1 (250个Parquet文件)
- **分词器**: 标准中文分词器 (~100K词汇表)

### 性能表现
- **峰值性能**: **151.6M tokens/s** (稳定状态)
- **最优配置**: 5读取 + 120分词 + 3写入 workers (在128核CPU, batch-size=2048环境下)
- **I/O吞吐量**: 411.9 MB/s 输入, 578.2 MB/s 输出 (峰值性能区间)

### 时间估算

| 数据规模 | Token数量 | 预估转换时间 | 说明 |
|----------|-----------|-------------|------|
| 小规模   | 1B tokens | ~6.4秒      | 单本小说/文档集 |
| 中规模   | 10B tokens | ~1.1分钟    | 中型语料库 |
| 大规模   | 100B tokens | ~10.8分钟   | 大型预训练数据集 |
| 超大规模 | 1T tokens | ~1.8小时    | 超大规模语料库 |

**注意**: 实际转换时间受以下因素影响：
- 硬件配置（CPU核数、内存带宽、存储速度）
- 数据特征（文本长度、压缩比、文件数量）
- 分词器复杂度（词汇表大小、算法类型）
- 系统负载（其他进程占用、I/O竞争）

### 🤖 智能Worker分配

**无需手动配置！** 系统会根据CPU核心数自动选择最优的worker分配策略：

```bash
# 系统会自动检测CPU核心数并应用最优配置
./target/release/parquet2mbt \
  --input-dir /data/corpus \
  --tokenizer /models/tokenizer.json \
  --output-prefix /output/corpus
```

**分层自适应分配策略**（基于性能测试数据）:
- **0-32核**: 2读取 + 1写入 + 剩余分词
- **33-64核**: 3读取 + 1写入 + 剩余分词
- **65-96核**: 4读取 + 2写入 + 剩余分词
- **97-160核**: 6读取 + 2写入 + 剩余分词
- **160+核**: 按比例分配（读/写worker有上限，避免过度分配）

**高级用户** 仍可手动指定worker数量来覆盖自动配置：
```bash
# 手动指定（仅在特殊需求时使用）
./target/release/parquet2mbt \
  --input-dir /data/corpus \
  --tokenizer /models/tokenizer.json \
  --output-prefix /output/corpus \
  --read-workers 4 \
  --tokenize-workers 122 \
  --write-workers 2
```

## 快速开始（两种方式）

### 方法一：使用 Docker

### 1. 构建Docker镜像
```bash
docker build -t parquet2mbt:latest -f deploy/docker/Dockerfile .
```
> **提示**: 如遇权限问题，请先将当前用户加入 `docker` 组。

### 2. 运行转换任务
```bash
# 准备本地目录
DATA_DIR=/path/to/your/parquet_files
OUT_DIR=/path/to/your/output_dir
TOKENIZER=/path/to/your/tokenizer.json

# 使用Docker运行
docker run --rm --init \
  -v "$DATA_DIR":/data:ro \
  -v "$OUT_DIR":/out \
  -v "$TOKENIZER":/models/tokenizer.json:ro \
  parquet2mbt:latest \
  parquet2mbt \
  --input-dir /data \
  --tokenizer /models/tokenizer.json \
  --output-prefix /out/dataset
```

### 方法二：使用预编译二进制（推荐）

1. 前往 Releases 页面下载与你平台匹配的二进制：
   - `parquet2mbt`
2. 校验与赋权并运行：
```bash
chmod +x parquet2mbt

# 最小示例
./parquet2mbt \
  --input-dir /data/corpus \
  --tokenizer /models/tokenizer.json \
  --output-prefix /output/corpus
```

## 主要参数概览

仅列出最核心的参数，**所有参数的详细说明请参见 [用户指南](doc/user_guide.md)**。

- `--input-dir <PATH>`: **(必需)** 输入Parquet文件所在目录
- `--output-prefix <PATH>`: **(必需)** 输出文件前缀
- `--tokenizer <PATH>`: **(必需)** Tokenizer文件路径
- `--batch-size <INT>`: 批处理大小，影响内存与性能
- `--target-shard-size-mb <MB>`: 输出分片的目标大小（MB）
- `--no-write`: 测试模式，不产生输出文件
- `--help`: 显示全部参数

---

## 数据集组织结构与列要求

### 目录与文件
- **输入目录**：通过 `--input-dir` 指定，工具会对该目录进行递归扫描。
- **文件匹配**：通过 `--pattern` 指定，默认 `*.parquet`，仅匹配文件名（不匹配完整路径）。
- **稳定顺序**：扫描到的文件会进行字典序排序，确保多次运行具有稳定顺序。

- **数据总路径（corpus root）**：包含多个数据集的顶层目录。将 `--input-dir` 指向该目录时，会递归读取其下所有匹配的 Parquet 文件，并合并处理到同一输出前缀中。
- **数据集路径（dataset）**：单个数据集所在目录。将 `--input-dir` 指向该目录时，仅处理该数据集（同样递归其子目录）。
- **合并行为**：无论是指向数据总路径还是单个数据集，所有读取到的样本会按扫描顺序被编码并写入同一组 `<prefix>.bin/.idx`（含分片轮转）。如需区分不同数据集，建议分别运行，使用不同的 `--output-prefix`。

### 支持的列模式（二选一，优先使用 chatml/messages）
- **ChatML 消息列表（推荐）**：列名为 `messages`，Arrow 类型需为 `List<Struct{ role: Utf8/LargeUtf8, content: Utf8/LargeUtf8 }>` 或其 LargeList 变体。
  - 每行表示一段对话，内部元素为包含 `role` 与 `content` 的结构体。
  - 渲染格式：每条消息转换为 `<|im_start|>{role}\n{content}<|im_end|>\n` 并顺序拼接。
  - 字段名大小写不敏感，但需能解析为 `role` 与 `content`。
- **纯文本列**：列名为 `content`，类型为 `Utf8/LargeUtf8` 或 `Binary/LargeBinary`（二进制将按 UTF-8 解码）。

若同一文件同时含有 `messages` 与 `content`，将优先使用 `messages`；当两者都不存在时，工具会报错并中止：`未发现文本列：需要列名 'content' 或 'messages'`。

### 拼接与文档边界
- **多列拼接**：当前实现默认选择 `messages` 或 `content` 单列作为文本来源；若需多列拼接，建议在上游预处理为单列 `content`（可使用 `--concat-sep` 控制拼接分隔符的下游表现）。
- **文档边界 `--doc-boundary`**：当前按行（Row）处理；`File` 模式为预留选项，不改变现有分词粒度。
- **换行规整**：默认会将连续≥3个换行压缩为2个（可通过环境变量 `P2MBT_COLLAPSE_NEWLINES=0` 关闭）。

### 最小可用示例
```
data_root
 ├─ dataset_a/
 │   ├─ 0001.parquet
 │   └─ 0002.parquet
 └─ dataset_b/
     ├─ part-0001.parquet
     └─ part-0002.parquet

# 递归读取 data_root 下所有匹配文件，并合并写入同一输出前缀，但是会在前缀后有数据集的区分，表现为用'.'分隔
parquet2mbt \
  --input-dir ./data_root \
  --tokenizer ./testdata/tokenizer.json \
  --output-prefix ./output/all_datasets

output
 ├─ all_datasets.dataset_a.shard_00_00001.bin
 ├─ all_datasets.dataset_a.shard_00_00001.idx
 ├─ all_datasets.dataset_a.shard_01_00001.bin
 ├─ all_datasets.dataset_a.shard_01_00001.idx
 ├─ all_datasets.dataset_b.shard_00_00001.bin
 ├─ all_datasets.dataset_b.shard_00_00001.idx
 ├─ all_datasets.dataset_b.shard_01_00001.bin
 └─ all_datasets.dataset_b.shard_01_00001.idx

```

### 常见问题
- **列名不叫 content/messages 可以吗？** 当前需要列名严格为 `content` 或 `messages`；请在数据准备阶段重命名列。
- **messages 的元素结构字段顺序不一致？** 只要字段名能识别为 `role` 与 `content` 即可，顺序无关；大小写不敏感。
- **二进制内容如何处理？** 将按 UTF-8 解码（无效字节替换）；如需精准控制请在上游转换为 Utf8。

---

## 本地构建（面向开发者）

```bash
# 1. 克隆仓库
git clone https://github.com/jk3456a/Parquet2MBT.git
cd Parquet2MBT

# 2. 编译
cargo build --release

# 3. 运行
./target/release/parquet2mbt --help
```

---

## 输出格式

工具会生成与[Megatron-LM](https://github.com/NVIDIA/Megatron-LM)完全兼容的 `.bin` 和 `.idx` 文件。

- `<prefix>.bin`: 包含所有Token ID的二进制数据。
- `<prefix>.idx`: 索引文件，记录每个文档在 `.bin` 文件中的偏移量。

当使用多个写入线程时，会自动生成分片文件，如 `<prefix>.shard_00_00001.bin`。

## 性能监控

工具会定期输出性能指标：
```
read_mb_per_sec: 245.2, convert_mb_per_sec: 89.4, records_per_sec: 12450, tokens_per_sec: 2.1M
files: 15/100, batches: 1250, records: 1.2M, tokens: 245.8M, input: 2.1GB, output: 983MB
```

## 环境变量

- `RUST_LOG`: 控制日志级别（`debug|info|warn|error`）
- `RAYON_NUM_THREADS`: 控制Rayon内部并行线程数（仅在使用`--use-rayon-tokenize`时生效）

## 示例

### 处理单列文本
```bash
./target/release/parquet2mbt \
  --input-dir ./testdata/data \
  --tokenizer ./testdata/tokenizer.json \
  --output-prefix ./output/dataset
```

### 处理多列文本并拼接
```bash
./target/release/parquet2mbt \
  --input-dir /data/books \
  --concat-sep "\n\n" \
  --tokenizer /models/tokenizer.json \
  --output-prefix /output/books_dataset \
  --dtype u32
```

### 高性能生产配置（推荐）
```bash
# 使用默认配置（推荐）
./target/release/parquet2mbt \
  --input-dir /data/corpus \
  --tokenizer /models/tokenizer.json \
  --output-prefix /output/corpus 

# 或手动指定（高级用户）
./target/release/parquet2mbt \
  --input-dir /data/corpus \
  --tokenizer /models/tokenizer.json \
  --output-prefix /output/corpus \
  --read-workers 4 \
  --tokenize-workers 122 \
  --write-workers 2 \
  --batch-size 8192 \
  --target-shard-size-mb 2048
```

### 性能测试配置
```bash
# 纯I/O测试
./target/release/parquet2mbt \
  --input-dir /data/test \
  --tokenizer /models/tokenizer.json \
  --output-prefix /tmp/test \
  --no-write \
  --workers 8

# 完整流水线测试
./target/release/parquet2mbt \
  --input-dir /data/test \
  --tokenizer /models/tokenizer.json \
  --output-prefix /tmp/test \
  --no-write \
  --workers 128
```

## 项目结构

```
├── src/
│   ├── main.rs              # 程序入口
│   ├── cli/                 # 命令行参数解析
│   ├── config/              # 配置管理
│   ├── scanner/             # 文件扫描
│   ├── reader/              # Parquet文件读取
│   ├── preprocessor/        # 文本预处理
│   ├── tokenizer/           # 分词处理
│   ├── writer/              # 二进制文件写入
│   ├── index/               # 索引文件生成
│   └── pipeline/            # 流水线调度
├── doc/                     # 项目文档
├── testdata/                # 测试数据
└── tools/                   # 辅助工具脚本
```

## 依赖项

主要依赖：
- `arrow` & `parquet`: Apache Arrow生态，用于高效读取Parquet文件
- `tokenizers`: HuggingFace分词器Rust实现
- `rayon`: 数据并行处理
- `clap`: 命令行参数解析
- `tracing`: 结构化日志

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 贡献

欢迎提交Issue和Pull Request！

## 相关文档

- [产品需求文档 (PRD)](doc/PRD.md) - 详细的功能规格和架构设计
- [用户指南](doc/user_guide.md) - 详细的使用说明和参数解释
- [风洞测试分析报告](doc/windtunnel_analysis_report.md) - 完整的性能测试和优化分析
- [性能报告](doc/disk_speed_report.md) - 存储性能测试和建议
