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

## 快速开始（推荐使用Docker）

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
