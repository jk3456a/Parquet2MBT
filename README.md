# Parquet2MBT

一个高性能的Parquet到Megatron二进制格式转换工具，专为大规模语言模型训练数据预处理而设计。

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
- **峰值性能**: **155.2M tokens/s** (稳定状态)
- **最优配置**: 4读取 + 122分词 + 2写入 workers
- **I/O吞吐量**: 517 MB/s 输入，580 MB/s 输出

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
  --text-cols content \
  --tokenizer /models/tokenizer.json \
  --output-prefix /output/corpus
```

**分层自适应分配策略**（基于性能测试数据）:
- **0-32核**: 1读取 + 1写入 + 剩余分词
- **33-64核**: 2读取 + 1写入 + 剩余分词  
- **65-96核**: 3读取 + 2写入 + 剩余分词
- **97-128核**: 4读取 + 2写入 + 剩余分词
- **128+核**: 按比例分配（约3%读取 + 1.5%写入 + 95.5%分词）

**高级用户** 仍可手动指定worker数量来覆盖自动配置：
```bash
# 手动指定（仅在特殊需求时使用）
./target/release/parquet2mbt \
  --input-dir /data/corpus \
  --text-cols content \
  --tokenizer /models/tokenizer.json \
  --output-prefix /output/corpus \
  --read-workers 4 \
  --tokenize-workers 122 \
  --write-workers 2
```

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/jk3456a/Parquet2MBT.git
cd Parquet2MBT

# 构建发布版本
cargo build --release
```

### 基本用法

```bash
./target/release/parquet2mbt \
  --input-dir /data/corpus \
  --pattern "*.parquet" \
  --text-cols content,title \
  --tokenizer /path/to/tokenizer.json \
  --output-prefix /data/out/corpus
```

**注意**: 新版本采用智能worker分配，无需手动指定参数，系统会根据CPU核心数自动选择最优配置：
- **总线程数**: CPU核数
- **读取线程**: 2-4个（根据CPU核数自适应）
- **分词线程**: 大部分核心（80-90%）
- **写入线程**: 1-2个（根据CPU核数自适应）
- **批处理大小**: 8192

## 主要参数

### 必需参数
- `--input-dir <PATH>`: 输入目录路径（支持递归扫描）
- `--text-cols <COLS>`: 要提取的文本列名，逗号分隔（如：`title,content`）
- `--tokenizer <PATH>`: HuggingFace tokenizer文件路径（`.json`或`.model`）
- `--output-prefix <PATH>`: 输出文件前缀（生成`<prefix>.bin`和`<prefix>.idx`）

### 可选参数

#### 基础配置
- `--pattern <GLOB>`: 文件匹配模式（默认：`*.parquet`）
- `--batch-size <INT>`: 批处理大小（默认：8192）
- `--dtype <TYPE>`: 输出数据类型（`auto|u16|u32`，默认：`auto`）
- `--doc-boundary <TYPE>`: 文档边界策略（`row|file`，默认：`row`）
- `--concat-sep <STR>`: 多列拼接分隔符（默认：`\n`）
- `--metrics-interval <SEC>`: 指标输出间隔秒数（默认：5）
- `--resume`: 启用断点续传，跳过已完成的文件
- `--target-shard-size-mb <MB>`: 分片文件大小限制（默认：2048MB）

#### 并行处理配置
- `--workers <INT>`: 总工作线程数（默认：CPU核数）
- `--read-workers <INT>`: 读取工作线程数（默认：4）
- `--tokenize-workers <INT>`: 分词工作线程数（默认：CPU核数-6）
- `--write-workers <INT>`: 写入工作线程数（默认：2）
- `--queue-cap <INT>`: 内部队列容量（默认：8）

#### 高级功能
- `--no-write`: 仅测试模式，不写入文件（用于性能测试）
- `--no-tokenize`: 跳过分词，仅做读取和预处理（用于I/O测试）
- `--use-rayon-tokenize`: 启用Rayon在tokenize阶段内部并行化（实验性功能）


## 输出格式

### 标准输出
工具生成两个文件：
- `<prefix>.bin`: 包含所有token ID的二进制文件
- `<prefix>.idx`: 文档边界索引文件，兼容Megatron-LM格式

### 分片输出（多Write Worker）
当使用多个Write Worker时，生成分片文件：
- `<prefix>.shard_00_00001.bin`, `<prefix>.shard_01_00001.bin`, ... : 各Worker的分片数据文件
- `<prefix>.shard_00_00001.idx`, `<prefix>.shard_01_00001.idx`, ... : 对应的索引文件

分片文件命名规则：`shard_{worker_id}_{sequence}.{bin|idx}`

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
  --text-cols content \
  --tokenizer ./testdata/tokenizer/tokenizer.json \
  --output-prefix ./output/dataset
```

### 处理多列文本并拼接
```bash
./target/release/parquet2mbt \
  --input-dir /data/books \
  --text-cols message,content \
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
  --text-cols content \
  --tokenizer /models/tokenizer.json \
  --output-prefix /output/corpus \
  --target-shard-size-mb 2048

# 或手动指定（高级用户）
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

### 性能测试配置
```bash
# 纯I/O测试
./target/release/parquet2mbt \
  --input-dir /data/test \
  --text-cols content \
  --tokenizer /models/tokenizer.json \
  --output-prefix /tmp/test \
  --no-tokenize --no-write \
  --workers 8

# 完整流水线测试
./target/release/parquet2mbt \
  --input-dir /data/test \
  --text-cols content \
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
