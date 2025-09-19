# Parquet2MBT

一个高性能的Parquet到Megatron二进制格式转换工具，专为大规模语言模型训练数据预处理而设计。

## 功能特性

- 🚀 **高性能**: 多线程流水线架构，充分利用CPU和I/O资源
- 📊 **批量处理**: 支持递归目录扫描，批量转换Parquet文件
- 🔧 **灵活配置**: 支持多列文本拼接、自定义分词器、可配置输出格式
- 📈 **实时监控**: 内置指标监控，实时显示转换进度和性能统计
- 🛡️ **可靠性**: 支持断点续传、原子写入、错误恢复
- 🎯 **兼容性**: 输出格式完全兼容Megatron-LM的IndexedDataset

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
  --output-prefix /data/out/corpus \
  --batch-size 32768 \
  --workers $(nproc) \
  --dtype auto
```

## 主要参数

### 必需参数
- `--input-dir <PATH>`: 输入目录路径（支持递归扫描）
- `--text-cols <COLS>`: 要提取的文本列名，逗号分隔（如：`title,content`）
- `--tokenizer <PATH>`: HuggingFace tokenizer文件路径（`.json`或`.model`）
- `--output-prefix <PATH>`: 输出文件前缀（生成`<prefix>.bin`和`<prefix>.idx`）

### 可选参数
- `--pattern <GLOB>`: 文件匹配模式（默认：`*.parquet`）
- `--batch-size <INT>`: 批处理大小（默认：32768）
- `--workers <INT>`: 分词工作线程数（默认：CPU核数-2）
- `--dtype <TYPE>`: 输出数据类型（`auto|u16|u32`，默认：`auto`）
- `--doc-boundary <TYPE>`: 文档边界策略（`row|file`，默认：`row`）
- `--concat-sep <STR>`: 多列拼接分隔符（默认：`\n`）
- `--metrics-interval <SEC>`: 指标输出间隔秒数（默认：5）
- `--resume`: 启用断点续传，跳过已完成的文件

## 输出格式

工具生成两个文件：
- `<prefix>.bin`: 包含所有token ID的二进制文件
- `<prefix>.idx`: 文档边界索引文件，兼容Megatron-LM格式

## 性能监控

工具会定期输出性能指标：
```
read_mb_per_sec: 245.2, convert_mb_per_sec: 89.4, records_per_sec: 12450, tokens_per_sec: 2.1M
files: 15/100, batches: 1250, records: 1.2M, tokens: 245.8M, input: 2.1GB, output: 983MB
```

## 环境变量

- `RUST_LOG`: 控制日志级别（`debug|info|warn|error`）
- `RAYON_NUM_THREADS`: 控制内部并行线程数

## 示例

### 处理单列文本
```bash
./target/release/parquet2mbt \
  --input-dir ./testdata/data \
  --text-cols content \
  --tokenizer ./testdata/tokenizer/tokenizer.json \
  --output-prefix ./output/dataset \
  --batch-size 16384
```

### 处理多列文本并拼接
```bash
./target/release/parquet2mbt \
  --input-dir /data/books \
  --text-cols title,content \
  --concat-sep "\n\n" \
  --tokenizer /models/tokenizer.json \
  --output-prefix /output/books_dataset \
  --workers 16 \
  --dtype u32
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
- [性能报告](doc/disk_speed_report.md) - 性能测试和优化建议
