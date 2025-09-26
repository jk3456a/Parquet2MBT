use clap::{Parser, ValueEnum};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Parser)]
#[command(name = "parquet2mbt", version, about = "Convert Parquet to Megatron .bin/.idx")] 
pub struct Args {
    /// 输入目录
    #[arg(long, value_name = "PATH", help_heading = "必需参数")]
    pub input_dir: String,

    /// 文件匹配（glob），默认 *.parquet
    #[arg(long, value_name = "GLOB", default_value = "*.parquet", help_heading = "基础配置")]
    pub pattern: String,

    /// 输出前缀（将生成 <prefix>.bin/.idx）
    #[arg(long, value_name = "PATH", help_heading = "必需参数")]
    pub output_prefix: String,

    /// 文档边界：row 或 file
    #[arg(long, value_enum, default_value_t = DocBoundary::Row, help_heading = "基础配置")]
    pub doc_boundary: DocBoundary,

    /// 拼接分隔符（多列或多行拼接）
    #[arg(long, value_name = "STR", default_value = "\n", help_heading = "基础配置")]
    pub concat_sep: String,

    /// tokenizer 路径（tokenizer.json 或 sentencepiece.model）
    #[arg(long, value_name = "PATH", help_heading = "必需参数")]
    pub tokenizer: String,

    /// 批大小（一次 encode_batch 的条数）
    #[arg(long, value_name = "INT", default_value_t = 8192, help_heading = "基础配置")]
    pub batch_size: usize,

    /// 分词切片的行数（从读取批中再切分给tokenizer的每次条数）
    #[arg(long, value_name = "INT", default_value_t = 1024, help_heading = "基础配置")]
    pub tokenize_chunk_rows: usize,

    /// 目标分片大小（MB），达到后轮转生成下一个 .bin/.idx 分片
    #[arg(long, value_name = "MB", default_value_t = 2048, help_heading = "基础配置")]
    pub target_shard_size_mb: usize,

    /// 总工作线程数（默认：CPU核数）
    #[arg(long, value_name = "INT", help_heading = "并行处理配置")]
    pub workers: Option<usize>,

    /// 读取阶段worker数量（可选，默认按照核心数自动分配）
    #[arg(long, value_name = "INT", help_heading = "并行处理配置")]
    pub read_workers: Option<usize>,

    /// 分词阶段worker数量（可选，默认按照核心数自动分配）
    #[arg(long, value_name = "INT", help_heading = "并行处理配置")]
    pub tokenize_workers: Option<usize>,

    /// 写入阶段worker数量（可选，默认按照核心数自动分配）
    #[arg(long, value_name = "INT", help_heading = "并行处理配置")]
    pub write_workers: Option<usize>,

    /// 有界队列容量
    #[arg(long, value_name = "INT", default_value_t = 32, help_heading = "并行处理配置")]
    pub queue_cap: usize,

    /// 元素 dtype（auto|u16|i32）
    #[arg(long, value_enum, default_value_t = DType::Auto, help_heading = "基础配置")]
    pub dtype: DType,

    /// 指标输出间隔（秒），0 表示关闭 stdout 指标
    #[arg(long, value_name = "SECONDS", default_value_t = 5, help_heading = "基础配置")]
    pub metrics_interval: u64,

    /// 不写出 .bin/.idx，仅做读/预处理/分词与指标
    #[arg(long, default_value_t = false, help_heading = "高级功能")]
    pub no_write: bool,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
pub enum DocBoundary { Row, File }

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
pub enum DType { Auto, U16, I32 }

pub fn parse() -> Args { Args::parse() }


