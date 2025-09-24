use clap::{Parser, ValueEnum};
use serde::{Serialize, Deserialize};

#[derive(Debug, Clone, Parser)]
#[command(name = "parquet2mbt", version, about = "Convert Parquet to Megatron .bin/.idx")] 
pub struct Args {
    /// 输入目录
    #[arg(long, value_name = "PATH")]
    pub input_dir: String,

    /// 文件匹配（glob），默认 *.parquet
    #[arg(long, value_name = "GLOB", default_value = "*.parquet")]
    pub pattern: String,

    /// 输出前缀（将生成 <prefix>.bin/.idx）
    #[arg(long, value_name = "PATH")]
    pub output_prefix: String,

    /// 文本列，逗号分隔
    #[arg(long, value_name = "COLS", default_value = "text")]
    pub text_cols: String,

    /// 文档边界：row 或 file
    #[arg(long, value_enum, default_value_t = DocBoundary::Row)]
    pub doc_boundary: DocBoundary,

    /// 拼接分隔符（多列或多行拼接）
    #[arg(long, value_name = "STR", default_value = "\n")]
    pub concat_sep: String,

    /// tokenizer 路径（tokenizer.json 或 sentencepiece.model）
    #[arg(long, value_name = "PATH")]
    pub tokenizer: String,

    /// 批大小（一次 encode_batch 的条数）
    #[arg(long, value_name = "INT", default_value_t = 8192)]
    pub batch_size: usize,

    /// 分词切片的行数（从读取批中再切分给tokenizer的每次条数）
    #[arg(long, value_name = "INT", default_value_t = 1024)]
    pub tokenize_chunk_rows: usize,

    /// 总工作线程数（默认：CPU核数）
    #[arg(long, value_name = "INT")]
    pub workers: Option<usize>,

    /// 读取阶段worker数量（可选，默认4个）
    #[arg(long, value_name = "INT")]
    pub read_workers: Option<usize>,

    /// 分词阶段worker数量（可选，默认nproc-6）
    #[arg(long, value_name = "INT")]
    pub tokenize_workers: Option<usize>,

    /// 写入阶段worker数量（可选，默认2个）
    #[arg(long, value_name = "INT")]
    pub write_workers: Option<usize>,

    /// 有界队列容量
    #[arg(long, value_name = "INT", default_value_t = 32)]
    pub queue_cap: usize,

    /// 元素 dtype（auto|u16|i32）
    #[arg(long, value_enum, default_value_t = DType::Auto)]
    pub dtype: DType,

    /// 保持文件顺序（牺牲吞吐）
    #[arg(long, default_value_t = false)]
    pub keep_order: bool,

    /// 断点续转：跳过已完成文件
    #[arg(long, default_value_t = false)]
    pub resume: bool,

    /// 指标输出间隔（秒），0 表示关闭 stdout 指标
    #[arg(long, value_name = "SECONDS", default_value_t = 5)]
    pub metrics_interval: u64,

    /// 不写出 .bin/.idx，仅做读/预处理/分词与指标
    #[arg(long, default_value_t = false)]
    pub no_write: bool,

    /// 不执行分词，仅做读/预处理与指标（可与no-write组合测试纯IO）
    #[arg(long, default_value_t = false)]
    pub no_tokenize: bool,

    /// 目标分片大小（MB），达到后轮转生成下一个 .bin/.idx 分片
    #[arg(long, value_name = "MB", default_value_t = 2048)]
    pub target_shard_size_mb: usize,

}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
pub enum DocBoundary { Row, File }

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
pub enum DType { Auto, U16, I32 }

pub fn parse() -> Args { Args::parse() }


