use crate::cli::{Args, DType, DocBoundary};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub input_dir: String,
    pub pattern: String,
    pub output_prefix: String,
    pub text_cols: Vec<String>,
    pub doc_boundary: DocBoundary,
    pub concat_sep: String,
    pub tokenizer: String,
    pub batch_size: usize,
    pub workers: usize,
    pub read_workers: Option<usize>,
    pub tokenize_workers: Option<usize>,
    pub write_workers: Option<usize>,
    pub queue_cap: usize,
    pub dtype: DType,
    pub keep_order: bool,
    pub resume: bool,
    pub metrics_interval: u64,
    pub no_write: bool,
    pub no_tokenize: bool,
    pub target_shard_size_mb: usize,
    pub use_rayon_tokenize: bool,
}

impl Config {
    pub fn from_args(a: &Args) -> Result<Self> {
        if a.output_prefix.is_empty() { bail!("--output-prefix 不能为空"); }
        if a.input_dir.is_empty() { bail!("--input-dir 不能为空"); }
        if a.tokenizer.is_empty() { bail!("--tokenizer 不能为空"); }
        let workers = a.workers.unwrap_or_else(|| std::cmp::max(1, num_cpus::get().saturating_sub(2)));
        
        // 验证worker参数的合理性
        if let Some(read_w) = a.read_workers {
            if read_w == 0 { bail!("--read-workers 必须大于0"); }
        }
        if let Some(tokenize_w) = a.tokenize_workers {
            if tokenize_w == 0 { bail!("--tokenize-workers 必须大于0"); }
        }
        if let Some(write_w) = a.write_workers {
            if write_w == 0 { bail!("--write-workers 必须大于0"); }
        }
        
        // 如果同时指定了所有worker类型，检查总数是否合理
        if let (Some(read_w), Some(tokenize_w), Some(write_w)) = (a.read_workers, a.tokenize_workers, a.write_workers) {
            let total = read_w + tokenize_w + write_w;
            if total > workers * 2 {  // 允许一定程度的过度订阅，但不能太过分
                bail!("手动指定的worker总数({})过多，建议不超过总worker数的2倍({})", total, workers * 2);
            }
        }
        Ok(Self {
            input_dir: a.input_dir.clone(),
            pattern: a.pattern.clone(),
            output_prefix: a.output_prefix.clone(),
            text_cols: a.text_cols.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect(),
            doc_boundary: a.doc_boundary,
            concat_sep: a.concat_sep.clone(),
            tokenizer: a.tokenizer.clone(),
            batch_size: a.batch_size,
            workers,
            read_workers: a.read_workers,
            tokenize_workers: a.tokenize_workers,
            write_workers: a.write_workers,
            queue_cap: a.queue_cap,
            dtype: a.dtype,
            keep_order: a.keep_order,
            resume: a.resume,
            metrics_interval: a.metrics_interval,
            no_write: a.no_write,
            no_tokenize: a.no_tokenize,
            target_shard_size_mb: a.target_shard_size_mb,
            use_rayon_tokenize: a.use_rayon_tokenize,
        })
    }
}


