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
    pub queue_cap: usize,
    pub dtype: DType,
    pub keep_order: bool,
    pub resume: bool,
    pub metrics_interval: u64,
    pub no_write: bool,
}

impl Config {
    pub fn from_args(a: &Args) -> Result<Self> {
        if a.output_prefix.is_empty() { bail!("--output-prefix 不能为空"); }
        if a.input_dir.is_empty() { bail!("--input-dir 不能为空"); }
        if a.tokenizer.is_empty() { bail!("--tokenizer 不能为空"); }
        let workers = a.workers.unwrap_or_else(|| std::cmp::max(1, num_cpus::get().saturating_sub(2)));
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
            queue_cap: a.queue_cap,
            dtype: a.dtype,
            keep_order: a.keep_order,
            resume: a.resume,
            metrics_interval: a.metrics_interval,
            no_write: a.no_write,
        })
    }
}


