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
}

impl Config {
    /// 基于测试数据的智能worker分配策略 - 分层配置
    fn calculate_optimal_workers(total_workers: usize) -> (usize, usize, usize) {
        let (read_workers, write_workers) = match total_workers {
            // 0-32核: 1读取 + 1写入
            0..=32 => (1, 1),
            // 32-64核: 2读取 + 1写入  
            33..=64 => (2, 1),
            // 64-96核: 3读取 + 2写入
            65..=96 => (3, 2),
            // 96-128核: 4读取 + 2写入
            97..=128 => (4, 2),
            // 128+核: 按2:64:1比例分配
            _ => {
                // 对于超大核心数，使用固定比例：约3%读取，1.5%写入，95.5%分词
                let read_w = ((total_workers as f64 * 0.03).round() as usize).max(2).min(8);
                let write_w = ((total_workers as f64 * 0.015).round() as usize).max(1).min(4);
                (read_w, write_w)
            }
        };
        
        let tokenize_workers = total_workers.saturating_sub(read_workers).saturating_sub(write_workers).max(1);
        
        (read_workers, tokenize_workers, write_workers)
    }

    pub fn from_args(a: &Args) -> Result<Self> {
        if a.output_prefix.is_empty() { bail!("--output-prefix 不能为空"); }
        if a.input_dir.is_empty() { bail!("--input-dir 不能为空"); }
        if a.tokenizer.is_empty() { bail!("--tokenizer 不能为空"); }
        let workers = a.workers.unwrap_or_else(|| num_cpus::get());
        
        // 验证手动指定的worker参数
        if let Some(read_w) = a.read_workers {
            if read_w == 0 { bail!("--read-workers 必须大于0"); }
        }
        if let Some(tokenize_w) = a.tokenize_workers {
            if tokenize_w == 0 { bail!("--tokenize-workers 必须大于0"); }
        }
        if let Some(write_w) = a.write_workers {
            if write_w == 0 { bail!("--write-workers 必须大于0"); }
        }
        
        // 智能worker分配：如果用户没有手动指定，使用基于测试数据的最优配置
        let (optimal_read, optimal_tokenize, optimal_write) = Self::calculate_optimal_workers(workers);
        
        let read_workers = a.read_workers.unwrap_or(optimal_read);
        let tokenize_workers = a.tokenize_workers.unwrap_or(optimal_tokenize);
        let write_workers = a.write_workers.unwrap_or(optimal_write);
        
        // 验证最终配置的合理性
        let total_specified = read_workers + tokenize_workers + write_workers;
        if total_specified > workers * 2 {
            bail!("Worker总数({})过多，建议不超过总worker数的2倍({})", total_specified, workers * 2);
        }
        
        // 输出配置信息
        if a.read_workers.is_none() || a.tokenize_workers.is_none() || a.write_workers.is_none() {
            tracing::info!(
                "使用智能worker分配 (基于{}核CPU): read={}, tokenize={}, write={}, total={}",
                workers, read_workers, tokenize_workers, write_workers, total_specified
            );
        } else {
            tracing::info!(
                "使用手动worker分配: read={}, tokenize={}, write={}, total={}",
                read_workers, tokenize_workers, write_workers, total_specified
            );
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
            read_workers: Some(read_workers),
            tokenize_workers: Some(tokenize_workers),
            write_workers: Some(write_workers),
            queue_cap: a.queue_cap,
            dtype: a.dtype,
            keep_order: a.keep_order,
            resume: a.resume,
            metrics_interval: a.metrics_interval,
            no_write: a.no_write,
            no_tokenize: a.no_tokenize,
            target_shard_size_mb: a.target_shard_size_mb,
        })
    }
}


