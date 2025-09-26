use crate::cli::{Args, DType, DocBoundary};
use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub input_dir: String,
    pub pattern: String,
    pub output_prefix: String,
    pub doc_boundary: DocBoundary,
    pub concat_sep: String,
    pub tokenizer: String,
    pub batch_size: usize,
    pub tokenize_chunk_rows: usize,
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
    pub target_shard_size_mb: usize,
}

impl Config {
    /// 基于实测数据的智能 worker 分配策略
    /// 原则：读是瓶颈→相对多给 read；写较轻→固定 2；其余给 tokenize
    fn calculate_optimal_workers(total_workers: usize) -> (usize, usize, usize) {
        let (read_workers, write_workers) = match total_workers {
            // 小机型
            0..=32 => (2, 1),          // 读2 写1，其余分词
            33..=64 => (3, 1),         // 读3 写1
            65..=96 => (4, 2),         // 读4 写2
            // 128核级别（用户实测最优）
            97..=160 => (6, 2),        // 读6 写2，其余分词（128核≈6/2/其余）
            // 超大核数：按比例回退，限制上限避免过分放大
            _ => {
                let read_w = ((total_workers as f64 / 21.0).round() as usize).clamp(6, 16);
                let write_w = ((total_workers as f64 / 64.0).round() as usize).clamp(2, 4);
                (read_w, write_w)
            }
        };

        let tokenize_workers = total_workers
            .saturating_sub(read_workers)
            .saturating_sub(write_workers)
            .max(1);

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
            doc_boundary: a.doc_boundary,
            concat_sep: a.concat_sep.clone(),
            tokenizer: a.tokenizer.clone(),
            batch_size: a.batch_size,
            tokenize_chunk_rows: a.tokenize_chunk_rows,
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
            target_shard_size_mb: a.target_shard_size_mb,
        })
    }
}


