use anyhow::Result;
use std::path::{Path, PathBuf};
use std::collections::HashMap;
use crossbeam_channel::bounded;
use std::sync::Arc;
use std::time::Duration;

use crate::config::Config;
use crate::cli::DType;
use crate::tokenizer::Tok;
use crate::index::IdxDType;
use crate::metrics::{Metrics, spawn_stdout_reporter};
use crate::writer::RotatingWriterPool;
use crate::reader::{ReaderPool, ReadBatch};
use crate::tokenizer::{TokenizerPool, TokenizedBatch};

// 使用 reader/tokenizer pools 提供的批次类型

/// 环境变量守护器：设置 TOKENIZERS_PARALLELISM，函数结束时恢复原值
struct EnvVarGuard {
    key: String,
    original_value: Option<String>,
}

impl EnvVarGuard {
    fn new(key: &str, value: &str) -> Self {
        let original_value = std::env::var(key).ok();
        std::env::set_var(key, value);
        Self {
            key: key.to_string(),
            original_value,
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.original_value {
            Some(val) => std::env::set_var(&self.key, val),
            None => std::env::remove_var(&self.key),
        }
    }
}

/// 解析数据集名称：取 Parquet 文件的直接父目录名作为数据集名（l1）。
/// 允许 l0 与 l1 之间存在任意中间目录；若无法取得父目录名则返回 "root"。
fn derive_dataset_name(file_path: &Path, _input_root: &str) -> String {
    file_path
        .parent()
        .and_then(|p| p.file_name())
        .map(|s| s.to_string_lossy().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "root".to_string())
}

/// 流水线调度器 - 纯协调器，不包含业务逻辑
pub fn run(cfg: Config, files: Vec<PathBuf>) -> Result<()> {
    let cpu_cores = num_cpus::get();
    let file_count = files.len();
    
    // Worker分配：支持手动指定或自动分配
    let write_workers = cfg.write_workers.unwrap_or(2);
    
    let (read_workers, tokenize_workers) = {
        let read_w = if let Some(manual_read) = cfg.read_workers {
            manual_read.min(file_count).max(1)
        } else {
            4.min(file_count)  // 默认4个read workers
        };
        let tokenize_w = if let Some(manual_tokenize) = cfg.tokenize_workers {
            manual_tokenize.max(1)
        } else {
            // 默认 nproc - 6 (read=4 + write=2)
            cfg.workers.saturating_sub(6).max(1)
        };
        (read_w, tokenize_w)
    };
    
    // 验证worker分配的合理性
    let total_specified = read_workers + tokenize_workers + write_workers;
    if cfg.read_workers.is_some() || cfg.tokenize_workers.is_some() || cfg.write_workers.is_some() {
        if total_specified > cfg.workers {
            tracing::warn!(
                "手动指定的worker总数({})超过了总worker数({}), 这可能导致过度订阅",
                total_specified, cfg.workers
            );
        }
        tracing::info!(
            "使用手动worker分配: read={}, tokenize={}, write={}, total={}",
            read_workers, tokenize_workers, write_workers, total_specified
        );
    } else {
        tracing::info!(
            "使用自动worker分配: read={}, tokenize={}, write={}, total={}",
            read_workers, tokenize_workers, write_workers, total_specified
        );
    }
    
    // 根据CPU与外部tokenize worker数智能配置tokenizers内部并行，避免过度订阅
    // 单文件或外部tokenize worker不多时，开启内部并行；多文件+大量外部worker时关闭
    let enable_tokenizer_parallel = (file_count == 1)
        || (tokenize_workers <= (cpu_cores / 2).max(1));
    let _env_guard = EnvVarGuard::new(
        "TOKENIZERS_PARALLELISM",
        if enable_tokenizer_parallel { "true" } else { "false" }
    );
    
    tracing::info!(
        "Multi-stage pipeline configuration: read_workers={}, tokenize_workers={}, write_workers={}, total_cpu_cores={}", 
        read_workers, tokenize_workers, write_workers, cpu_cores
    );

    // 初始化 tokenizer
    let tpath = &cfg.tokenizer;
    let tk = Tok::from_path(tpath)?;
    let vs = tk.vocab_size(true);
    tracing::info!(tokenizer_path = tpath.as_str(), vocab_size = vs, "tokenizer loaded");
    let tok = Arc::new(tk);

    // 选择 dtype（auto 基于词表规模）
    let dtype = match cfg.dtype {
        DType::U16 => IdxDType::U16,
        DType::I32 => IdxDType::I32,
        DType::Auto => {
            let vs = tok.vocab_size(true);
            let inferred = if vs < 65500 { IdxDType::U16 } else { IdxDType::I32 };
            tracing::info!(vocab_size = vs, inferred_dtype = match inferred { IdxDType::U16 => "u16", IdxDType::I32 => "i32" }, "dtype auto selection");
            inferred
        }
    };

    // 初始化指标系统
    let metrics = Arc::new(Metrics::new());
    let (shutdown_tx, shutdown_rx) = bounded::<()>(1);
    let metrics_handle = if cfg.metrics_interval > 0 {
        Some(spawn_stdout_reporter(
            metrics.clone(),
            Duration::from_secs(cfg.metrics_interval),
            shutdown_rx,
        ))
    } else {
        None
    };

    // 多阶段流水线通信channels（有界队列以形成背压，稳定内存并耦合阶段速度）
    let qcap = cfg.queue_cap.max(2);
    let (read_tx, read_rx) = bounded::<ReadBatch>(qcap);               // Stage 1 -> Stage 2
    let (tokenize_tx, tokenize_rx) = bounded::<TokenizedBatch>(qcap);  // Stage 2 -> Stage 3

    tracing::info!(
        read_workers = read_workers,
        tokenize_workers = tokenize_workers, 
        write_workers = write_workers,
        files_count = files.len(),
        batch_size = cfg.batch_size,
        "Starting multi-stage pipeline processing"
    );

    // 阶段3：为每个写入worker创建独立的轮转写入器（无锁设计）
    let max_shard_bytes = if cfg.no_write {
        0
    } else {
        (cfg.target_shard_size_mb as u64).saturating_mul(1_048_576)
    };

    // 使用统一的 ReaderPool 与 TokenizerPool
    let reader_pool = ReaderPool::new(files.clone(), &cfg, metrics.clone(), read_tx.clone())?;
    let tokenizer_pool = TokenizerPool::new(&cfg, tok.clone(), dtype, metrics.clone(), read_rx, tokenize_tx.clone())?;

    // Reader/Tokenizer 并发由各自 Pool 控制

    // 启动写入工作线程 - 每个worker独享轮转写入器（无锁设计）
    let write_handles: Vec<_> = (0..write_workers).map(|worker_id| {
        let tokenize_rx = tokenize_rx.clone();
        let output_prefix = cfg.output_prefix.clone();
        let input_root = cfg.input_dir.clone();
        let no_write = cfg.no_write;
        let metrics = metrics.clone();
        
        std::thread::spawn(move || -> Result<usize> {
            let mut docs_written = 0;
            // dataset -> personal rotating writer
            let mut dataset_writers: HashMap<String, RotatingWriterPool> = HashMap::new();
            
            if no_write {
                // no-write 模式：仅消费数据
                tracing::debug!("Write worker {} started (no-write)", worker_id);
                while let Ok(batch) = tokenize_rx.recv() {
                    docs_written += batch.doc_lens.len();
                }
                tracing::debug!("Write worker {} finished (no-write)", worker_id);
                return Ok(docs_written);
            }

            tracing::debug!("Write worker {} started with per-dataset rotating writers", worker_id);
            
            while let Ok(batch) = tokenize_rx.recv() {
                // 基于源文件路径解析数据集
                let ds = derive_dataset_name(&batch.file_path, &input_root);
                // 获取或创建该数据集的写入器
                if !dataset_writers.contains_key(&ds) {
                    let ds_prefix = format!("{}.{}", output_prefix, ds);
                    let writer = RotatingWriterPool::new(
                        ds_prefix,
                        1,
                        max_shard_bytes,
                        dtype,
                        metrics.clone(),
                        worker_id,
                    )?;
                    dataset_writers.insert(ds.clone(), writer);
                }

                let writer = dataset_writers.get_mut(&ds).expect("writer must exist");
                // 无锁批量写入：该数据集独立的 writer
                for tokens in batch.token_data.iter() { writer.write_document(0, tokens)?; }
                docs_written += batch.doc_lens.len();
            }
            
            // 完成时 finalize 所有数据集写入器
            for (_ds, writer) in dataset_writers.into_iter() { let _ = writer.finalize_all()?; }
            
            tracing::debug!("Write worker {} finished", worker_id);
            Ok(docs_written)
        })
    }).collect();

    // 等待读取与分词池结束并关闭通道
    reader_pool.wait_all()?;
    drop(read_tx);
    tokenizer_pool.wait_all()?;
    drop(tokenize_tx);
    
    // 等待所有写入worker完成
    let mut total_docs_written: usize = 0;
    for h in write_handles { 
        match h.join() {
            Ok(res) => { total_docs_written += res?; }
            Err(_) => { anyhow::bail!("write worker panicked"); }
        }
    }

    // 个人轮转写入器已在各自线程中finalize，无需额外处理

    // 输出最终统计
    let files = metrics.files_total.load(std::sync::atomic::Ordering::Relaxed);
    let batches = metrics.batches_total.load(std::sync::atomic::Ordering::Relaxed);
    let records = metrics.records_total.load(std::sync::atomic::Ordering::Relaxed);
    let tokens = metrics.tokens_total.load(std::sync::atomic::Ordering::Relaxed);
    let input_bytes = metrics.input_bytes_total.load(std::sync::atomic::Ordering::Relaxed);
    let output_bytes = metrics.output_bytes_total.load(std::sync::atomic::Ordering::Relaxed);
    
    if !cfg.no_write {
        tracing::info!(
            total_docs = total_docs_written,
            output_prefix = cfg.output_prefix.as_str(),
            "Pipeline completed successfully"
        );
    }

    tracing::info!(
        files = files,
        batches = batches,
        records = records,
        tokens = tokens,
        input_mb = input_bytes / 1_048_576,
        output_mb = output_bytes / 1_048_576,
        "Final statistics"
    );

    // 停止 metrics reporter 并等待其输出最终 summary
    if let Some(tx) = Some(shutdown_tx) { let _ = tx.send(()); }
    if let Some(h) = metrics_handle { let _ = h.join(); }

    Ok(())
}