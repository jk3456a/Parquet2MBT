use anyhow::Result;
use std::path::PathBuf;
use crossbeam_channel::bounded;
use std::sync::Arc;
use std::time::Duration;

use crate::config::Config;
use crate::cli::DType;
use crate::tokenizer::Tok;
use crate::index::IdxDType;
use crate::metrics::{Metrics, spawn_stdout_reporter};
use crate::reader::open_parquet_batches_with_names;
use crate::preprocessor::extract_text_columns;
use crate::writer::RotatingWriterPool;
use std::fs;
use anyhow::Context;

// 多阶段流水线数据结构
#[derive(Debug)]
struct ReadBatch {
    texts: Vec<String>,
    file_path: PathBuf,
}

#[derive(Debug)]
struct TokenizedBatch {
    doc_lens: Vec<u32>,
    token_data: Vec<Vec<u32>>,
    file_path: PathBuf,
}

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

/// 流水线调度器 - 纯协调器，不包含业务逻辑
pub fn run(cfg: Config, files: Vec<PathBuf>) -> Result<()> {
    let cpu_cores = num_cpus::get();
    let file_count = files.len();
    
    // Worker分配：支持手动指定或自动分配
    let write_workers = cfg.write_workers.unwrap_or(2);
    
    let (read_workers, tokenize_workers) = if cfg.no_tokenize {
        // 无分词模式: 尊重手动指定的read_workers，tokenize_workers 为 0
        let read_w = if let Some(manual_read) = cfg.read_workers {
            manual_read.min(file_count).max(1)
        } else {
            cfg.workers.saturating_sub(write_workers).max(1).min(file_count)
        };
        (read_w, 0)
    } else {
        // 正常模式
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
    // 单文件或外部tokenize worker不多时，开启内部并行；
    // 多文件+大量外部worker时关闭，避免过度订阅
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
    let tok = if cfg.no_tokenize {
        None
    } else {
        Some(Arc::new(Tok::from_path(&cfg.tokenizer)?))
    };

    // 选择 dtype（auto 基于词表规模）
    let dtype = match cfg.dtype {
        DType::U16 => IdxDType::U16,
        DType::I32 => IdxDType::I32,
        DType::Auto => {
            if let Some(ref tokenizer) = tok {
                let vs = tokenizer.vocab_size(true);
                if vs < 65500 { IdxDType::U16 } else { IdxDType::I32 }
            } else {
                IdxDType::U16 // no_tokenize 模式默认使用 U16
            }
        }
    };

    // 初始化指标系统
    let metrics = Arc::new(Metrics::new());
    let (_shutdown_tx, shutdown_rx) = bounded::<()>(1);
    let _metrics_handle = if cfg.metrics_interval > 0 {
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

    // 阶段2：Tokenization worker池 (直接创建，不用Pool)
    let tokenize_handles: Vec<_> = if cfg.no_tokenize {
        // no_tokenize 模式: 启动一个直通worker，仅做数据透传
        let tokenize_tx_clone = tokenize_tx.clone();
        vec![std::thread::spawn(move || -> Result<()> {
            tracing::debug!("Passthrough worker started (no-tokenize mode)");
            while let Ok(read_batch) = read_rx.recv() {
                // 模拟一个空的tokenize结果
                let tokenized_batch = TokenizedBatch {
                    doc_lens: vec![0; read_batch.texts.len()], // 每个文档长度为0
                    token_data: vec![vec![]; read_batch.texts.len()], // 无token数据
                    file_path: read_batch.file_path,
                };
                if tokenize_tx_clone.send(tokenized_batch).is_err() {
                    break; // 下游已关闭
                }
            }
            tracing::debug!("Passthrough worker finished");
            Ok(())
        })]
    } else {
        // 传统的多线程 worker 池
        (0..tokenize_workers).map(|worker_id| {
            let tok = tok.clone().unwrap();
            let metrics = metrics.clone();
            let read_rx = read_rx.clone();
            let tokenize_tx = tokenize_tx.clone();

            std::thread::spawn(move || -> Result<()> {
                tracing::debug!("Tokenization worker {} started", worker_id);

                while let Ok(read_batch) = read_rx.recv() {
                    let input_bytes: usize = read_batch.texts.iter().map(|s| s.len()).sum();
                    metrics.inc_tokenize_input_bytes(input_bytes as u64);

                    let t2 = std::time::Instant::now();
                    let ids_batch = tok.encode_batch_ids(&read_batch.texts, false)?;
                    metrics.add_tokenize_time(t2.elapsed().as_nanos() as u64);

                    let mut doc_lens = Vec::new();
                    let mut token_data = Vec::new();

                    for ids in ids_batch {
                        metrics.inc_tokens(ids.len() as u64);
                        let bytes_written = match dtype {
                            IdxDType::U16 => ids.len() as u64 * 2,
                            IdxDType::I32 => ids.len() as u64 * 4
                        };
                        metrics.inc_output_bytes(bytes_written);

                        doc_lens.push(ids.len() as u32);
                        token_data.push(ids);
                    }

                    if !doc_lens.is_empty() {
                        let tokenized_batch = TokenizedBatch {
                            doc_lens,
                            token_data,
                            file_path: read_batch.file_path,
                        };
                        if tokenize_tx.send(tokenized_batch).is_err() {
                            break; // 下游已关闭
                        }
                    }
                }

                tracing::debug!("Tokenization worker {} finished", worker_id);
                Ok(())
            })
        }).collect()
    };

    // 阶段1：读取和预处理worker池 (直接创建，不用Pool)
    let read_handles: Vec<_> = if file_count == 1 {
        // 单文件优化：直接在主线程读取，避免额外的线程开销
        vec![std::thread::spawn({
            let files = files.clone();
            let cfg = cfg.clone();
            let metrics = metrics.clone();
            let read_tx = read_tx.clone();
            
            move || -> Result<()> {
                tracing::debug!("Single-file read worker started");
                for path in &files {
                    read_single_file(path, &cfg, &metrics, &read_tx)?;
                }
                tracing::debug!("Single-file read worker finished");
                Ok(())
            }
        })]
    } else {
        // 多文件：每个worker处理一部分文件
        (0..read_workers).map(|worker_id| {
            let files = files.clone();
            let cfg = cfg.clone();
            let metrics = metrics.clone();
            let read_tx = read_tx.clone();
            
            std::thread::spawn(move || -> Result<()> {
                tracing::debug!("Read worker {} started", worker_id);
                
                let files_per_worker = (files.len() + read_workers - 1) / read_workers;
                let start_idx = worker_id * files_per_worker;
                let end_idx = ((worker_id + 1) * files_per_worker).min(files.len());
                
                for file_idx in start_idx..end_idx {
                    let path = &files[file_idx];
                    read_single_file(path, &cfg, &metrics, &read_tx)?;
                }
                
                tracing::debug!("Read worker {} finished", worker_id);
                Ok(())
            })
        }).collect()
    };

    // 启动写入工作线程 - 每个worker独享轮转写入器（无锁设计）
    let write_handles: Vec<_> = (0..write_workers).map(|worker_id| {
        let tokenize_rx = tokenize_rx.clone();
        let output_prefix = cfg.output_prefix.clone();
        let no_write = cfg.no_write;
        let metrics = metrics.clone();
        
        std::thread::spawn(move || -> Result<usize> {
            let mut docs_written = 0;
            
            if no_write {
                // no-write 模式：仅消费数据
                tracing::debug!("Write worker {} started (no-write)", worker_id);
                while let Ok(batch) = tokenize_rx.recv() {
                    docs_written += batch.doc_lens.len();
                }
                tracing::debug!("Write worker {} finished (no-write)", worker_id);
                return Ok(docs_written);
            }

            // 每个worker独享自己的轮转写入器 - 完全无锁！
            let mut personal_writer = RotatingWriterPool::new(
                output_prefix,
                1, // 每个pool只管理1个worker
                max_shard_bytes,
                dtype,
                metrics.clone(),
            )?;
            
            tracing::debug!("Write worker {} started with personal rotating writer", worker_id);
            
            while let Ok(batch) = tokenize_rx.recv() {
                // 无锁批量写入！
                for tokens in batch.token_data.iter() {
                    personal_writer.write_document(0, tokens)?; // 在个人pool中worker_id总是0
                }
                docs_written += batch.doc_lens.len();
            }
            
            // 完成时finalize个人写入器
            let _results = personal_writer.finalize_all()?;
            
            tracing::debug!("Write worker {} finished", worker_id);
            Ok(docs_written)
        })
    }).collect();

    // 等待所有读取worker完成
    for handle in read_handles {
        handle.join().unwrap()?;
    }
    drop(read_tx); // 关闭读取->tokenize通道
    
    // 等待所有tokenize worker完成
    for handle in tokenize_handles {
        handle.join().unwrap()?;
    }
    drop(tokenize_tx); // 关闭tokenize->write通道
    
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

    Ok(())
}

fn read_single_file(
    path: &PathBuf,
    cfg: &Config,
    metrics: &Arc<Metrics>,
    read_tx: &crossbeam_channel::Sender<ReadBatch>,
) -> Result<()> {
    metrics.inc_files(1);
    // 预估输入字节：按文件大小累计（粗略近似 I/O）
    if let Ok(meta) = fs::metadata(&path) { 
        metrics.inc_input_bytes(meta.len() as u64); 
    }

    // 优先按列名进行投影，减少反序列化与 I/O
    let t0 = std::time::Instant::now();
    let stream = open_parquet_batches_with_names(&path, Some(&cfg.text_cols), Some(cfg.batch_size))
        .with_context(|| format!("open batches for {:?}", path))?;
    metrics.add_reader_time(t0.elapsed().as_nanos() as u64);

    let schema = &stream.schema;
    // 通过列名解析索引；若找不到则回退到所有 Utf8/LargeUtf8 列
    let mut text_col_indices: Vec<usize> = cfg
        .text_cols
        .iter()
        .filter_map(|name| schema.column_with_name(name).map(|(i, _)| i))
        .collect();
    if text_col_indices.is_empty() {
        for (i, f) in schema.fields().iter().enumerate() {
            let dt = f.data_type();
            let is_text = matches!(dt, arrow_schema::DataType::Utf8 | arrow_schema::DataType::LargeUtf8);
            if is_text { text_col_indices.push(i); }
        }
    }
    if text_col_indices.is_empty() {
        anyhow::bail!("未找到可用的文本列；请通过 --text-cols 指定列名");
    }

    // 显式迭代 reader.next() 来准确计量 Parquet 解码/拉取一批的耗时
    let mut rdr = stream.reader;
    loop {
        let t_fetch = std::time::Instant::now();
        let next = rdr.next();
        let fetch_ns = t_fetch.elapsed().as_nanos() as u64;
        match next {
            Some(batch_res) => {
                metrics.add_reader_time(fetch_ns);
                let batch = batch_res?;
                metrics.inc_batches(1);
                metrics.inc_records(batch.num_rows() as u64);

                // 预处理：提取文本列并拼接
                let t1 = std::time::Instant::now();
                let texts = extract_text_columns(&batch, &text_col_indices, &cfg.concat_sep)?;
                metrics.add_preprocess_time(t1.elapsed().as_nanos() as u64);

                // 发送到tokenization阶段：将大批拆成更小的tokenize批次，提升并行度
                if !texts.is_empty() {
                    let chunk_rows: usize = std::cmp::max(256usize, (cfg.batch_size / 16).max(1));
                    // 零拷贝分片：通过 split_off 按块移动所有权，避免 String 克隆
                    let mut curr = texts; // 接管所有权
                    loop {
                        let next = if curr.len() > chunk_rows {
                            curr.split_off(chunk_rows)
                        } else {
                            Vec::new()
                        };
                        let read_batch = ReadBatch {
                            texts: curr, // move 所有权
                            file_path: path.clone(),
                        };
                        // 有界通道：若下游繁忙，会在此阻塞，形成良性背压
                        if read_tx.send(read_batch).is_err() {
                            tracing::warn!(file = ?path, "downstream closed while sending read batch");
                            return Ok(()); // 下游已关闭，正常退出
                        }
                        if next.is_empty() { break; }
                        curr = next;
                    }
                }
            }
            None => break,
        }
    }

    tracing::debug!(
        file = ?path,
        "File reading completed"
    );
    
    Ok(())
}