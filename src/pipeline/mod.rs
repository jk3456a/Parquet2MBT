use anyhow::{Context, Result};
use std::path::PathBuf;
use crossbeam_channel::{bounded, unbounded};
use rayon::prelude::*;

use crate::config::Config;
use crate::reader::open_parquet_batches_with_names;
use crate::preprocessor::extract_text_columns;
use crate::tokenizer::Tok;
use crate::writer::BinWriter;
use crate::index::{write_index, IdxDType};
use crate::metrics::{Metrics, spawn_stdout_reporter};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::fs;
use crate::cli::DType;
use std::sync::atomic::Ordering;

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

pub fn run(cfg: Config, files: Vec<PathBuf>) -> Result<()> {
    // 智能worker分配：根据文件数量和总worker数动态调整
    let cpu_cores = num_cpus::get();
    let file_count = files.len();
    
    // 阶段1：读取worker数不能超过文件数量
    let read_workers = if file_count == 1 {
        // 单文件：只需要1个读取worker，其他worker用于tokenization
        1
    } else {
        // 多文件：读取worker数 = min(文件数, 20% of total workers)
        (cfg.workers * 20 / 100).max(1).min(file_count)
    };
    
    // 阶段2：CPU密集型 - tokenization（分配大部分worker）
    let tokenize_workers = cfg.workers - read_workers - 1;  // 剩余大部分给tokenization
    
    // 阶段3：写入只需要1个worker（流式写入）
    let write_workers = 1;
    
    // 根据CPU与外部tokenize worker数智能配置tokenizers内部并行，避免过度订阅
    // 单文件或外部tokenize worker不多时，开启内部并行；
    // 多文件+大量外部worker时关闭，避免过度订阅
    let enable_tokenizer_parallel = (file_count == 1)
        || (tokenize_workers <= (cpu_cores / 2).max(1));
    std::env::set_var(
        "TOKENIZERS_PARALLELISM",
        if enable_tokenizer_parallel { "true" } else { "false" }
    );
    
    tracing::info!(
        "Multi-stage pipeline configuration: read_workers={}, tokenize_workers={}, write_workers={}, total_cpu_cores={}", 
        read_workers, tokenize_workers, write_workers, cpu_cores
    );

    // Metrics 初始化与 reporter（若启用）
    let metrics = Arc::new(Metrics::new());
    let (shutdown_tx, shutdown_rx) = bounded::<()>(1);
    let reporter_handle = if cfg.metrics_interval > 0 {
        Some(spawn_stdout_reporter(
            metrics.clone(),
            Duration::from_secs(cfg.metrics_interval),
            shutdown_rx,
        ))
    } else {
        None
    };

    // 初始化 tokenizer
    let tok = Arc::new(Tok::from_path(&cfg.tokenizer)?);

    // 选择 dtype（auto 基于词表规模）
    let dtype = match cfg.dtype {
        DType::U16 => IdxDType::U16,
        DType::I32 => IdxDType::I32,
        DType::Auto => {
            let vs = tok.vocab_size(true);
            if vs < 65500 { IdxDType::U16 } else { IdxDType::I32 }
        }
    };

    // 多阶段流水线通信channels（有界队列以形成背压，稳定内存并耦合阶段速度）
    let qcap = cfg.queue_cap.max(2);
    let (read_tx, read_rx) = bounded::<ReadBatch>(qcap);               // Stage 1 -> Stage 2
    let (tokenize_tx, tokenize_rx) = bounded::<TokenizedBatch>(qcap);  // Stage 2 -> Stage 3
    let total_docs = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    tracing::info!(
        read_workers = read_workers,
        tokenize_workers = tokenize_workers, 
        write_workers = write_workers,
        files_count = files.len(),
        batch_size = cfg.batch_size,
        "Starting multi-stage pipeline processing"
    );

    // 阶段3：写入worker池
    let write_total_docs = total_docs.clone();
    let write_output_prefix = cfg.output_prefix.clone();
    let write_no_write = cfg.no_write;
    let write_handle = std::thread::spawn(move || -> Result<Vec<u32>> {
        let mut all_doc_lens = Vec::new();
        let mut bin_writer = if !write_no_write {
            let bin_path = format!("{}.bin", write_output_prefix);
            Some(std::io::BufWriter::new(
                std::fs::OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(&bin_path)
                    .with_context(|| format!("create bin file: {}", bin_path))?
            ))
        } else {
            None
        };
        
        while let Ok(batch) = tokenize_rx.recv() {
            write_total_docs.fetch_add(batch.doc_lens.len(), Ordering::Relaxed);
            
            // 流式写入：立即写入token数据到磁盘，避免内存累积
            if let Some(ref mut writer) = bin_writer {
                use std::io::Write;
                for tokens in &batch.token_data {
                    // 根据dtype写入token数据
                    match dtype {
                        IdxDType::U16 => {
                            for &token in tokens {
                                if token > u16::MAX as u32 {
                                    return Err(anyhow::anyhow!("token id {} exceeds u16", token));
                                }
                                writer.write_all(&(token as u16).to_le_bytes())?;
                            }
                        }
                        IdxDType::I32 => {
                            for &token in tokens {
                                if token > i32::MAX as u32 {
                                    return Err(anyhow::anyhow!("token id {} exceeds i32", token));
                                }
                                writer.write_all(&(token as i32).to_le_bytes())?;
                            }
                        }
                    }
                }
                // 只有需要写入时才保留doc_lens用于生成索引
                all_doc_lens.extend(batch.doc_lens);
            } else {
                // no-write模式：处理完立即释放，不保存任何数据
                // batch.doc_lens 和 batch.token_data 在这里自动释放
                tracing::debug!("Processed batch with {} docs, immediately released", batch.doc_lens.len());
                // 注意：在no-write模式下，不累积任何数据到all_doc_lens
            }
        }
        
        if let Some(mut writer) = bin_writer {
            use std::io::Write;
            writer.flush().with_context(|| "flush bin file")?;
        }
        
        Ok(all_doc_lens)
    });

    // 阶段2：Tokenization worker池
    let tokenize_handles: Vec<_> = (0..tokenize_workers).map(|worker_id| {
        let tok = tok.clone();
        let metrics = metrics.clone();
        let read_rx = read_rx.clone();
        let tokenize_tx = tokenize_tx.clone();
        let no_write_mode = cfg.no_write;
        
        std::thread::spawn(move || -> Result<()> {
            tracing::debug!("Tokenization worker {} started", worker_id);
            
            while let Ok(read_batch) = read_rx.recv() {
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
                    if !no_write_mode {
                        let tokenized_batch = TokenizedBatch {
                            doc_lens,
                            token_data,
                            file_path: read_batch.file_path,
                        };
                        tokenize_tx.send(tokenized_batch).unwrap();
                    } else {
                        // no-write 模式：不经写入阶段，立刻释放 token 数据
                    }
                }
            }
            
            tracing::debug!("Tokenization worker {} finished", worker_id);
            Ok(())
        })
    }).collect();

    // 阶段1：读取和预处理worker池
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
    
    // 等待写入worker完成
    let all_doc_lens = write_handle.join().unwrap()?;

    // 写入索引文件（.bin文件已在消费者中流式写入）
    if !cfg.no_write {
        let bin_path = format!("{}.bin", cfg.output_prefix);
        let idx_path = format!("{}.idx", cfg.output_prefix);
        
        let seq_count = all_doc_lens.len();
        let mut doc_indices: Vec<u64> = Vec::with_capacity(seq_count + 1);
        for i in 0..=seq_count { 
            doc_indices.push(i as u64); 
        }
        
        let t4 = std::time::Instant::now();
        write_index(&idx_path, &all_doc_lens, &doc_indices, dtype, None)?;
        metrics.add_index_time(t4.elapsed().as_nanos() as u64);
        
        tracing::info!(
            total_docs = seq_count,
            bin_path = bin_path,
            idx_path = idx_path,
            "Files written successfully (bin file streamed, idx file generated)"
        );
    }

    // 优雅停止 metrics reporter
    if let Some(h) = reporter_handle { 
        let _ = shutdown_tx.send(()); 
        let _ = h.join(); 
    }

    print_final_summary(&metrics);
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
                        let mut next = if curr.len() > chunk_rows {
                            curr.split_off(chunk_rows)
                        } else {
                            Vec::new()
                        };
                        let read_batch = ReadBatch {
                            texts: curr, // move 所有权
                            file_path: path.clone(),
                        };
                        // 有界通道：若下游繁忙，会在此阻塞，形成良性背压
                        read_tx.send(read_batch).unwrap();
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


fn print_final_summary(metrics: &Arc<Metrics>) {
    // 最终汇总（无论是否命中周期窗口，都打印一次）
    let elapsed_precise = metrics.elapsed_precise();
    let elapsed_ms = metrics.uptime_millis();
    let input = metrics.input_bytes_total.load(Ordering::Relaxed);
    let output = metrics.output_bytes_total.load(Ordering::Relaxed);
    let files = metrics.files_total.load(Ordering::Relaxed);
    let batches = metrics.batches_total.load(Ordering::Relaxed);
    let records = metrics.records_total.load(Ordering::Relaxed);
    let tokens = metrics.tokens_total.load(Ordering::Relaxed);
    let _reader_ns = metrics.reader_ns_total.load(Ordering::Relaxed);
    let _preprocess_ns = metrics.preprocess_ns_total.load(Ordering::Relaxed);
    let _tokenize_ns = metrics.tokenize_ns_total.load(Ordering::Relaxed);
    let _write_ns = metrics.write_ns_total.load(Ordering::Relaxed);
    let _index_ns = metrics.index_ns_total.load(Ordering::Relaxed);
    
    // 基于精确墙钟时间的实际吞吐量
    let overall_tokens_per_sec = tokens as f64 / elapsed_precise;
    let overall_records_per_sec = records as f64 / elapsed_precise;
    let read_avg = (input as f64) / 1048576.0 / elapsed_precise;
    let convert_avg = (output as f64) / 1048576.0 / elapsed_precise;
    
    tracing::info!(
        component = "summary",
        elapsed_ms = elapsed_ms,
        elapsed_secs = format!("{:.3}", elapsed_precise).as_str(),
        input_bytes_total = input,
        output_bytes_total = output,
        files_total = files,
        batches_total = batches,
        records_total = records,
        tokens_total = tokens,
        // 实际吞吐量（基于精确墙钟时间）
        overall_tokens_per_sec = format!("{:.0}", overall_tokens_per_sec).as_str(),
        overall_records_per_sec = format!("{:.0}", overall_records_per_sec).as_str(),
        read_avg_mb_per_sec = format!("{:.2}", read_avg).as_str(),
        convert_avg_mb_per_sec = format!("{:.2}", convert_avg).as_str(),
        "run summary"
    );
}