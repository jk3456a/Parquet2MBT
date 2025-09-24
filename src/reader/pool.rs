use anyhow::{Context, Result};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use crossbeam_channel::{Sender, Receiver};

use crate::config::Config;
use crate::metrics::Metrics;
use crate::reader::open_parquet_batches_with_names;
use crate::preprocessor::extract_text_columns;
use std::fs;

/// 读取批次数据
#[derive(Debug)]
pub struct ReadBatch {
    pub texts: Vec<String>,
    pub file_path: PathBuf,
}

/// 读取器池 - 负责文件读取和预处理
pub struct ReaderPool {
    handles: Vec<JoinHandle<Result<()>>>,
}

impl ReaderPool {
    pub fn new(
        files: Vec<PathBuf>,
        cfg: &Config,
        metrics: Arc<Metrics>,
        output_tx: Sender<ReadBatch>,
    ) -> Result<Self> {
        let file_count = files.len();
        let read_workers = cfg.read_workers.unwrap_or_else(|| {
            if file_count == 1 { 1 } else { 4.min(file_count) }
        });

        tracing::info!(
            read_workers = read_workers,
            files_count = file_count,
            "Starting reader pool"
        );

        let handles = if file_count == 1 {
            // 单文件优化：直接在一个线程读取
            vec![Self::spawn_single_file_worker(
                files,
                cfg.clone(),
                metrics,
                output_tx,
            )]
        } else {
            // 多文件：分片处理
            Self::spawn_multi_file_workers(
                files,
                read_workers,
                cfg.clone(),
                metrics,
                output_tx,
            )
        };

        Ok(Self { handles })
    }

    fn spawn_single_file_worker(
        files: Vec<PathBuf>,
        cfg: Config,
        metrics: Arc<Metrics>,
        output_tx: Sender<ReadBatch>,
    ) -> JoinHandle<Result<()>> {
        std::thread::spawn(move || -> Result<()> {
            tracing::debug!("Single-file reader worker started");
            for path in &files {
                Self::read_single_file(path, &cfg, &metrics, &output_tx)?;
            }
            tracing::debug!("Single-file reader worker finished");
            Ok(())
        })
    }

    fn spawn_multi_file_workers(
        files: Vec<PathBuf>,
        read_workers: usize,
        cfg: Config,
        metrics: Arc<Metrics>,
        output_tx: Sender<ReadBatch>,
    ) -> Vec<JoinHandle<Result<()>>> {
        (0..read_workers)
            .map(|worker_id| {
                let files = files.clone();
                let cfg = cfg.clone();
                let metrics = metrics.clone();
                let output_tx = output_tx.clone();

                std::thread::spawn(move || -> Result<()> {
                    tracing::debug!("Reader worker {} started", worker_id);

                    let files_per_worker = (files.len() + read_workers - 1) / read_workers;
                    let start_idx = worker_id * files_per_worker;
                    let end_idx = ((worker_id + 1) * files_per_worker).min(files.len());

                    for file_idx in start_idx..end_idx {
                        let path = &files[file_idx];
                        Self::read_single_file(path, &cfg, &metrics, &output_tx)?;
                    }

                    tracing::debug!("Reader worker {} finished", worker_id);
                    Ok(())
                })
            })
            .collect()
    }

    fn read_single_file(
        path: &PathBuf,
        cfg: &Config,
        metrics: &Arc<Metrics>,
        output_tx: &Sender<ReadBatch>,
    ) -> Result<()> {
        metrics.inc_files(1);
        
        // 预估输入字节：按文件大小累计
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
                if is_text {
                    text_col_indices.push(i);
                }
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
                        let chunk_rows: usize = cfg.batch_size.max(1);
                        
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
                            if output_tx.send(read_batch).is_err() {
                                tracing::warn!(file = ?path, "downstream closed while sending read batch");
                                return Ok(()); // 下游已关闭，正常退出
                            }
                            
                            if next.is_empty() {
                                break;
                            }
                            curr = next;
                        }
                    }
                }
                None => break,
            }
        }

        tracing::debug!(file = ?path, "File reading completed");
        Ok(())
    }

    pub fn wait_all(self) -> Result<()> {
        for handle in self.handles {
            match handle.join() {
                Ok(res) => res?,
                Err(_) => anyhow::bail!("reader worker panicked"),
            }
        }
        Ok(())
    }
}
