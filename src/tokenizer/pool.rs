use anyhow::Result;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use crossbeam_channel::{Sender, Receiver};

use crate::config::Config;
use crate::metrics::Metrics;
use crate::tokenizer::Tok;
use crate::reader::pool::ReadBatch;
use crate::index::IdxDType;

/// 分词后的批次数据
#[derive(Debug)]
pub struct TokenizedBatch {
    pub doc_lens: Vec<u32>,
    pub token_data: Vec<Vec<u32>>,
    pub file_path: PathBuf,
}

/// 分词器池 - 负责文本分词处理
pub struct TokenizerPool {
    handles: Vec<JoinHandle<Result<()>>>,
}

impl TokenizerPool {
    pub fn new(
        cfg: &Config,
        tokenizer: Arc<Tok>,
        dtype: IdxDType,
        metrics: Arc<Metrics>,
        input_rx: Receiver<ReadBatch>,
        output_tx: Sender<TokenizedBatch>,
    ) -> Result<Self> {
        let tokenize_workers = cfg.tokenize_workers.unwrap_or_else(|| {
            cfg.workers.saturating_sub(6).max(1)
        });

        tracing::info!(
            tokenize_workers = tokenize_workers,
            "Starting tokenizer pool"
        );

        let handles = if cfg.no_tokenize {
            // no_tokenize 模式: 启动一个直通worker，仅做数据透传
            vec![Self::spawn_passthrough_worker(input_rx, output_tx)]
        } else {
            // 多线程 worker 池
            Self::spawn_workers(
                tokenize_workers,
                tokenizer,
                dtype,
                metrics,
                input_rx,
                output_tx,
            )
        };

        Ok(Self { handles })
    }

    fn spawn_passthrough_worker(
        input_rx: Receiver<ReadBatch>,
        output_tx: Sender<TokenizedBatch>,
    ) -> JoinHandle<Result<()>> {
        std::thread::spawn(move || -> Result<()> {
            tracing::debug!("Passthrough tokenizer worker started (no-tokenize mode)");
            
            while let Ok(read_batch) = input_rx.recv() {
                // 模拟一个空的tokenize结果
                let tokenized_batch = TokenizedBatch {
                    doc_lens: vec![0; read_batch.texts.len()], // 每个文档长度为0
                    token_data: vec![vec![]; read_batch.texts.len()], // 无token数据
                    file_path: read_batch.file_path,
                };
                
                if output_tx.send(tokenized_batch).is_err() {
                    break; // 下游已关闭
                }
            }
            
            tracing::debug!("Passthrough tokenizer worker finished");
            Ok(())
        })
    }

    fn spawn_workers(
        tokenize_workers: usize,
        tokenizer: Arc<Tok>,
        dtype: IdxDType,
        metrics: Arc<Metrics>,
        input_rx: Receiver<ReadBatch>,
        output_tx: Sender<TokenizedBatch>,
    ) -> Vec<JoinHandle<Result<()>>> {
        (0..tokenize_workers)
            .map(|worker_id| {
                let tok = tokenizer.clone();
                let metrics = metrics.clone();
                let input_rx = input_rx.clone();
                let output_tx = output_tx.clone();

                std::thread::spawn(move || -> Result<()> {
                    tracing::debug!("Tokenization worker {} started", worker_id);

                    while let Ok(read_batch) = input_rx.recv() {
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
                                IdxDType::I32 => ids.len() as u64 * 4,
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
                            
                            if output_tx.send(tokenized_batch).is_err() {
                                break; // 下游已关闭
                            }
                        }
                    }

                    tracing::debug!("Tokenization worker {} finished", worker_id);
                    Ok(())
                })
            })
            .collect()
    }

    pub fn wait_all(self) -> Result<()> {
        for handle in self.handles {
            match handle.join() {
                Ok(res) => res?,
                Err(_) => anyhow::bail!("tokenizer worker panicked"),
            }
        }
        Ok(())
    }
}
