use anyhow::{Context, Result};
use std::collections::VecDeque;
use std::io::{BufWriter, Write};
use std::fs::File;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU32, Ordering};
use crossbeam_channel::{bounded, Sender};
use std::thread::JoinHandle;

use crate::index::{write_index, IdxDType};
use crate::metrics::Metrics;

/// 单个分片写入器
struct ShardWriter {
    bin_writer: BufWriter<File>,
    doc_lens: Vec<u32>,
    bytes_written: u64,
    shard_id: u32,
    worker_id: usize,
    output_prefix: String,
}

impl ShardWriter {
    fn new(output_prefix: String, worker_id: usize, shard_id: u32) -> Result<Self> {
        let bin_path = format!("{}.shard_{:02}_{:05}.bin", output_prefix, worker_id, shard_id);
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(&bin_path)
            .with_context(|| format!("create bin file: {}", bin_path))?;
        
        Ok(Self {
            bin_writer: BufWriter::new(file),
            doc_lens: Vec::new(),
            bytes_written: 0,
            shard_id,
            worker_id,
            output_prefix,
        })
    }

    fn write_document(&mut self, tokens: &[u32], dtype: IdxDType) -> Result<()> {
        let doc_len = tokens.len() as u32;
        
        // 批量写入优化：构建整个文档的字节缓冲区
        match dtype {
            IdxDType::U16 => {
                let mut buffer = Vec::with_capacity(tokens.len() * 2);
                for &token in tokens {
                    if token > u16::MAX as u32 {
                        return Err(anyhow::anyhow!("token id {} exceeds u16", token));
                    }
                    buffer.extend_from_slice(&(token as u16).to_le_bytes());
                }
                self.bin_writer.write_all(&buffer)?;
                self.bytes_written += buffer.len() as u64;
            }
            IdxDType::I32 => {
                let mut buffer = Vec::with_capacity(tokens.len() * 4);
                for &token in tokens {
                    if token > i32::MAX as u32 {
                        return Err(anyhow::anyhow!("token id {} exceeds i32", token));
                    }
                    buffer.extend_from_slice(&(token as i32).to_le_bytes());
                }
                self.bin_writer.write_all(&buffer)?;
                self.bytes_written += buffer.len() as u64;
            }
        }
        
        self.doc_lens.push(doc_len);
        Ok(())
    }

    fn should_rotate(&self, max_bytes: u64) -> bool {
        max_bytes > 0 && self.bytes_written >= max_bytes && !self.doc_lens.is_empty()
    }

    fn finalize(mut self, dtype: IdxDType, metrics: &Arc<Metrics>) -> Result<(u32, usize)> {
        // Flush bin file
        self.bin_writer.flush()
            .with_context(|| format!("flush shard {}_{}", self.worker_id, self.shard_id))?;
        
        // Write index file
        if !self.doc_lens.is_empty() {
            let idx_path = format!("{}.shard_{:02}_{:05}.idx", 
                                 self.output_prefix, self.worker_id, self.shard_id);
            
            let seq_count = self.doc_lens.len();
            let doc_indices: Vec<u64> = (0..=seq_count as u64).collect();
            
            let t_idx = std::time::Instant::now();
            write_index(&idx_path, &self.doc_lens, &doc_indices, dtype, None)
                .with_context(|| format!("write index for shard {}_{}", self.worker_id, self.shard_id))?;
            metrics.add_index_time(t_idx.elapsed().as_nanos() as u64);
        }
        
        let doc_count = self.doc_lens.len();
        tracing::info!(
            write_worker = self.worker_id,
            shard_seq = self.shard_id,
            bytes = self.bytes_written,
            docs = doc_count,
            "shard finalized"
        );
        
        Ok((self.shard_id, doc_count))
    }
}

/// 分片完成任务
struct FinalizeTask {
    writer: ShardWriter,
    dtype: IdxDType,
    metrics: Arc<Metrics>,
}

/// 轮转写入器池
pub struct RotatingWriterPool {
    active_writers: Vec<Option<ShardWriter>>,
    standby_queue: Arc<Mutex<VecDeque<ShardWriter>>>,
    finalize_tx: Sender<FinalizeTask>,
    finalize_handles: Vec<JoinHandle<Result<()>>>,
    next_shard_id: AtomicU32,
    output_prefix: String,
    max_shard_bytes: u64,
    dtype: IdxDType,
    metrics: Arc<Metrics>,
    num_workers: usize,
    standby_pool_size: usize,
}

impl RotatingWriterPool {
    pub fn new(
        output_prefix: String,
        num_workers: usize,
        max_shard_bytes: u64,
        dtype: IdxDType,
        metrics: Arc<Metrics>,
    ) -> Result<Self> {
        let standby_pool_size = num_workers * 2; // 2x standby pool
        let (finalize_tx, finalize_rx) = bounded::<FinalizeTask>(standby_pool_size);
        
        // 创建初始活跃 writers
        let mut active_writers = Vec::with_capacity(num_workers);
        let next_shard_id = AtomicU32::new(1);
        
        for worker_id in 0..num_workers {
            let shard_id = next_shard_id.fetch_add(1, Ordering::Relaxed);
            let writer = ShardWriter::new(output_prefix.clone(), worker_id, shard_id)?;
            active_writers.push(Some(writer));
        }
        
        // 创建预备 writers 队列
        let standby_queue = Arc::new(Mutex::new(VecDeque::new()));
        for _ in 0..standby_pool_size {
            let worker_id = 0; // 预备 writer 的 worker_id 在分配时确定
            let shard_id = next_shard_id.fetch_add(1, Ordering::Relaxed);
            let writer = ShardWriter::new(output_prefix.clone(), worker_id, shard_id)?;
            standby_queue.lock().unwrap().push_back(writer);
        }
        
        // 启动后台 finalize 线程池
        let num_finalize_threads = (num_workers / 2).max(1);
        let mut finalize_handles = Vec::new();
        
        for thread_id in 0..num_finalize_threads {
            let finalize_rx = finalize_rx.clone();
            let handle = std::thread::spawn(move || -> Result<()> {
                tracing::debug!("Finalize thread {} started", thread_id);
                while let Ok(task) = finalize_rx.recv() {
                    task.writer.finalize(task.dtype, &task.metrics)?;
                }
                tracing::debug!("Finalize thread {} finished", thread_id);
                Ok(())
            });
            finalize_handles.push(handle);
        }
        
        Ok(Self {
            active_writers,
            standby_queue,
            finalize_tx,
            finalize_handles,
            next_shard_id,
            output_prefix,
            max_shard_bytes,
            dtype,
            metrics,
            num_workers,
            standby_pool_size,
        })
    }

    pub fn write_document(&mut self, worker_id: usize, tokens: &[u32]) -> Result<()> {
        if worker_id >= self.num_workers {
            return Err(anyhow::anyhow!("invalid worker_id: {}", worker_id));
        }

        // 检查是否需要轮转
        let should_rotate = if let Some(writer) = self.active_writers[worker_id].as_ref() {
            writer.should_rotate(self.max_shard_bytes)
        } else {
            false
        };

        if should_rotate {
            self.rotate_writer(worker_id)?;
        }

        // 写入文档
        let writer = self.active_writers[worker_id].as_mut()
            .ok_or_else(|| anyhow::anyhow!("no active writer for worker {}", worker_id))?;
        writer.write_document(tokens, self.dtype)?;
        
        Ok(())
    }

    fn rotate_writer(&mut self, worker_id: usize) -> Result<()> {
        // 取出当前 writer
        let old_writer = self.active_writers[worker_id].take()
            .ok_or_else(|| anyhow::anyhow!("no writer to rotate for worker {}", worker_id))?;

        // 从预备队列获取新 writer
        let mut new_writer = {
            let mut queue = self.standby_queue.lock().unwrap();
            queue.pop_front()
                .ok_or_else(|| anyhow::anyhow!("no standby writer available"))?
        };

        // 更新新 writer 的 worker_id
        new_writer.worker_id = worker_id;

        // 立即替换活跃 writer
        self.active_writers[worker_id] = Some(new_writer);

        // 异步 finalize 旧 writer
        let finalize_task = FinalizeTask {
            writer: old_writer,
            dtype: self.dtype,
            metrics: self.metrics.clone(),
        };
        
        self.finalize_tx.send(finalize_task)
            .map_err(|_| anyhow::anyhow!("failed to send finalize task"))?;

        // 补充预备队列
        self.replenish_standby_pool()?;

        Ok(())
    }

    fn replenish_standby_pool(&mut self) -> Result<()> {
        let mut queue = self.standby_queue.lock().unwrap();
        while queue.len() < self.standby_pool_size {
            let shard_id = self.next_shard_id.fetch_add(1, Ordering::Relaxed);
            let writer = ShardWriter::new(self.output_prefix.clone(), 0, shard_id)?;
            queue.push_back(writer);
        }
        Ok(())
    }

    pub fn finalize_all(self) -> Result<Vec<(u32, usize)>> {
        let mut results = Vec::new();

        // Finalize 所有活跃 writers
        for (_worker_id, writer_opt) in self.active_writers.into_iter().enumerate() {
            if let Some(writer) = writer_opt {
                let result = writer.finalize(self.dtype, &self.metrics)?;
                results.push(result);
            }
        }

        // 关闭 finalize 通道并等待所有任务完成
        drop(self.finalize_tx);
        for handle in self.finalize_handles {
            handle.join().map_err(|_| anyhow::anyhow!("finalize thread panicked"))??;
        }

        Ok(results)
    }
}
