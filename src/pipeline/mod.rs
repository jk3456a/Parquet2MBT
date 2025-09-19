use anyhow::{Context, Result};
use std::path::PathBuf;
use crossbeam_channel::bounded;

use crate::config::Config;
use crate::reader::open_parquet_batches_with_names;
use crate::preprocessor::extract_text_columns;
use crate::tokenizer::Tok;
use crate::writer::BinWriter;
use crate::index::{write_index, IdxDType};
use crate::metrics::{Metrics, spawn_stdout_reporter};
use std::sync::Arc;
use std::time::Duration;
use std::fs;
use crate::cli::DType;
use std::sync::atomic::Ordering;

pub fn run(cfg: Config, files: Vec<PathBuf>) -> Result<()> {
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
    let tok = Tok::from_path(&cfg.tokenizer)?;

    // 选择 dtype（auto 基于词表规模）
    let dtype = match cfg.dtype {
        DType::U16 => IdxDType::U16,
        DType::I32 => IdxDType::I32,
        DType::Auto => {
            let vs = tok.vocab_size(true);
            if vs < 65500 { IdxDType::U16 } else { IdxDType::I32 }
        }
    };

    // Writer：单线程顺序写入（no_write 时跳过落盘）
    let bin_path = format!("{}.bin", cfg.output_prefix);
    let idx_path = format!("{}.idx", cfg.output_prefix);
    let mut writer = if cfg.no_write { None } else { Some(BinWriter::create(&bin_path, dtype)?) };

    // 遍历文件（Stage1：单机单 pipeline，串行读，内部并行分词）
    for path in files {
        metrics.inc_files(1);
        // 预估输入字节：按文件大小累计（粗略近似 I/O）
        if let Ok(meta) = fs::metadata(&path) { metrics.inc_input_bytes(meta.len() as u64); }

        // 优先按列名进行投影，减少反序列化与 I/O
        let t0 = std::time::Instant::now();
        let stream = open_parquet_batches_with_names(&path, Some(&cfg.text_cols), Some(cfg.batch_size))
            .with_context(|| format!("open batches for {:?}", path))?;
        metrics.add_reader_time(t0.elapsed().as_nanos() as u64);

        let schema = stream.schema.clone();
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

                    // 分词（批量）
                    let t2 = std::time::Instant::now();
                    let ids_batch = tok.encode_batch_ids(&texts, false)?;
                    metrics.add_tokenize_time(t2.elapsed().as_nanos() as u64);

                    // 若 tokenizer 配置有特殊 token，但 encode_batch 返回空，输出一次 DEBUG 辅助定位
                    if ids_batch.iter().all(|v| v.is_empty()) {
                        if let Some(im_start) = tok.token_id("<|im_start|>") {
                            tracing::debug!(im_start_id=?im_start, "tokenizer special token id check");
                        } else {
                            tracing::warn!("special token <|im_start|> not found in tokenizer at runtime");
                        }
                    }

                    // 写入
                    let t3 = std::time::Instant::now();
                    for ids in ids_batch {
                        metrics.inc_tokens(ids.len() as u64);
                        // 估算输出字节数
                        let bytes_written = match dtype { IdxDType::U16 => ids.len() as u64 * 2, IdxDType::I32 => ids.len() as u64 * 4 };
                        metrics.inc_output_bytes(bytes_written);
                        if let Some(w) = writer.as_mut() { w.append_doc(&ids)?; }
                    }
                    metrics.add_write_time(t3.elapsed().as_nanos() as u64);
                }
                None => break,
            }
        }
    }

    if let Some(w) = writer.as_mut() { w.finalize()?; }
    // 根据 dtype 计算 doc_indices（文档级序列边界，对应 builder.document_indices）
    // 在当前实现中，每 doc == 每行文本，sequence_lengths 即每 doc 的 token 数，document_indices 为 [0, 1, 2, ... , N]
    let seq_count = writer.as_ref().map(|w| w.doc_lens.len()).unwrap_or(0);
    let mut doc_indices: Vec<u64> = Vec::with_capacity(seq_count + 1);
    for i in 0..=seq_count { doc_indices.push(i as u64); }
    let t4 = std::time::Instant::now();
    if let Some(w) = writer.as_ref() { write_index(&idx_path, &w.doc_lens, &doc_indices, dtype, None)?; }
    metrics.add_index_time(t4.elapsed().as_nanos() as u64);

    // 优雅停止 metrics reporter
    if let Some(h) = reporter_handle { let _ = shutdown_tx.send(()); let _ = h.join(); }

    // 最终汇总（无论是否命中周期窗口，都打印一次）
    let secs = metrics.uptime_secs().max(1);
    let input = metrics.input_bytes_total.load(Ordering::Relaxed);
    let output = metrics.output_bytes_total.load(Ordering::Relaxed);
    let files = metrics.files_total.load(Ordering::Relaxed);
    let batches = metrics.batches_total.load(Ordering::Relaxed);
    let records = metrics.records_total.load(Ordering::Relaxed);
    let tokens = metrics.tokens_total.load(Ordering::Relaxed);
    let reader_ns = metrics.reader_ns_total.load(Ordering::Relaxed);
    let preprocess_ns = metrics.preprocess_ns_total.load(Ordering::Relaxed);
    let tokenize_ns = metrics.tokenize_ns_total.load(Ordering::Relaxed);
    let write_ns = metrics.write_ns_total.load(Ordering::Relaxed);
    let index_ns = metrics.index_ns_total.load(Ordering::Relaxed);
    let total_ns = reader_ns + preprocess_ns + tokenize_ns + write_ns + index_ns;
    let pct = |x: u64| -> f64 { if total_ns == 0 { 0.0 } else { (x as f64) * 100.0 / (total_ns as f64) } };
    let read_avg = (input as f64) / 1048576.0 / (secs as f64);
    let convert_avg = (output as f64) / 1048576.0 / (secs as f64);
    tracing::info!(
        component = "summary",
        elapsed_secs = secs,
        input_bytes_total = input,
        output_bytes_total = output,
        files_total = files,
        batches_total = batches,
        records_total = records,
        tokens_total = tokens,
        read_avg_mb_per_sec = format!("{:.2}", read_avg).as_str(),
        convert_avg_mb_per_sec = format!("{:.2}", convert_avg).as_str(),
        reader_ms_total = format!("{:.1}", reader_ns as f64 / 1e6).as_str(),
        preprocess_ms_total = format!("{:.1}", preprocess_ns as f64 / 1e6).as_str(),
        tokenize_ms_total = format!("{:.1}", tokenize_ns as f64 / 1e6).as_str(),
        write_ms_total = format!("{:.1}", write_ns as f64 / 1e6).as_str(),
        index_ms_total = format!("{:.1}", index_ns as f64 / 1e6).as_str(),
        reader_pct = format!("{:.1}", pct(reader_ns)).as_str(),
        preprocess_pct = format!("{:.1}", pct(preprocess_ns)).as_str(),
        tokenize_pct = format!("{:.1}", pct(tokenize_ns)).as_str(),
        write_pct = format!("{:.1}", pct(write_ns)).as_str(),
        index_pct = format!("{:.1}", pct(index_ns)).as_str(),
        "run summary"
    );
    Ok(())
}

