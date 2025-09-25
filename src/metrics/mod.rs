use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use crossbeam_channel::{tick, Receiver};

pub struct Metrics {
    start: Instant,
    pub input_bytes_total: AtomicU64,
    pub output_bytes_total: AtomicU64,
    pub records_total: AtomicU64,
    pub tokens_total: AtomicU64,
    pub files_total: AtomicU64,
    pub batches_total: AtomicU64,
    pub errors_total: AtomicU64,
    // Stage timings (ns)
    pub reader_ns_total: AtomicU64,
    pub preprocess_ns_total: AtomicU64,
    pub tokenize_ns_total: AtomicU64,
    pub write_ns_total: AtomicU64,
    pub index_ns_total: AtomicU64,
    pub tokenize_input_bytes_total: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            input_bytes_total: AtomicU64::new(0),
            output_bytes_total: AtomicU64::new(0),
            records_total: AtomicU64::new(0),
            tokens_total: AtomicU64::new(0),
            files_total: AtomicU64::new(0),
            batches_total: AtomicU64::new(0),
            errors_total: AtomicU64::new(0),
            reader_ns_total: AtomicU64::new(0),
            preprocess_ns_total: AtomicU64::new(0),
            tokenize_ns_total: AtomicU64::new(0),
            write_ns_total: AtomicU64::new(0),
            index_ns_total: AtomicU64::new(0),
            tokenize_input_bytes_total: AtomicU64::new(0),
        }
    }

    pub fn uptime_secs(&self) -> u64 { self.start.elapsed().as_secs() }
    
    pub fn uptime_millis(&self) -> u128 { self.start.elapsed().as_millis() }
    
    pub fn elapsed_precise(&self) -> f64 { self.start.elapsed().as_secs_f64() }

    pub fn inc_input_bytes(&self, v: u64) { self.input_bytes_total.fetch_add(v, Ordering::Relaxed); }
    pub fn inc_output_bytes(&self, v: u64) { self.output_bytes_total.fetch_add(v, Ordering::Relaxed); }
    pub fn inc_tokenize_input_bytes(&self, v: u64) { self.tokenize_input_bytes_total.fetch_add(v, Ordering::Relaxed); }
    pub fn inc_records(&self, v: u64) { self.records_total.fetch_add(v, Ordering::Relaxed); }
    pub fn inc_tokens(&self, v: u64) { self.tokens_total.fetch_add(v, Ordering::Relaxed); }
    pub fn inc_files(&self, v: u64) { self.files_total.fetch_add(v, Ordering::Relaxed); }
    pub fn inc_batches(&self, v: u64) { self.batches_total.fetch_add(v, Ordering::Relaxed); }
    pub fn inc_errors(&self, v: u64) { self.errors_total.fetch_add(v, Ordering::Relaxed); }

    pub fn add_reader_time(&self, ns: u64) { self.reader_ns_total.fetch_add(ns, Ordering::Relaxed); }
    pub fn add_preprocess_time(&self, ns: u64) { self.preprocess_ns_total.fetch_add(ns, Ordering::Relaxed); }
    pub fn add_tokenize_time(&self, ns: u64) { self.tokenize_ns_total.fetch_add(ns, Ordering::Relaxed); }
    pub fn add_write_time(&self, ns: u64) { self.write_ns_total.fetch_add(ns, Ordering::Relaxed); }
    pub fn add_index_time(&self, ns: u64) { self.index_ns_total.fetch_add(ns, Ordering::Relaxed); }
}

pub fn spawn_stdout_reporter(metrics: Arc<Metrics>, interval: Duration, shutdown_rx: Receiver<()>) -> std::thread::JoinHandle<()> {
    let ticker = tick(interval);
    std::thread::spawn(move || {
        let mut prev_input: u64 = 0;
        let mut prev_output: u64 = 0;
        let mut prev_tokens: u64 = 0;
        let mut prev_records: u64 = 0;
        let mut prev_tokenize_input: u64 = 0;
        loop {
            crossbeam_channel::select! {
                recv(shutdown_rx) -> _ => {
                    // 在退出前打印一次最终汇总
                    let input = metrics.input_bytes_total.load(Ordering::Relaxed);
                    let output = metrics.output_bytes_total.load(Ordering::Relaxed);
                    let tokens = metrics.tokens_total.load(Ordering::Relaxed);
                    let records = metrics.records_total.load(Ordering::Relaxed);
                    let files = metrics.files_total.load(Ordering::Relaxed);
                    let batches = metrics.batches_total.load(Ordering::Relaxed);
                    let errors = metrics.errors_total.load(Ordering::Relaxed);
                    let tokenize_input_bytes = metrics.tokenize_input_bytes_total.load(Ordering::Relaxed);

                    let uptime_secs = metrics.uptime_secs();
                    let overall_tokens_per_sec = if uptime_secs > 0 { tokens as f64 / uptime_secs as f64 } else { 0.0 };
                    let overall_records_per_sec = if uptime_secs > 0 { records as f64 / uptime_secs as f64 } else { 0.0 };
                    let overall_read_mbps = if uptime_secs > 0 { (input as f64) / 1048576.0 / (uptime_secs as f64) } else { 0.0 };
                    let overall_convert_mbps = if uptime_secs > 0 { (output as f64) / 1048576.0 / (uptime_secs as f64) } else { 0.0 };
                    let overall_tokenize_input_mbps = if uptime_secs > 0 { (tokenize_input_bytes as f64) / 1048576.0 / (uptime_secs as f64) } else { 0.0 };

                    tracing::info!(
                        component = "metrics",
                        summary = true,
                        uptime_secs = uptime_secs,
                        input_bytes_total = input,
                        output_bytes_total = output,
                        files_total = files,
                        batches_total = batches,
                        records_total = records,
                        tokens_total = tokens,
                        errors_total = errors,
                        overall_tokens_per_sec = format!("{:.0}", overall_tokens_per_sec).as_str(),
                        overall_records_per_sec = format!("{:.0}", overall_records_per_sec).as_str(),
                        overall_read_mb_per_sec = format!("{:.2}", overall_read_mbps).as_str(),
                        overall_convert_mb_per_sec = format!("{:.2}", overall_convert_mbps).as_str(),
                        overall_tokenize_input_mb_per_sec = format!("{:.2}", overall_tokenize_input_mbps).as_str(),
                        "metrics summary"
                    );
                    break;
                }
                recv(ticker) -> _ => {
                    let input = metrics.input_bytes_total.load(Ordering::Relaxed);
                    let output = metrics.output_bytes_total.load(Ordering::Relaxed);
                    let tokens = metrics.tokens_total.load(Ordering::Relaxed);
                    let records = metrics.records_total.load(Ordering::Relaxed);
                    let files = metrics.files_total.load(Ordering::Relaxed);
                    let batches = metrics.batches_total.load(Ordering::Relaxed);
                    let errors = metrics.errors_total.load(Ordering::Relaxed);
                    let reader_ns = metrics.reader_ns_total.load(Ordering::Relaxed);
                    let preprocess_ns = metrics.preprocess_ns_total.load(Ordering::Relaxed);
                    let tokenize_ns = metrics.tokenize_ns_total.load(Ordering::Relaxed);
                    let write_ns = metrics.write_ns_total.load(Ordering::Relaxed);
                    let index_ns = metrics.index_ns_total.load(Ordering::Relaxed);
                    let tokenize_input_bytes = metrics.tokenize_input_bytes_total.load(Ordering::Relaxed);

                    let delta_input = input.saturating_sub(prev_input);
                    let delta_output = output.saturating_sub(prev_output);
                    let delta_tokens = tokens.saturating_sub(prev_tokens);
                    let delta_records = records.saturating_sub(prev_records);
                    let delta_tokenize_input = tokenize_input_bytes.saturating_sub(prev_tokenize_input);
                    prev_input = input;
                    prev_output = output;
                    prev_tokens = tokens;
                    prev_records = records;
                    prev_tokenize_input = tokenize_input_bytes;

                    let secs = interval.as_secs().max(1);
                    let in_mbps = (delta_input as f64) / 1048576.0 / (secs as f64);
                    let out_mbps = (delta_output as f64) / 1048576.0 / (secs as f64);
                    let tps = (delta_tokens as f64) / (secs as f64);
                    let rps = (delta_records as f64) / (secs as f64);
                    let tokenize_input_mbps = (delta_tokenize_input as f64) / 1048576.0 / (secs as f64);

                    // 计算基于墙钟时间的实际吞吐量
                    let uptime_secs = metrics.uptime_secs();
                    let overall_tokens_per_sec = if uptime_secs > 0 { tokens as f64 / uptime_secs as f64 } else { 0.0 };
                    let overall_records_per_sec = if uptime_secs > 0 { records as f64 / uptime_secs as f64 } else { 0.0 };
                    let overall_read_mbps = if uptime_secs > 0 { (input as f64) / 1048576.0 / (uptime_secs as f64) } else { 0.0 };
                    let overall_convert_mbps = if uptime_secs > 0 { (output as f64) / 1048576.0 / (uptime_secs as f64) } else { 0.0 };
                    let overall_tokenize_input_mbps = if uptime_secs > 0 { (tokenize_input_bytes as f64) / 1048576.0 / (uptime_secs as f64) } else { 0.0 };

                    tracing::info!(
                        component = "metrics",
                        uptime_secs = uptime_secs,
                        input_bytes_total = input,
                        output_bytes_total = output,
                        files_total = files,
                        batches_total = batches,
                        records_total = records,
                        tokens_total = tokens,
                        errors_total = errors,
                        // 基于墙钟时间的实际吞吐量
                        overall_tokens_per_sec = format!("{:.0}", overall_tokens_per_sec).as_str(),
                        overall_records_per_sec = format!("{:.0}", overall_records_per_sec).as_str(),
                        overall_read_mb_per_sec = format!("{:.2}", overall_read_mbps).as_str(),
                        overall_convert_mb_per_sec = format!("{:.2}", overall_convert_mbps).as_str(),
                        overall_tokenize_input_mb_per_sec = format!("{:.2}", overall_tokenize_input_mbps).as_str(),
                        // 间隔内的瞬时吞吐量
                        interval_tokens_per_sec = format!("{:.0}", tps).as_str(),
                        interval_records_per_sec = format!("{:.0}", rps).as_str(),
                        interval_read_mb_per_sec = format!("{:.2}", in_mbps).as_str(),
                        interval_convert_mb_per_sec = format!("{:.2}", out_mbps).as_str(),
                        interval_tokenize_input_mb_per_sec = format!("{:.2}", tokenize_input_mbps).as_str(),
                        "metrics snapshot"
                    );
                }
            }
        }
    })
}

