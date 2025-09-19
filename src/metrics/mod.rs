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
        }
    }

    pub fn uptime_secs(&self) -> u64 { self.start.elapsed().as_secs() }

    pub fn inc_input_bytes(&self, v: u64) { self.input_bytes_total.fetch_add(v, Ordering::Relaxed); }
    pub fn inc_output_bytes(&self, v: u64) { self.output_bytes_total.fetch_add(v, Ordering::Relaxed); }
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
        loop {
            crossbeam_channel::select! {
                recv(shutdown_rx) -> _ => { break; }
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

                    let delta_input = input.saturating_sub(prev_input);
                    let delta_output = output.saturating_sub(prev_output);
                    let delta_tokens = tokens.saturating_sub(prev_tokens);
                    let delta_records = records.saturating_sub(prev_records);
                    prev_input = input;
                    prev_output = output;
                    prev_tokens = tokens;
                    prev_records = records;

                    let secs = interval.as_secs().max(1);
                    let in_mbps = (delta_input as f64) / 1048576.0 / (secs as f64);
                    let out_mbps = (delta_output as f64) / 1048576.0 / (secs as f64);
                    let tps = (delta_tokens as f64) / (secs as f64);
                    let rps = (delta_records as f64) / (secs as f64);

                    let total_ns = reader_ns + preprocess_ns + tokenize_ns + write_ns + index_ns;
                    let pct = |x: u64| -> f64 { if total_ns==0 { 0.0 } else { (x as f64) * 100.0 / (total_ns as f64) } };

                    tracing::info!(
                        component = "metrics",
                        uptime_secs = metrics.uptime_secs(),
                        input_bytes_total = input,
                        output_bytes_total = output,
                        files_total = files,
                        batches_total = batches,
                        records_total = records,
                        tokens_total = tokens,
                        errors_total = errors,
                        read_mb_per_sec = format!("{:.2}", in_mbps).as_str(),
                        convert_mb_per_sec = format!("{:.2}", out_mbps).as_str(),
                        tokens_per_sec = format!("{:.0}", tps).as_str(),
                        records_per_sec = format!("{:.0}", rps).as_str(),
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
                        "metrics snapshot"
                    );
                }
            }
        }
    })
}

