use criterion::{criterion_group, criterion_main, Criterion};
use serde::Deserialize;
use std::fs::{create_dir_all, File};
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::process::{Command, Stdio};

#[derive(Debug, Deserialize)]
struct WindConfig {
    input_dir: String,
    tokenizer: String,
    #[serde(default = "default_pattern")] pattern: String,
    #[serde(default)] workers: Option<usize>,
    #[serde(default)] workers_list: Option<Vec<usize>>,
    #[serde(default = "default_batch")] batch_size: usize,
    #[serde(default)] batch_sizes: Option<Vec<usize>>,
    #[serde(default)] no_write: Option<bool>,
    #[serde(default)] use_rayon_tokenize: Option<bool>,
    #[serde(default)] rayon_num_threads: Option<usize>,
    #[serde(default = "default_bin")] bin: String,
    #[serde(default = "default_out")] out_dir: String,
    #[serde(default)] read_workers: Option<Vec<usize>>,
    #[serde(default)] write_workers: Option<Vec<usize>>,
    #[serde(default)] fixed_tokenize_workers: Option<usize>,
}

fn default_pattern() -> String { "*.parquet".to_string() }
fn default_batch() -> usize { 8192 }
fn default_bin() -> String { "./target/release/parquet2mbt".to_string() }
fn default_out() -> String { "wind_results".to_string() }

fn load_config() -> Option<WindConfig> {
    let path = std::env::var("P2MBT_WIND_TOML").unwrap_or_else(|_| "wind.toml".to_string());
    let mut f = match File::open(&path) {
        Ok(f) => f, Err(_) => { eprintln!("未找到配置: {}，跳过 wind_bench", path); return None; }
    };
    let mut s = String::new();
    if f.read_to_string(&mut s).is_err() { eprintln!("读取配置失败: {}", path); return None; }
    match toml::from_str::<WindConfig>(&s) {
        Ok(mut cfg) => {
            if cfg.workers.is_none() { cfg.workers = Some(num_cpus::get()); }
            Some(cfg)
        }
        Err(e) => { eprintln!("解析TOML失败: {}", e); None }
    }
}

pub fn wind_matrix(_c: &mut Criterion) {
    let Some(cfg) = load_config() else { return; };
    let worker_vec: Vec<usize> = cfg.workers_list.clone().unwrap_or_else(|| vec![cfg.workers.unwrap_or_else(|| num_cpus::get())]);
    let batch_vec: Vec<usize> = cfg.batch_sizes.clone().unwrap_or_else(|| vec![cfg.batch_size]);
    let out_dir = PathBuf::from(&cfg.out_dir);
    let _ = create_dir_all(&out_dir);
    let ts = chrono::Local::now().format("%Y%m%d_%H%M%S").to_string();
    let csv_path = out_dir.join(format!("wind_bench_{}.csv", ts));
    let mut csv = File::create(&csv_path).expect("create csv");
    writeln!(csv, "workers,batch_size,read_workers,tokenize_workers,write_workers,uptime_secs,overall_tokens_per_sec,overall_records_per_sec,overall_read_mb_per_sec,overall_convert_mb_per_sec,interval_tokens_per_sec,interval_records_per_sec,interval_read_mb_per_sec,interval_convert_mb_per_sec,tokens_total,records_total").unwrap();

    // 允许 read/write 缺省（走自动分配）
    let read_list_opt = cfg.read_workers.clone();
    let write_list_opt = cfg.write_workers.clone();
    let read_vals: Vec<Option<usize>> = match read_list_opt { Some(v) => v.into_iter().map(Some).collect(), None => vec![None] };
    let write_vals: Vec<Option<usize>> = match write_list_opt { Some(v) => v.into_iter().map(Some).collect(), None => vec![None] };

    for &bs in &batch_vec {
        for &workers in &worker_vec {
            for r_opt in &read_vals {
                for w_opt in &write_vals {
                    // 计算 tokenize 数
                    let t_val = if let Some(ft) = cfg.fixed_tokenize_workers { Some(ft) } else {
                        match (r_opt, w_opt) {
                            (Some(r), Some(w)) => Some(workers.saturating_sub(*r).saturating_sub(*w)),
                            _ => None, // 自动分配
                        }
                    };
                    if let Some(t) = t_val { if t == 0 { eprintln!("skip r={:?} w={:?} (t=0)", r_opt, w_opt); continue; } }

                    let prefix = out_dir.join(format!("case_w{}_bs{}_r{:?}_t{:?}_w{:?}_{}", workers, bs, r_opt, t_val, w_opt, ts));

                    let mut cmd = Command::new(&cfg.bin);
                    cmd.arg("--input-dir").arg(&cfg.input_dir)
                        .arg("--pattern").arg(&cfg.pattern)
                        .arg("--tokenizer").arg(&cfg.tokenizer)
                        .arg("--output-prefix").arg(prefix.to_string_lossy().to_string())
                        .arg("--batch-size").arg(bs.to_string())
                        .arg("--workers").arg(workers.to_string())
                        .arg("--metrics-interval").arg("10");
                    if let Some(r) = r_opt { cmd.arg("--read-workers").arg(r.to_string()); }
                    if let Some(w) = w_opt { cmd.arg("--write-workers").arg(w.to_string()); }
                    if let Some(t) = t_val { cmd.arg("--tokenize-workers").arg(t.to_string()); }
                    if cfg.no_write.unwrap_or(false) { cmd.arg("--no-write"); }
                    // 程序内部控制 tokenizers 并行；此处不再传递未知 CLI
                    if let Some(rt) = cfg.rayon_num_threads { cmd.env("RAYON_NUM_THREADS", rt.to_string()); }

                    cmd.stdout(Stdio::piped());
                    let mut child = cmd.spawn().expect("spawn parquet2mbt");
                    let stdout = child.stdout.take().unwrap();
                    let reader = std::io::BufReader::new(stdout);
                    let mut last_any: Option<String> = None;       // 任意最后一条
                    let mut last_snapshot: Option<String> = None;  // 最后一条 snapshot（含 interval_*）
                    let mut last_summary: Option<String> = None;   // 最后一条 summary（仅 overall_*）
                    for line in reader.lines() {
                        if let Ok(l) = line {
                            if l.contains("component=\"metrics\"") {
                                last_any = Some(l.clone());
                                if l.contains("summary=true") || l.contains("metrics summary") {
                                    last_summary = Some(l);
                                } else {
                                    last_snapshot = Some(l);
                                }
                            }
                        }
                    }
                    let status = child.wait().expect("wait child");
                    let _ = std::fs::remove_file(format!("{}*", prefix.to_string_lossy()));
                    if !status.success() { eprintln!("run failed"); continue; }
                    let m_overall = last_summary.as_ref().or(last_snapshot.as_ref()).or(last_any.as_ref());
                    let m_interval = last_snapshot.as_ref().or(last_any.as_ref());
                    if let Some(mo) = m_overall {
                        let get_overall = |k: &str| extract_quoted(mo, k);
                        let getn_overall = |k: &str| extract_value(mo, k);
                        let (itps, irps, ir_mb, ic_mb) = if let Some(mi) = m_interval {
                            (
                                extract_quoted(mi, "interval_tokens_per_sec="),
                                extract_quoted(mi, "interval_records_per_sec="),
                                extract_quoted(mi, "interval_read_mb_per_sec="),
                                extract_quoted(mi, "interval_convert_mb_per_sec="),
                            )
                        } else { (String::new(), String::new(), String::new(), String::new()) };

                        writeln!(csv, "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                            workers, bs,
                            r_opt.map(|v| v.to_string()).unwrap_or_else(|| "auto".to_string()),
                            t_val.map(|v| v.to_string()).unwrap_or_else(|| "auto".to_string()),
                            w_opt.map(|v| v.to_string()).unwrap_or_else(|| "auto".to_string()),
                            getn_overall("uptime_secs="), get_overall("overall_tokens_per_sec="), get_overall("overall_records_per_sec="),
                            get_overall("overall_read_mb_per_sec="), get_overall("overall_convert_mb_per_sec="),
                            itps, irps, ir_mb, ic_mb,
                            getn_overall("tokens_total="), getn_overall("records_total=")
                        ).unwrap();
                    }
                }
            }
        }
    }
    eprintln!("wind bench CSV -> {}", csv_path.display());
}

fn extract_value(line: &str, key: &str) -> String {
    if let Some(pos) = line.find(key) {
        let rest = &line[pos + key.len()..];
        let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
        return rest[..end].to_string();
    }
    String::new()
}

fn extract_quoted(line: &str, key: &str) -> String {
    if let Some(pos) = line.find(key) {
        let rest = &line[pos + key.len()..];
        if let Some(start) = rest.find('"') {
            let rest2 = &rest[start + 1..];
            if let Some(end) = rest2.find('"') { return rest2[..end].to_string(); }
        }
    }
    String::new()
}

criterion_group!(benches, wind_matrix);
criterion_main!(benches);


