use anyhow::Result;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

use crate::config::Config;

pub fn scan_inputs(cfg: &Config) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let pattern = glob::Pattern::new(&cfg.pattern).unwrap_or(glob::Pattern::new("*.parquet").unwrap());
    for entry in WalkDir::new(&cfg.input_dir).into_iter().filter_map(|e| e.ok()) {
        if !entry.file_type().is_file() { continue; }
        let p = entry.path();
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            if pattern.matches(name) {
                files.push(p.to_path_buf());
            }
        }
    }
    files.sort();
    Ok(files)
}


