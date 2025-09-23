use anyhow::{anyhow, Context, Result};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::index::IdxDType;

pub mod rotating;
pub use rotating::RotatingWriterPool;

pub struct BinWriter {
    f: BufWriter<File>,
    pub total_tokens: u64,
    pub doc_lens: Vec<u32>,
    dtype: IdxDType,
}

impl BinWriter {
    pub fn create<P: AsRef<Path>>(path: P, dtype: IdxDType) -> Result<Self> {
        let f = OpenOptions::new().create(true).truncate(true).write(true).open(&path).with_context(|| format!("open bin: {:?}", path.as_ref()))?;
        Ok(Self { f: BufWriter::new(f), total_tokens: 0, doc_lens: Vec::new(), dtype })
    }

    pub fn append_doc(&mut self, ids: &[u32]) -> Result<()> {
        match self.dtype {
            IdxDType::U16 => {
                let mut buf: Vec<u16> = Vec::with_capacity(ids.len());
                for &v in ids { if v > u16::MAX as u32 { return Err(anyhow!("token id {} exceeds u16", v)); } buf.push(v as u16); }
                let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len()*2) };
                self.f.write_all(bytes)?;
            }
            IdxDType::I32 => {
                let mut buf: Vec<i32> = Vec::with_capacity(ids.len());
                for &v in ids { if v > i32::MAX as u32 { return Err(anyhow!("token id {} exceeds i32", v)); } buf.push(v as i32); }
                let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len()*4) };
                self.f.write_all(bytes)?;
            }
        }
        self.total_tokens += ids.len() as u64;
        self.doc_lens.push(ids.len() as u32);
        Ok(())
    }

    pub fn finalize(&mut self) -> Result<()> { self.f.flush()?; Ok(()) }
}

// 占位：.idx 写入将在 index 模块完成


