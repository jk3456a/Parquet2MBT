use anyhow::{Context, Result};
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

const INDEX_HEADER: &[u8; 9] = b"MMIDIDX\x00\x00"; // 9 bytes

#[derive(Clone, Copy)]
pub enum IdxDType { U16, I32 }

impl IdxDType {
    pub fn code(self) -> u8 { match self { IdxDType::U16 => 8, IdxDType::I32 => 4 } }
    pub fn item_size(self) -> usize { match self { IdxDType::U16 => 2, IdxDType::I32 => 4 } }
}

pub fn write_index<P: AsRef<Path>>(idx_path: P, seq_lengths: &[u32], doc_indices: &[u64], dtype: IdxDType, seq_modes: Option<&[i8]>) -> Result<()> {
    // 布局按照 Megatron _IndexWriter：
    // header(9) + version(u64=1) + dtype_code(u8)
    // + sequence_count(u64)
    // + document_count(u64)
    // + sequence_lengths(int32)[sequence_count]
    // + sequence_pointers(int64)[sequence_count]
    // + document_indices(int64)[document_count]
    // + sequence_modes(int8)[sequence_count] (可选)

    let mut w = BufWriter::new(OpenOptions::new().create(true).truncate(true).write(true).open(&idx_path).with_context(|| format!("open idx: {:?}", idx_path.as_ref()))?);
    // header + version + dtype code
    w.write_all(INDEX_HEADER)?;
    w.write_all(&1u64.to_le_bytes())?; // version
    w.write_all(&[dtype.code()])?;

    let seq_count = seq_lengths.len() as u64;
    let doc_count = doc_indices.len() as u64;
    w.write_all(&seq_count.to_le_bytes())?;
    w.write_all(&doc_count.to_le_bytes())?;

    // sequence_lengths as int32
    for &len in seq_lengths { let v = len as i32; w.write_all(&v.to_le_bytes())?; }

    // sequence_pointers as int64 (byte offsets in .bin, based on dtype item size)
    let mut ptr: i64 = 0;
    for &len in seq_lengths { w.write_all(&ptr.to_le_bytes())?; ptr += (len as usize * dtype.item_size()) as i64; }

    // document_indices as int64
    for &d in doc_indices { let v = d as i64; w.write_all(&v.to_le_bytes())?; }

    // optional modes
    if let Some(modes) = seq_modes { for &m in modes { w.write_all(&(m as i8).to_le_bytes())?; } }
    Ok(())
}


