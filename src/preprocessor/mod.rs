use anyhow::Result;
use arrow_array::{Array, LargeStringArray, StringArray, LargeBinaryArray, BinaryArray, RecordBatch};

pub fn extract_text_columns(batch: &RecordBatch, col_indices: &[usize], sep: &str) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let mut parts: Vec<&str> = Vec::with_capacity(col_indices.len());
        for &ci in col_indices {
            let col = batch.column(ci);
            if let Some(sa) = col.as_any().downcast_ref::<StringArray>() {
                if sa.is_valid(row) { parts.push(sa.value(row)); }
            } else if let Some(la) = col.as_any().downcast_ref::<LargeStringArray>() {
                if la.is_valid(row) { parts.push(la.value(row)); }
            } else if let Some(ba) = col.as_any().downcast_ref::<BinaryArray>() {
                if ba.is_valid(row) {
                    let bytes = ba.value(row);
                    if let Ok(s) = std::str::from_utf8(bytes) { parts.push(s); }
                }
            } else if let Some(lba) = col.as_any().downcast_ref::<LargeBinaryArray>() {
                if lba.is_valid(row) {
                    let bytes = lba.value(row);
                    if let Ok(s) = std::str::from_utf8(bytes) { parts.push(s); }
                }
            }
        }
        out.push(parts.join(sep).trim().to_string());
    }
    Ok(out)
}


