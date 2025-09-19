use anyhow::{Context, Result};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReader;
use std::fs::File;
use std::path::Path;
use arrow_schema::SchemaRef;
use arrow_array::RecordBatchReader;

pub struct BatchStream {
    pub schema: SchemaRef,
    pub reader: ParquetRecordBatchReader,
}

pub fn open_parquet_batches<P: AsRef<Path>>(path: P, projection: Option<&[usize]>, batch_size: Option<usize>) -> Result<BatchStream> {
    let file = File::open(&path).with_context(|| format!("open parquet: {:?}", path.as_ref()))?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    if let Some(bs) = batch_size { builder = builder.with_batch_size(bs); }
    if let Some(cols) = projection { 
        let mask = ProjectionMask::leaves(builder.metadata().file_metadata().schema_descr(), cols.iter().copied());
        builder = builder.with_projection(mask);
    }
    let schema = builder.schema().clone();
    let reader = builder.build()?;
    Ok(BatchStream { schema, reader })
}

/// 基于列名进行投影（roots 索引）。当列名不存在时将忽略该列；当全部列名均未命中时，不应用投影（读取全部列）。
pub fn open_parquet_batches_with_names<P: AsRef<Path>>(path: P, column_names: Option<&[String]>, batch_size: Option<usize>) -> Result<BatchStream> {
    let file = File::open(&path).with_context(|| format!("open parquet: {:?}", path.as_ref()))?;
    let mut builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    if let Some(bs) = batch_size { builder = builder.with_batch_size(bs); }
    if let Some(names) = column_names {
        // 先获取完整 Arrow schema，从中解析顶层字段索引（roots）
        let full_schema = builder.schema().clone();
        let mut root_indices: Vec<usize> = Vec::new();
        for name in names.iter() {
            if let Some((i, _)) = full_schema.column_with_name(name) { root_indices.push(i); }
        }
        if !root_indices.is_empty() {
            let mask = ProjectionMask::roots(builder.metadata().file_metadata().schema_descr(), root_indices.iter().copied());
            builder = builder.with_projection(mask);
        }
    }
    let reader = builder.build()?;
    // 使用 reader 的 schema（投影后）以保证与批列数一致
    let schema = reader.schema().clone();
    Ok(BatchStream { schema, reader })
}


