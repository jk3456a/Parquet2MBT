use anyhow::Result;
use arrow_array::{Array, LargeStringArray, StringArray, LargeBinaryArray, BinaryArray, RecordBatch};
use arrow_array::{GenericStringArray, LargeListArray, ListArray, StructArray};
use arrow_schema::DataType;
use std::borrow::Cow;
use std::sync::OnceLock;

/// 将连续的换行（≥3个）压缩为两个换行，近似等价于
/// Python: re.sub(r"([ \t]*\n){3,}", "\n\n", text)
fn collapse_excess_newlines(input: &str) -> String {
    let bytes = input.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b == b'\n' {
            // 统计连续的 \n 数量（与前导空白不完全一致，但可近似）
            let mut cnt = 1usize;
            let mut j = i + 1;
            while j < bytes.len() && bytes[j] == b'\n' { cnt += 1; j += 1; }
            // 保留至多两个换行
            let keep = cnt.min(2);
            for _ in 0..keep { out.push(b'\n'); }
            i = j;
            continue;
        }
        out.push(b);
        i += 1;
    }
    String::from_utf8(out).unwrap_or_else(|_| input.to_string())
}

#[inline]
fn maybe_collapse_excess_newlines(input: &str) -> Cow<'_, str> {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    let enabled = *ENABLED.get_or_init(|| {
        match std::env::var("P2MBT_COLLAPSE_NEWLINES") {
            Ok(v) => {
                let v = v.to_ascii_lowercase();
                !(v == "0" || v == "false" || v == "off")
            }
            Err(_) => true, // 默认开启
        }
    });
    if enabled && input.as_bytes().windows(3).any(|w| w == b"\n\n\n") {
        Cow::Owned(collapse_excess_newlines(input))
    } else {
        Cow::Borrowed(input)
    }
}

pub fn extract_text_columns(batch: &RecordBatch, col_indices: &[usize], sep: &str) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let mut joined: Option<String> = None;
        let mut first = true;
        for &ci in col_indices {
            let col = batch.column(ci);
            let piece: Option<Cow<'_, str>> = if let Some(sa) = col.as_any().downcast_ref::<StringArray>() {
                if sa.is_valid(row) { Some(Cow::Borrowed(sa.value(row))) } else { None }
            } else if let Some(la) = col.as_any().downcast_ref::<LargeStringArray>() {
                if la.is_valid(row) { Some(Cow::Borrowed(la.value(row))) } else { None }
            } else if let Some(ba) = col.as_any().downcast_ref::<BinaryArray>() {
                if ba.is_valid(row) { Some(Cow::Owned(String::from_utf8_lossy(ba.value(row)).into_owned())) } else { None }
            } else if let Some(lba) = col.as_any().downcast_ref::<LargeBinaryArray>() {
                if lba.is_valid(row) { Some(Cow::Owned(String::from_utf8_lossy(lba.value(row)).into_owned())) } else { None }
            } else { None };

            if let Some(p) = piece {
                let j = joined.get_or_insert_with(String::new);
                if !first { j.push_str(sep); } else { first = false; }
                j.push_str(&p);
            }
        }
        if let Some(j) = joined {
            let norm = maybe_collapse_excess_newlines(&j);
            if !norm.is_empty() { out.push(norm.into_owned()); }
        }
    }
    Ok(out)
}

/// 解析 plain(content) 或 chatml(messages) 并渲染为字符串；若两者都有，优先 messages。
/// messages 期望为 List<Struct{role: Utf8/LargeUtf8, content: Utf8/LargeUtf8}>
pub fn process_plain_or_chatml(batch: &RecordBatch, content_idx: Option<usize>, messages_idx: Option<usize>, sep: &str) -> Result<Vec<String>> {
    if let Some(mi) = messages_idx {
        let col = batch.column(mi);
        // 支持 List 或 LargeList
        let render_row = |row: usize| -> Option<String> {
            match col.data_type() {
                DataType::List(_) => {
                    let la = col.as_any().downcast_ref::<ListArray>()?;
                    if !la.is_valid(row) { return None; }
                    let offs = la.value_offsets();
                    let start = offs[row] as usize; let end = offs[row+1] as usize;
                    let offsets = (start, end - start);
                    let values = la.values();
                    let sa = values.as_any().downcast_ref::<StructArray>()?;
                    Some(render_messages_struct(sa, offsets))
                }
                DataType::LargeList(_) => {
                    let la = col.as_any().downcast_ref::<LargeListArray>()?;
                    if !la.is_valid(row) { return None; }
                    let offs = la.value_offsets();
                    let start = offs[row] as usize; let end = offs[row+1] as usize;
                    let offsets = (start, end - start);
                    let values = la.values();
                    let sa = values.as_any().downcast_ref::<StructArray>()?;
                    Some(render_messages_struct(sa, offsets))
                }
                _ => None,
            }
        };
        let mut out = Vec::with_capacity(batch.num_rows());
        for r in 0..batch.num_rows() { if let Some(s) = render_row(r) { out.push(s); } }
        return Ok(out);
    }
    if let Some(ci) = content_idx {
        return extract_text_columns(batch, &[ci], sep);
    }
    Ok(Vec::new())
}

fn render_messages_struct(sa: &StructArray, offsets: (usize, usize)) -> String {
    // 期望字段名包含 role/content（大小写不敏感）
    let mut role_idx = None;
    let mut content_idx = None;
    for (i, f) in sa.fields().iter().enumerate() {
        let n = f.name().to_ascii_lowercase();
        if n == "role" { role_idx = Some(i); }
        if n == "content" { content_idx = Some(i); }
    }
    let mut parts: Vec<String> = Vec::new();
    let (start, len) = offsets;
    let end = start + len;
    let role_arr = role_idx.and_then(|i| sa.column(i).as_any().downcast_ref::<GenericStringArray<i32>>());
    let role_arr_l = role_idx.and_then(|i| sa.column(i).as_any().downcast_ref::<GenericStringArray<i64>>());
    let content_arr = content_idx.and_then(|i| sa.column(i).as_any().downcast_ref::<GenericStringArray<i32>>());
    let content_arr_l = content_idx.and_then(|i| sa.column(i).as_any().downcast_ref::<GenericStringArray<i64>>());
    for i in start..end {
        let role = if let Some(a) = role_arr { if a.is_valid(i) { a.value(i).to_string() } else { String::new() } }
                   else if let Some(a) = role_arr_l { if a.is_valid(i) { a.value(i).to_string() } else { String::new() } }
                   else { String::new() };
        let mut content = if let Some(a) = content_arr { if a.is_valid(i) { a.value(i).to_string() } else { String::new() } }
                          else if let Some(a) = content_arr_l { if a.is_valid(i) { a.value(i).to_string() } else { String::new() } }
                          else { String::new() };
        // 严格换行规整
        let norm = maybe_collapse_excess_newlines(&content);
        parts.push(format!("<|im_start|>{}\n{}<|im_end|>\n", role, norm));
    }
    parts.join("")
}


