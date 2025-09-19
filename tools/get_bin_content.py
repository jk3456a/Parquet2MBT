import struct, json, sys, os
from typing import Tuple, List, Optional, Dict, Any


def read_header_and_counts(idx_path: str) -> Tuple[int, int, int, int, int]:
    """读取 idx 头部并返回 (header_end_offset, version, dtype_code, seq_cnt, doc_cnt)。"""
    with open(idx_path, "rb") as f:
        header = f.read(9)
        assert header == b"MMIDIDX\x00\x00", "invalid idx header"
        version = struct.unpack("<Q", f.read(8))[0]
        dtype_code = f.read(1)[0]  # 0x04=i32, 0x08=u16
        seq_cnt = struct.unpack("<Q", f.read(8))[0]
        doc_cnt = struct.unpack("<Q", f.read(8))[0]
        header_end = 9 + 8 + 1 + 8 + 8
        return header_end, version, dtype_code, seq_cnt, doc_cnt


def read_doc_meta(idx_path: str, doc_idx: int) -> Tuple[int, int, int, int, int]:
    """读取指定样本(doc_idx)的 (length, pointer_bytes, dtype_code, seq_cnt, doc_cnt)。"""
    header_end, version, dtype_code, seq_cnt, doc_cnt = read_header_and_counts(idx_path)
    if doc_idx < 0 or doc_idx >= seq_cnt:
        raise IndexError(f"doc_idx {doc_idx} out of range [0,{seq_cnt})")
    with open(idx_path, "rb") as f:
        # 定位到 sequence_lengths[doc_idx]
        f.seek(header_end + 4 * doc_idx)
        length = struct.unpack("<i", f.read(4))[0]
        # 定位到 sequence_pointers[doc_idx]
        pointers_base = header_end + 4 * seq_cnt
        f.seek(pointers_base + 8 * doc_idx)
        pointer = struct.unpack("<q", f.read(8))[0]
    return length, pointer, dtype_code, seq_cnt, doc_cnt


def read_doc_ids(bin_path: str, idx_path: str, doc_idx: int = 0, limit: int = 16) -> Tuple[List[int], int, int, int, int]:
    """读取指定样本前 N 个 token id，返回 (ids, dtype_code, seq_cnt, doc_cnt, doc_len)。"""
    doc_len, pointer, dtype_code, seq_cnt, doc_cnt = read_doc_meta(idx_path, doc_idx)
    item_size = 4 if dtype_code == 0x04 else 2
    to_read = min(doc_len, max(0, limit))
    with open(bin_path, "rb") as fb:
        fb.seek(pointer)
        raw = fb.read(item_size * to_read)
    if item_size == 4:
        ids = list(struct.unpack("<" + "i" * (len(raw) // 4), raw))
    else:
        ids = list(struct.unpack("<" + "H" * (len(raw) // 2), raw))
    return ids, dtype_code, seq_cnt, doc_cnt, doc_len


def decode_ids(ids: List[int], tokenizer_path: str) -> Tuple[Optional[List[Optional[str]]], Optional[str], str]:
    """尝试用 HF tokenizers 解码；失败则回退到词表映射。返回 (tokens, decoded_text, backend)。"""
    try:
        from tokenizers import Tokenizer  # type: ignore
        tok = Tokenizer.from_file(tokenizer_path)
        tokens = [tok.id_to_token(i) for i in ids]
        text = tok.decode(ids, skip_special_tokens=False)
        return tokens, text, "hf_tokenizers"
    except Exception:
        try:
            with open(tokenizer_path, "r") as f:
                tj = json.load(f)
            vocab = tj.get("model", {}).get("vocab", {})  # token -> id
            inv = {v: k for k, v in vocab.items()}          # id -> token
            tokens = [inv.get(i) for i in ids]
            text = "".join([t for t in tokens if isinstance(t, str)])
            return tokens, text, "vocab_only"
        except Exception:
            return None, None, "none"


def run(bin_path: str, idx_path: str, tokenizer_path: Optional[str] = None, doc_idx: int = 0, limit: int = 16) -> Dict[str, Any]:
    ids, dtype_code, seq_cnt, doc_cnt, doc_len = read_doc_ids(bin_path, idx_path, doc_idx=doc_idx, limit=limit)
    tokens: Optional[List[Optional[str]]] = None
    decoded_text: Optional[str] = None
    backend = "none"
    if tokenizer_path:
        tokens, decoded_text, backend = decode_ids(ids, tokenizer_path)
    # 读取版本信息
    _, version, _, _, _ = read_header_and_counts(idx_path)
    return {
        "version": version,
        "dtype_code": dtype_code,
        "seq_cnt": seq_cnt,
        "doc_cnt": doc_cnt,
        "doc_idx": doc_idx,
        "doc_len": doc_len,
        "ids": ids,
        "tokens": tokens,
        "decoded_text": decoded_text,
        "decoder_backend": backend,
    }


def main(argv: List[str]) -> None:
    # 用法：python tools/get_bin_content.py BIN IDX [TOKENIZER_JSON] [DOC_IDX] [LIMIT]
    bin_path = argv[1] if len(argv) > 1 else \
        "/cache/lizhen/repos/DataPlat/Sstable/Parquet2MBT/testdata/output/run_1758105234.bin"
    idx_path = argv[2] if len(argv) > 2 else \
        "/cache/lizhen/repos/DataPlat/Sstable/Parquet2MBT/testdata/output/run_1758105234.idx"
    tokenizer_path = argv[3] if len(argv) > 3 else \
        "/cache/lizhen/repos/DataPlat/Sstable/Parquet2MBT/testdata/tokenizer/tokenizer.json"
    doc_idx = int(argv[4]) if len(argv) > 4 else 0
    limit = int(argv[5]) if len(argv) > 5 else 16
    out = run(bin_path, idx_path, tokenizer_path, doc_idx, limit)
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv)