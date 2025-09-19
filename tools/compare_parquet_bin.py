import argparse
import json
import struct
from typing import List, Optional, Tuple

import pyarrow.parquet as pq


def read_header_and_counts(idx_path: str) -> Tuple[int, int, int, int, int]:
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
    header_end, version, dtype_code, seq_cnt, doc_cnt = read_header_and_counts(idx_path)
    if doc_idx < 0 or doc_idx >= seq_cnt:
        raise IndexError(f"doc_idx {doc_idx} out of range [0,{seq_cnt})")
    with open(idx_path, "rb") as f:
        # sequence_lengths[doc_idx]
        f.seek(header_end + 4 * doc_idx)
        length = struct.unpack("<i", f.read(4))[0]
        # sequence_pointers[doc_idx]
        pointers_base = header_end + 4 * seq_cnt
        f.seek(pointers_base + 8 * doc_idx)
        pointer = struct.unpack("<q", f.read(8))[0]
    return length, pointer, dtype_code, seq_cnt, doc_cnt


def read_doc_ids(bin_path: str, idx_path: str, doc_idx: int) -> List[int]:
    doc_len, pointer, dtype_code, _, _ = read_doc_meta(idx_path, doc_idx)
    item_size = 4 if dtype_code == 0x04 else 2
    with open(bin_path, "rb") as fb:
        fb.seek(pointer)
        raw = fb.read(item_size * doc_len)
    if item_size == 4:
        ids = list(struct.unpack("<" + "i" * (len(raw) // 4), raw))
    else:
        ids = list(struct.unpack("<" + "H" * (len(raw) // 2), raw))
    return ids


def load_row_text(parquet_path: str, cols: List[str], row_idx: int, sep: str) -> str:
    table = pq.read_table(parquet_path, columns=cols)
    parts: List[str] = []
    for c in cols:
        arr = table.column(c).to_pylist()
        if row_idx >= len(arr):
            s = ""
        else:
            v = arr[row_idx]
            if isinstance(v, bytes):
                try:
                    s = v.decode("utf-8", errors="ignore")
                except Exception:
                    s = ""
            elif v is None:
                s = ""
            else:
                s = str(v)
        parts.append(s)
    return sep.join(parts).strip()


def try_decode_with_tokenizer(ids: List[int], tokenizer_path: Optional[str]) -> Tuple[Optional[List[Optional[str]]], Optional[str], str]:
    if not tokenizer_path:
        return None, None, "none"
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
            vocab = tj.get("model", {}).get("vocab", {})
            inv = {v: k for k, v in vocab.items()}
            tokens = [inv.get(i) for i in ids]
            text = "".join([t for t in tokens if isinstance(t, str)])
            return tokens, text, "vocab_only"
        except Exception:
            return None, None, "none"


def encode_text(text: str, tokenizer_path: Optional[str]) -> Optional[List[int]]:
    if not tokenizer_path:
        return None
    try:
        from tokenizers import Tokenizer  # type: ignore
        tok = Tokenizer.from_file(tokenizer_path)
        enc = tok.encode(text, add_special_tokens=False)
        return enc.ids
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Compare a Parquet row with BIN/IDX doc tokens")
    ap.add_argument("parquet", help="Parquet file path")
    ap.add_argument("bin", help=".bin path")
    ap.add_argument("idx", help=".idx path")
    ap.add_argument("tokenizer", help="tokenizer.json path")
    ap.add_argument("--cols", default="content", help="comma separated column names")
    ap.add_argument("--doc-idx", type=int, default=0, help="document/row index to compare")
    ap.add_argument("--sep", default="\n", help="concat separator for multiple columns")
    ap.add_argument("--preview", type=int, default=200, help="print first N chars of source text")
    ap.add_argument("--head", type=int, default=32, help="print first N tokens for display")
    args = ap.parse_args()

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    src_text = load_row_text(args.parquet, cols, args.doc_idx, args.sep)
    bin_ids = read_doc_ids(args.bin, args.idx, args.doc_idx)
    enc_ids = encode_text(src_text, args.tokenizer)

    # 准备展示
    disp = {
        "parquet_path": args.parquet,
        "bin_path": args.bin,
        "idx_path": args.idx,
        "tokenizer": args.tokenizer,
        "cols": cols,
        "doc_idx": args.doc_idx,
        "src_text_preview": src_text[: args.preview],
        "src_text_len": len(src_text),
        "bin_ids_len": len(bin_ids),
        "enc_ids_len": (len(enc_ids) if enc_ids is not None else None),
        "first_bin_ids": bin_ids[: args.head],
        "first_enc_ids": (enc_ids[: args.head] if enc_ids is not None else None),
    }

    # 尝试解码前 N 个 bin token 片段与文本
    tokens, decoded_text, backend = try_decode_with_tokenizer(bin_ids[: args.head], args.tokenizer)
    disp.update({
        "first_bin_tokens": tokens,
        "decoded_head_text": decoded_text,
        "decoder_backend": backend,
    })

    # 简单一致性判断（长度对齐 + 头部若干个 id 对比）
    if enc_ids is not None:
        same_len = len(enc_ids) == len(bin_ids)
        same_head = enc_ids[: args.head] == bin_ids[: args.head]
        disp.update({"same_len": same_len, "same_head": same_head})

    print(json.dumps(disp, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


