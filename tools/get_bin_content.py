import struct, json, sys, os, argparse
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


def find_bin_idx_pairs(folder_path: str) -> List[Tuple[str, str]]:
    """查找文件夹中所有的 .bin 和对应的 .idx 文件对"""
    pairs = []
    
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在", file=sys.stderr)
        return []
    
    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个文件夹", file=sys.stderr)
        return []
    
    # 查找所有 .bin 文件
    bin_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.bin'):
                bin_files.append(os.path.join(root, file))
    
    # 为每个 .bin 文件查找对应的 .idx 文件
    for bin_path in sorted(bin_files):
        idx_path = bin_path.replace('.bin', '.idx')
        if os.path.exists(idx_path):
            pairs.append((bin_path, idx_path))
        else:
            print(f"警告: 找不到对应的索引文件 '{idx_path}'", file=sys.stderr)
    
    return pairs


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def get_preview(text: str, preview_length: int = 50) -> str:
    """获取文本预览"""
    if not text:
        return "<空文本>"
    if len(text) <= preview_length:
        return text
    else:
        return text[:preview_length] + "..."


def process_all_docs_in_file(bin_path: str, idx_path: str, tokenizer_path: Optional[str] = None, 
                           preview_only: bool = False, no_preview: bool = False, 
                           preview_length: int = 50) -> None:
    """处理单个 bin/idx 文件对中的所有文档"""
    try:
        # 读取文件基本信息
        _, version, dtype_code, seq_cnt, doc_cnt = read_header_and_counts(idx_path)
        
        bin_size = os.path.getsize(bin_path)
        idx_size = os.path.getsize(idx_path)
        
        print(f"\n=== 文件对: {os.path.basename(bin_path)} / {os.path.basename(idx_path)} ===", file=sys.stderr)
        print(f"BIN文件大小: {format_size(bin_size)}", file=sys.stderr)
        print(f"IDX文件大小: {format_size(idx_size)}", file=sys.stderr)
        print(f"版本: {version}, 数据类型: {dtype_code}, 序列数: {seq_cnt:,}, 文档数: {doc_cnt:,}", file=sys.stderr)
        
        total_tokens = 0
        total_text_length = 0
        total_text_size = 0
        
        # 处理每个文档
        for doc_idx in range(seq_cnt):
            try:
                # 先获取文档长度，然后读取完整文档
                doc_len, _, _, _, _ = read_doc_meta(idx_path, doc_idx)
                ids, _, _, _, _ = read_doc_ids(bin_path, idx_path, doc_idx=doc_idx, limit=doc_len)
                total_tokens += len(ids)
                
                # 解码文本
                decoded_text = ""
                if tokenizer_path:
                    _, decoded_text, _ = decode_ids(ids, tokenizer_path)
                
                if decoded_text:
                    text_length = len(decoded_text)
                    text_size = len(decoded_text.encode('utf-8'))
                    total_text_length += text_length
                    total_text_size += text_size
                    
                    if preview_only:
                        # 只显示预览
                        preview = get_preview(decoded_text, preview_length)
                        print(f"[文档{doc_idx}] ({len(ids)}tokens, {format_size(text_size)}, {text_length}字符) {preview}")
                    elif no_preview:
                        # 直接输出完整内容
                        print(f"[长度: {len(ids)}tokens, {format_size(text_size)}, {text_length}字符] {decoded_text}")
                    else:
                        # 显示预览信息 + 完整内容
                        preview = get_preview(decoded_text, preview_length)
                        print(f"[文档{doc_idx}] ({len(ids)}tokens, {format_size(text_size)}, {text_length}字符) {preview}", file=sys.stderr)
                        print(decoded_text)
                        print("---", file=sys.stderr)
                else:
                    # 无法解码时只显示token信息
                    if preview_only:
                        print(f"[文档{doc_idx}] ({len(ids)}tokens) <无法解码文本>")
                    elif no_preview:
                        print(f"[长度: {len(ids)}tokens] <无法解码文本>")
                    else:
                        print(f"[文档{doc_idx}] ({len(ids)}tokens) <无法解码文本>", file=sys.stderr)
                        print(f"Token IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")
                        print("---", file=sys.stderr)
                        
            except Exception as e:
                print(f"处理文档 {doc_idx} 时出错: {e}", file=sys.stderr)
                continue
        
        # 输出文件统计信息
        print(f"该文件包含 {seq_cnt:,} 个文档", file=sys.stderr)
        print(f"总token数: {total_tokens:,}", file=sys.stderr)
        if total_text_length > 0:
            print(f"总文本长度: {total_text_length:,} 字符", file=sys.stderr)
            print(f"总文本大小: {format_size(total_text_size)}", file=sys.stderr)
            print(f"平均每文档token数: {total_tokens // seq_cnt if seq_cnt > 0 else 0:,}", file=sys.stderr)
            print(f"平均每文档字符数: {total_text_length // seq_cnt if seq_cnt > 0 else 0:,}", file=sys.stderr)
        
    except Exception as e:
        print(f"处理文件对 '{bin_path}' / '{idx_path}' 时出错: {e}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="读取文件夹中所有 .bin/.idx 文件对的内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python get_bin_content.py /path/to/bin/folder
  python get_bin_content.py /path/to/bin/folder --tokenizer /path/to/tokenizer.json
  python get_bin_content.py /path/to/bin/folder --preview-only
  python get_bin_content.py /path/to/bin/folder --no-preview
        """
    )
    
    parser.add_argument(
        'folder_path',
        help='包含 .bin/.idx 文件的文件夹路径'
    )
    
    parser.add_argument(
        '--tokenizer',
        help='tokenizer.json 文件路径（用于解码文本）'
    )
    
    parser.add_argument(
        '--preview-only',
        action='store_true',
        help='只显示预览信息，不输出完整内容'
    )
    
    parser.add_argument(
        '--no-preview',
        action='store_true',
        help='不显示预览信息，直接输出完整内容'
    )
    
    parser.add_argument(
        '--preview-length',
        type=int,
        default=50,
        help='预览字符数 (默认: 50)'
    )
    
    args = parser.parse_args()
    
    # 查找所有 .bin/.idx 文件对
    file_pairs = find_bin_idx_pairs(args.folder_path)
    
    if not file_pairs:
        print(f"在文件夹 '{args.folder_path}' 中未找到 .bin/.idx 文件对", file=sys.stderr)
        sys.exit(1)
    
    print(f"找到 {len(file_pairs)} 个 .bin/.idx 文件对", file=sys.stderr)
    
    total_files = 0
    total_docs = 0
    total_tokens = 0
    
    # 处理每个文件对
    for file_idx, (bin_path, idx_path) in enumerate(file_pairs):
        try:
            print(f"\n处理第 {file_idx + 1}/{len(file_pairs)} 个文件对...", file=sys.stderr)
            
            # 获取文档数量用于统计
            _, _, _, seq_cnt, _ = read_header_and_counts(idx_path)
            total_docs += seq_cnt
            
            process_all_docs_in_file(
                bin_path, idx_path, 
                tokenizer_path=args.tokenizer,
                preview_only=args.preview_only,
                no_preview=args.no_preview,
                preview_length=args.preview_length
            )
            
            total_files += 1
            
        except Exception as e:
            print(f"处理文件对 '{bin_path}' / '{idx_path}' 时出错: {e}", file=sys.stderr)
            continue
    
    # 输出总结信息
    print(f"\n=== 总结 ===", file=sys.stderr)
    print(f"处理文件对数: {total_files}", file=sys.stderr)
    print(f"总文档数: {total_docs:,}", file=sys.stderr)


if __name__ == "__main__":
    main()