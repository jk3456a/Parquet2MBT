#!/usr/bin/env python3
"""
读取文件夹中所有 Parquet 文件的 content 列内容
- 输入：文件夹路径
- 输出：所有 content 列的内容到标准输出
- 预览：显示前50个字符和尺寸信息
"""
import sys
import os
import argparse
from typing import List, Generator, Tuple
import pyarrow.parquet as pq
import pyarrow as pa


def find_parquet_files(folder_path: str) -> List[str]:
    """递归查找文件夹中的所有 Parquet 文件"""
    parquet_files = []
    
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在", file=sys.stderr)
        return []
    
    if not os.path.isdir(folder_path):
        print(f"错误: '{folder_path}' 不是一个文件夹", file=sys.stderr)
        return []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    return sorted(parquet_files)


def get_content_from_parquet(file_path: str, column_name: str = 'content') -> Generator[Tuple[int, str], None, None]:
    """从 Parquet 文件中读取指定列的内容"""
    try:
        # 读取 Parquet 文件
        table = pq.read_table(file_path, columns=[column_name])
        
        # 转换为 Python 列表
        content_array = table.column(0).to_pylist()
        
        # 逐行返回内容
        for row_idx, content in enumerate(content_array):
            if content is not None:
                # 处理不同类型的内容
                if isinstance(content, (bytes, bytearray)):
                    try:
                        content_str = content.decode('utf-8', errors='ignore')
                    except Exception:
                        content_str = str(content)
                else:
                    content_str = str(content)
                
                if content_str.strip():  # 只返回非空内容
                    yield row_idx, content_str
                    
    except Exception as e:
        print(f"错误: 无法读取文件 '{file_path}': {e}", file=sys.stderr)


def format_size(size_bytes: int) -> str:
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f}{unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f}TB"


def get_preview(content: str, preview_length: int = 50) -> str:
    """获取内容预览"""
    if len(content) <= preview_length:
        return content
    else:
        return content[:preview_length] + "..."


def main():
    parser = argparse.ArgumentParser(
        description="读取文件夹中所有 Parquet 文件的 content 列内容",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python get_parquet_content.py /path/to/parquet/folder
  python get_parquet_content.py /path/to/parquet/folder --column text
  python get_parquet_content.py /path/to/parquet/folder --preview-only
  python get_parquet_content.py /path/to/parquet/folder --no-preview
        """
    )
    
    parser.add_argument(
        'folder_path',
        help='包含 Parquet 文件的文件夹路径'
    )
    
    parser.add_argument(
        '--column',
        default='content',
        help='要读取的列名 (默认: content)'
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
    
    # 查找所有 Parquet 文件
    parquet_files = find_parquet_files(args.folder_path)
    
    if not parquet_files:
        print(f"在文件夹 '{args.folder_path}' 中未找到 Parquet 文件", file=sys.stderr)
        sys.exit(1)
    
    print(f"找到 {len(parquet_files)} 个 Parquet 文件", file=sys.stderr)
    
    total_rows = 0
    total_size = 0
    total_content_size = 0
    total_content_length = 0
    
    # 处理每个文件
    for file_idx, file_path in enumerate(parquet_files):
        try:
            # 获取文件大小
            file_size = os.path.getsize(file_path)
            total_size += file_size
            
            print(f"\n=== 文件 {file_idx + 1}/{len(parquet_files)}: {os.path.basename(file_path)} ===", file=sys.stderr)
            print(f"文件大小: {format_size(file_size)}", file=sys.stderr)
            
            file_rows = 0
            
            # 读取内容
            for row_idx, content in get_content_from_parquet(file_path, args.column):
                file_rows += 1
                total_rows += 1
                
                # 计算内容大小和长度
                content_size = len(content.encode('utf-8'))
                content_length = len(content)
                total_content_size += content_size
                total_content_length += content_length
                
                if args.preview_only:
                    # 只显示预览
                    preview = get_preview(content, args.preview_length)
                    print(f"[文件{file_idx+1}:行{row_idx}] ({format_size(content_size)}, {content_length}字符) {preview}")
                elif args.no_preview:
                    # 直接输出完整内容
                    print(f"[长度: {format_size(content_size)}, {content_length}字符] {content}")
                else:
                    # 显示预览信息 + 完整内容
                    preview = get_preview(content, args.preview_length)
                    print(f"[文件{file_idx+1}:行{row_idx}] ({format_size(content_size)}, {content_length}字符) {preview}", file=sys.stderr)
                    print(content)
                    print("---", file=sys.stderr)
            
            print(f"该文件包含 {file_rows} 行有效内容", file=sys.stderr)
            
        except Exception as e:
            print(f"处理文件 '{file_path}' 时出错: {e}", file=sys.stderr)
            continue
    
    # 输出总结信息
    print(f"\n=== 总结 ===", file=sys.stderr)
    print(f"处理文件数: {len(parquet_files)}", file=sys.stderr)
    print(f"总行数: {total_rows:,}", file=sys.stderr)
    print(f"文件总大小: {format_size(total_size)}", file=sys.stderr)
    print(f"内容总大小: {format_size(total_content_size)}", file=sys.stderr)
    print(f"内容总字符数: {total_content_length:,}", file=sys.stderr)
    if total_rows > 0:
        avg_content_size = total_content_size / total_rows
        avg_content_length = total_content_length / total_rows
        print(f"平均每行大小: {format_size(int(avg_content_size))}", file=sys.stderr)
        print(f"平均每行字符数: {int(avg_content_length):,}", file=sys.stderr)


if __name__ == "__main__":
    main()
