#!/usr/bin/env python3
import sys
import pyarrow.parquet as pq

def main():
    # 检查命令行参数
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <parquet_file>")
        sys.exit(1)

    file_path = sys.argv[1]

    # 读取 Parquet 文件
    try:
        parquet_file = pq.ParquetFile(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        sys.exit(1)

    # 获取 schema
    schema = parquet_file.schema

    # 打印 schema
    print(schema)

if __name__ == "__main__":
    main()
