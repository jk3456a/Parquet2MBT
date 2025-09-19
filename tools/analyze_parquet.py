#!/usr/bin/env python3
import sys
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import json

def analyze_parquet(file_path):
    """分析 Parquet 文件的详细信息"""
    try:
        # 读取 Parquet 文件
        parquet_file = pq.ParquetFile(file_path)
        table = parquet_file.read()
        
        print(f"=== Parquet 文件分析: {file_path} ===\n")
        
        # 1. 基本信息
        print("1. 基本信息:")
        print(f"   行数: {table.num_rows:,}")
        print(f"   列数: {table.num_columns}")
        print(f"   文件大小: {parquet_file.metadata.serialized_size:,} bytes")
        print(f"   Row Groups: {parquet_file.num_row_groups}")
        print()
        
        # 2. Schema 信息
        print("2. Schema:")
        schema = parquet_file.schema
        for i, field in enumerate(schema):
            print(f"   列 {i}: {field.name} ({field.physical_type})")
        print()
        
        # 3. 每列的详细统计
        print("3. 列统计信息:")
        df = table.to_pandas()
        
        for col_name in df.columns:
            print(f"   === {col_name} ===")
            col_data = df[col_name]
            
            # 基本统计
            print(f"     数据类型: {col_data.dtype}")
            print(f"     非空值: {col_data.count():,}")
            print(f"     空值: {col_data.isnull().sum():,}")
            print(f"     唯一值: {col_data.nunique():,}")
            
            # 字符串列的特殊统计
            if col_data.dtype == 'object':
                # 长度统计
                lengths = col_data.str.len()
                print(f"     长度统计:")
                print(f"       平均长度: {lengths.mean():.1f}")
                print(f"       最小长度: {lengths.min()}")
                print(f"       最大长度: {lengths.max()}")
                print(f"       中位数长度: {lengths.median():.1f}")
                
                # 显示几个样本
                print(f"     样本数据 (前3个):")
                for i, sample in enumerate(col_data.head(3)):
                    if sample:
                        preview = str(sample)[:100] + "..." if len(str(sample)) > 100 else str(sample)
                        print(f"       [{i}]: {preview}")
                    else:
                        print(f"       [{i}]: <空值>")
            
            # 数值列的统计
            elif pd.api.types.is_numeric_dtype(col_data):
                print(f"     数值统计:")
                print(f"       最小值: {col_data.min()}")
                print(f"       最大值: {col_data.max()}")
                print(f"       平均值: {col_data.mean():.2f}")
                print(f"       中位数: {col_data.median():.2f}")
                print(f"       标准差: {col_data.std():.2f}")
            
            print()
        
        # 4. Row Group 信息
        print("4. Row Group 信息:")
        for i in range(parquet_file.num_row_groups):
            rg = parquet_file.metadata.row_group(i)
            print(f"   Row Group {i}:")
            print(f"     行数: {rg.num_rows:,}")
            print(f"     总字节数: {rg.total_byte_size:,}")
            # 计算每列的压缩信息
            total_compressed = sum(rg.column(j).total_compressed_size for j in range(rg.num_columns))
            print(f"     压缩后字节数: {total_compressed:,}")
            if total_compressed > 0:
                print(f"     压缩比: {rg.total_byte_size/total_compressed:.2f}x")
        print()
        
        # 5. 元数据
        print("5. 元数据:")
        metadata = parquet_file.metadata
        print(f"   创建者: {metadata.created_by}")
        print(f"   版本: {metadata.version}")
        if metadata.metadata:
            print(f"   自定义元数据: {dict(metadata.metadata)}")
        print()
        
        return {
            'num_rows': table.num_rows,
            'num_columns': table.num_columns,
            'schema': [(field.name, str(field.physical_type)) for field in schema],
            'column_stats': {
                col: {
                    'dtype': str(df[col].dtype),
                    'non_null': int(df[col].count()),
                    'null_count': int(df[col].isnull().sum()),
                    'unique_count': int(df[col].nunique()),
                    'avg_length': float(df[col].str.len().mean()) if df[col].dtype == 'object' else None,
                    'max_length': int(df[col].str.len().max()) if df[col].dtype == 'object' else None,
                    'min_length': int(df[col].str.len().min()) if df[col].dtype == 'object' else None,
                    'sample': str(df[col].iloc[0])[:200] if not df[col].empty else None
                } for col in df.columns
            }
        }
        
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <parquet_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    stats = analyze_parquet(file_path)
    
    if stats:
        # 保存统计信息到 JSON
        output_file = file_path.replace('.parquet', '_stats.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"详细统计信息已保存到: {output_file}")

if __name__ == "__main__":
    main()
