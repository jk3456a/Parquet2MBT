#!/usr/bin/env python3
"""
生成测试用的 Parquet 文件
基于现有 Parquet 文件的结构和统计信息生成相似的测试数据
"""

import sys
import os
import argparse
import random
import string
import uuid
import json
from typing import List, Dict, Any
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from faker import Faker

# 初始化 Faker
fake = Faker(['en_US', 'zh_CN'])

class ParquetGenerator:
    def __init__(self, template_stats: Dict[str, Any] = None):
        """
        初始化生成器
        
        Args:
            template_stats: 从现有文件分析得到的统计信息
        """
        self.template_stats = template_stats or self._default_stats()
        
    def _default_stats(self) -> Dict[str, Any]:
        """默认的统计信息（基于分析的现有文件）"""
        return {
            'num_rows': 10000,
            'num_columns': 4,
            'schema': [
                ('content', 'BYTE_ARRAY'),
                ('dataset_index', 'INT64'),
                ('uid', 'BYTE_ARRAY'),
                ('meta', 'BYTE_ARRAY')
            ],
            'column_stats': {
                'content': {
                    'dtype': 'object',
                    'non_null': 10000,
                    'null_count': 0,
                    'avg_length': 32021.4,
                    'max_length': 1877260,
                    'min_length': 0
                },
                'dataset_index': {
                    'dtype': 'int64',
                    'non_null': 10000,
                    'null_count': 0,
                    'min_val': 2234901,
                    'max_val': 4546200
                },
                'uid': {
                    'dtype': 'object',
                    'non_null': 10000,
                    'null_count': 0,
                    'avg_length': 36.0,
                    'max_length': 36,
                    'min_length': 36
                },
                'meta': {
                    'dtype': 'object',
                    'non_null': 0,
                    'null_count': 10000
                }
            }
        }
    
    def generate_content_text(self, target_length: int = None) -> str:
        """生成类似学术论文的内容文本"""
        if target_length is None:
            # 基于统计信息随机选择长度
            stats = self.template_stats['column_stats']['content']
            # 使用对数正态分布模拟真实的长度分布
            import numpy as np
            target_length = int(np.random.lognormal(
                mean=np.log(stats['avg_length']), 
                sigma=1.0
            ))
            target_length = max(stats['min_length'], 
                              min(target_length, stats['max_length']))
        
        if target_length == 0:
            return ""
        
        # 生成不同类型的内容
        content_types = [
            self._generate_academic_paper,
            self._generate_technical_document,
            self._generate_research_abstract,
            self._generate_book_chapter,
            self._generate_news_article
        ]
        
        generator = random.choice(content_types)
        content = generator()
        
        # 调整到目标长度
        if len(content) < target_length:
            # 重复内容直到达到目标长度
            repeat_count = (target_length // len(content)) + 1
            content = (content + "\n\n") * repeat_count
        
        return content[:target_length]
    
    def _generate_academic_paper(self) -> str:
        """生成学术论文样式的文本"""
        title = fake.sentence(nb_words=8).replace('.', '')
        author = fake.name()
        
        sections = []
        sections.append(f"# {title}")
        sections.append(f"\n**Author:** {author}")
        sections.append(f"**Institution:** {fake.company()}")
        
        # Abstract
        sections.append("\n## Abstract")
        sections.append(fake.text(max_nb_chars=500))
        
        # Introduction
        sections.append("\n## Introduction")
        for _ in range(random.randint(3, 6)):
            sections.append(fake.text(max_nb_chars=800))
        
        # Methods
        sections.append("\n## Methods")
        for _ in range(random.randint(2, 4)):
            sections.append(fake.text(max_nb_chars=600))
        
        # Results
        sections.append("\n## Results")
        for _ in range(random.randint(2, 5)):
            sections.append(fake.text(max_nb_chars=700))
        
        # Discussion
        sections.append("\n## Discussion")
        for _ in range(random.randint(2, 4)):
            sections.append(fake.text(max_nb_chars=650))
        
        # References
        sections.append("\n## References")
        for i in range(random.randint(10, 30)):
            sections.append(f"[{i+1}] {fake.sentence(nb_words=12)}")
        
        return "\n\n".join(sections)
    
    def _generate_technical_document(self) -> str:
        """生成技术文档样式的文本"""
        title = f"Technical Specification: {fake.bs().title()}"
        sections = [f"# {title}"]
        
        sections.append(f"\n## Overview")
        sections.append(fake.text(max_nb_chars=400))
        
        sections.append(f"\n## Architecture")
        for _ in range(random.randint(2, 4)):
            sections.append(f"### {fake.bs().title()}")
            sections.append(fake.text(max_nb_chars=500))
        
        sections.append(f"\n## Implementation Details")
        for _ in range(random.randint(3, 6)):
            sections.append(fake.text(max_nb_chars=600))
        
        return "\n\n".join(sections)
    
    def _generate_research_abstract(self) -> str:
        """生成研究摘要"""
        title = fake.sentence(nb_words=10).replace('.', '')
        return f"# {title}\n\n{fake.text(max_nb_chars=1500)}"
    
    def _generate_book_chapter(self) -> str:
        """生成书籍章节"""
        chapter_title = fake.sentence(nb_words=6).replace('.', '')
        sections = [f"# Chapter: {chapter_title}"]
        
        for _ in range(random.randint(5, 10)):
            sections.append(f"## {fake.sentence(nb_words=4).replace('.', '')}")
            for _ in range(random.randint(2, 5)):
                sections.append(fake.text(max_nb_chars=800))
        
        return "\n\n".join(sections)
    
    def _generate_news_article(self) -> str:
        """生成新闻文章"""
        headline = fake.sentence(nb_words=8).replace('.', '')
        sections = [f"# {headline}"]
        sections.append(f"\n**{fake.city()}, {fake.date()}** - ")
        
        for _ in range(random.randint(4, 8)):
            sections.append(fake.text(max_nb_chars=600))
        
        return "\n\n".join(sections)
    
    def generate_dataset_index(self) -> int:
        """生成 dataset_index"""
        stats = self.template_stats['column_stats']['dataset_index']
        return random.randint(stats['min_val'], stats['max_val'])
    
    def generate_uid(self) -> str:
        """生成 UUID"""
        return str(uuid.uuid4())
    
    def generate_meta(self) -> str:
        """生成 meta 字段（大部分为空）"""
        stats = self.template_stats['column_stats']['meta']
        null_ratio = stats['null_count'] / (stats['null_count'] + stats['non_null'])
        
        if random.random() < null_ratio:
            return None
        else:
            # 生成简单的 JSON 元数据
            meta = {
                'source': fake.company(),
                'category': random.choice(['research', 'technical', 'academic', 'news']),
                'language': random.choice(['en', 'zh', 'mixed']),
                'quality_score': round(random.uniform(0.5, 1.0), 2)
            }
            return json.dumps(meta)
    
    def generate_dataframe(self, num_rows: int = None) -> pd.DataFrame:
        """生成完整的 DataFrame"""
        if num_rows is None:
            num_rows = self.template_stats['num_rows']
        
        print(f"生成 {num_rows:,} 行数据...")
        
        data = {
            'content': [],
            'dataset_index': [],
            'uid': [],
            'meta': []
        }
        
        for i in range(num_rows):
            if i % 1000 == 0:
                print(f"  进度: {i:,}/{num_rows:,} ({i/num_rows*100:.1f}%)")
            
            data['content'].append(self.generate_content_text())
            data['dataset_index'].append(self.generate_dataset_index())
            data['uid'].append(self.generate_uid())
            data['meta'].append(self.generate_meta())
        
        print("  数据生成完成！")
        return pd.DataFrame(data)
    
    def save_parquet(self, df: pd.DataFrame, output_path: str, 
                    compression: str = 'snappy', 
                    row_group_size: int = 100):
        """保存为 Parquet 文件"""
        print(f"保存到 {output_path}...")
        
        # 转换为 PyArrow Table
        table = pa.Table.from_pandas(df)
        
        # 写入 Parquet 文件
        pq.write_table(
            table, 
            output_path,
            compression=compression,
            row_group_size=row_group_size,
            use_dictionary=True,  # 使用字典编码提高压缩率
            write_statistics=True  # 写入统计信息
        )
        
        print(f"文件已保存: {output_path}")
        
        # 显示文件信息
        file_size = os.path.getsize(output_path)
        print(f"文件大小: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")

def main():
    parser = argparse.ArgumentParser(description='生成测试用的 Parquet 文件')
    parser.add_argument('--output', '-o', required=True, help='输出文件路径')
    parser.add_argument('--rows', '-r', type=int, default=10000, help='生成的行数')
    parser.add_argument('--template', '-t', help='模板文件路径（用于分析统计信息）')
    parser.add_argument('--compression', '-c', default='snappy', 
                       choices=['snappy', 'gzip', 'brotli', 'lz4'],
                       help='压缩算法')
    parser.add_argument('--row-group-size', type=int, default=100,
                       help='Row Group 大小')
    parser.add_argument('--seed', type=int, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed:
        random.seed(args.seed)
        fake.seed_instance(args.seed)
    
    # 加载模板统计信息
    template_stats = None
    if args.template:
        if args.template.endswith('.json'):
            with open(args.template, 'r', encoding='utf-8') as f:
                template_stats = json.load(f)
        else:
            # 如果是 parquet 文件，先分析
            print(f"分析模板文件: {args.template}")
            from analyze_parquet import analyze_parquet
            template_stats = analyze_parquet(args.template)
    
    # 创建生成器
    generator = ParquetGenerator(template_stats)
    
    # 生成数据
    df = generator.generate_dataframe(args.rows)
    
    # 保存文件
    generator.save_parquet(df, args.output, args.compression, args.row_group_size)
    
    print("\n=== 生成完成 ===")
    print(f"输出文件: {args.output}")
    print(f"行数: {len(df):,}")
    print(f"列数: {len(df.columns)}")
    
    # 显示样本数据
    print("\n样本数据:")
    for i, row in df.head(3).iterrows():
        print(f"  行 {i}:")
        print(f"    content: {str(row['content'])[:100]}...")
        print(f"    dataset_index: {row['dataset_index']}")
        print(f"    uid: {row['uid']}")
        print(f"    meta: {row['meta']}")

if __name__ == "__main__":
    main()
