# Parquet 工具集

这个目录包含了用于分析和生成 Parquet 文件的工具脚本。

## 工具列表

### 1. `get_parquet_schema.py` - Schema 查看器
查看 Parquet 文件的基本 schema 信息。

```bash
python3 tools/get_parquet_schema.py <parquet_file>
```

**示例：**
```bash
python3 tools/get_parquet_schema.py testdata/data/part-fb89efa5c2f0e50b0e8b475fdbd77bd8.snappy.parquet
```

### 2. `analyze_parquet.py` - 详细分析器
深入分析 Parquet 文件的详细统计信息，包括：
- 基本信息（行数、列数、文件大小）
- Schema 详情
- 每列的统计信息（数据类型、空值、长度分布、样本数据）
- Row Group 信息（压缩比等）
- 元数据信息

```bash
python3 tools/analyze_parquet.py <parquet_file>
```

**示例：**
```bash
python3 tools/analyze_parquet.py testdata/data/part-fb89efa5c2f0e50b0e8b475fdbd77bd8.snappy.parquet
```

**输出：**
- 控制台显示详细分析结果
- 生成 `*_stats.json` 文件保存统计信息

### 3. `generate_parquet.py` - 数据生成器
基于现有 Parquet 文件的统计信息生成相似的测试数据。

```bash
python3 tools/generate_parquet.py [选项]
```

**主要参数：**
- `--output, -o`: 输出文件路径（必需）
- `--rows, -r`: 生成的行数（默认：10000）
- `--template, -t`: 模板文件路径（用于分析统计信息）
- `--compression, -c`: 压缩算法（snappy/gzip/brotli/lz4，默认：snappy）
- `--row-group-size`: Row Group 大小（默认：100）
- `--seed`: 随机种子（用于可重现的结果）

**示例：**

1. **基于模板生成数据：**
```bash
python3 tools/generate_parquet.py \
  --output testdata/test_1k.parquet \
  --rows 1000 \
  --template testdata/data/part-fb89efa5c2f0e50b0e8b475fdbd77bd8.snappy.parquet \
  --seed 42
```

2. **生成大文件：**
```bash
python3 tools/generate_parquet.py \
  --output testdata/large_test.parquet \
  --rows 100000 \
  --compression gzip \
  --row-group-size 1000
```

3. **使用 JSON 统计信息：**
```bash
# 先分析现有文件
python3 tools/analyze_parquet.py existing.parquet
# 使用生成的统计信息
python3 tools/generate_parquet.py \
  --output new_test.parquet \
  --template existing_stats.json \
  --rows 5000
```

## 生成的数据特点

`generate_parquet.py` 生成的数据模拟真实的文档数据集：

### Schema 结构
- `content` (string): 主要文本内容
- `dataset_index` (int64): 数据集索引
- `uid` (string): 唯一标识符 (UUID)
- `meta` (string): 元数据（大部分为空）

### 内容类型
生成器会创建多种类型的文本内容：
1. **学术论文** - 包含标题、作者、摘要、章节等
2. **技术文档** - 技术规范和实现细节
3. **研究摘要** - 简短的研究总结
4. **书籍章节** - 结构化的章节内容
5. **新闻文章** - 新闻报道格式

### 长度分布
- 基于原始数据的统计信息使用对数正态分布
- 支持从很短（0字符）到很长（>1M字符）的文本
- 平均长度约 32K 字符

## 环境要求

确保在 `dataplat` conda 环境中运行：

```bash
conda activate dataplat
pip install faker  # 如果还没安装
```

**依赖包：**
- `pyarrow` - Parquet 文件处理
- `pandas` - 数据处理
- `faker` - 生成测试数据
- `numpy` - 数值计算

## 使用场景

### 1. 性能测试
生成不同大小的测试文件来测试 `parquet2mbt` 的性能：

```bash
# 小文件测试
python3 tools/generate_parquet.py -o testdata/small.parquet -r 1000

# 中等文件测试  
python3 tools/generate_parquet.py -o testdata/medium.parquet -r 50000

# 大文件测试
python3 tools/generate_parquet.py -o testdata/large.parquet -r 500000
```

### 2. 功能测试
生成特定特征的数据来测试边界情况：

```bash
# 可重现的测试数据
python3 tools/generate_parquet.py -o testdata/reproducible.parquet --seed 12345

# 高压缩比数据
python3 tools/generate_parquet.py -o testdata/compressed.parquet -c gzip --row-group-size 10000
```

### 3. 数据分析
分析现有数据的特征：

```bash
# 分析原始数据
python3 tools/analyze_parquet.py original_data.parquet

# 比较生成数据与原始数据
python3 tools/analyze_parquet.py generated_data.parquet
```

## 注意事项

1. **内存使用**: 生成大文件时注意内存使用，建议分批生成
2. **磁盘空间**: 生成的文件可能很大，确保有足够磁盘空间
3. **随机性**: 使用 `--seed` 参数确保结果可重现
4. **压缩**: 不同压缩算法会影响文件大小和读取性能

## 故障排除

### 常见问题

1. **ImportError: No module named 'faker'**
   ```bash
   pip install faker
   ```

2. **内存不足**
   - 减少 `--rows` 参数
   - 增加 `--row-group-size` 来减少内存峰值

3. **文件过大**
   - 使用更强的压缩算法（gzip, brotli）
   - 调整 `--row-group-size` 优化压缩效果

4. **生成速度慢**
   - 减少文本长度（修改模板统计信息）
   - 使用更简单的压缩算法（snappy）
