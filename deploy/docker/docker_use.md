### Parquet2MBT Docker 使用说明（面向用户）

本页仅包含用户需要的最小信息：如何构建镜像、如何运行主工具与常用工具。

---

### 1. 构建镜像（基础）

```bash
docker build -t parquet2mbt:v1.0.0 \
  -f /path/to/Parquet2MBT/deploy/docker/Dockerfile \
  /path/to/Parquet2MBT
```

---

### 2. 运行主工具（最常用）

打印帮助
```bash
docker run --rm parquet2mbt:v1.0.0      # 默认会显示帮助
```

典型转换
```bash
DATA_DIR=/path/to/parquet_dir
OUT_DIR=/path/to/output_dir
TOKENIZER=/path/to/tokenizer.json

docker run --rm \
  -v "$DATA_DIR":/data:ro \
  -v "$OUT_DIR":/out \
  -v "$TOKENIZER":/models/tokenizer.json:ro \
  parquet2mbt:v1.0.0 \
  parquet2mbt \
  --input-dir /data \
  --tokenizer /models/tokenizer.json \
  --output-prefix /out/dataset
```

---

### 3. 常用工具脚本

分析 Parquet 文件
```bash
docker run --rm -v "$DATA_DIR":/data:ro parquet2mbt:v1.0.0 \
  python /app/tools/analyze_parquet.py /data/part_001.parquet
```

读取文件夹中 content 列（预览）
```bash
docker run --rm -v "$DATA_DIR":/data:ro parquet2mbt:v1.0.0 \
  python /app/tools/get_parquet_content.py /data --preview-only
```

对比 Parquet 与 BIN/IDX
```bash
docker run --rm \
  -v "$DATA_DIR":/data:ro \
  -v /path/to/bin_idx:/out:ro \
  -v "$TOKENIZER":/models/tokenizer.json:ro \
  parquet2mbt:v1.0.0 \
  python /app/tools/compare_parquet_bin.py \
  /data/part_001.parquet /out/output.bin /out/output.idx /models/tokenizer.json --doc-idx 0
```

---

### 4. 常见问题（简要）

- 二进制路径：容器内 `parquet2mbt` 已在 PATH 中，亦可显式使用 `/usr/local/bin/parquet2mbt`。



