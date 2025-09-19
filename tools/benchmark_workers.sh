#!/bin/bash

# 风洞实验脚本：同时扫描 workers 与 batch-size 对吞吐的影响
set -euo pipefail

# 默认配置（可通过环境变量或命令行覆盖）
INPUT_DIR="${INPUT_DIR:-/cache/lizhen/zh__CCI4.0-M2-Base-v1-newest_zh_cc-high-loss0__2025091500}"
PATTERN="${PATTERN:-*.parquet}"
TOKENIZER="${TOKENIZER:-/cache/lizhen/repos/DataPlat/Sstable/Parquet2MBT/testdata/tokenizer/tokenizer.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/cache/lizhen/repos/DataPlat/Sstable/Parquet2MBT/testdata/benchmark}"
BINARY="${BINARY:-./target/release/parquet2mbt}"
NO_WRITE="${NO_WRITE:-true}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
REPEAT="${REPEAT:-1}"

# 扫描维度
WORKER_COUNTS=(${WORKER_COUNTS:-16 32 48 64 80 96})
BATCH_SIZES=(${BATCH_SIZES:-8192 16384 32768 65536})

mkdir -p "$OUTPUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
RESULTS_CSV="$OUTPUT_DIR/windtunnel_w_bs_${TS}.csv"
RESULTS_LOG="$OUTPUT_DIR/windtunnel_w_bs_${TS}.log"

# CSV 表头
echo "workers,batch_size,elapsed_secs,overall_tokens_per_sec,overall_records_per_sec,read_avg_mb_per_sec,convert_avg_mb_per_sec,total_tokens,total_records,read_workers,tokenize_workers,write_workers" > "$RESULTS_CSV"

echo "=== 风洞实验：workers x batch_size ===" | tee "$RESULTS_LOG"
echo "开始时间: $(date)" | tee -a "$RESULTS_LOG"
echo "INPUT_DIR=$INPUT_DIR" | tee -a "$RESULTS_LOG"
echo "TOKENIZER=$TOKENIZER" | tee -a "$RESULTS_LOG"
echo "NO_WRITE=$NO_WRITE TIMEOUT_SEC=$TIMEOUT_SEC REPEAT=$REPEAT" | tee -a "$RESULTS_LOG"
echo "WORKERS: ${WORKER_COUNTS[*]}" | tee -a "$RESULTS_LOG"
echo "BATCH_SIZES: ${BATCH_SIZES[*]}" | tee -a "$RESULTS_LOG"
echo "" | tee -a "$RESULTS_LOG"

for bs in "${BATCH_SIZES[@]}"; do
  for workers in "${WORKER_COUNTS[@]}"; do
    for r in $(seq 1 "$REPEAT"); do
      echo "=== 测试 batch_size=$bs workers=$workers (run#$r) ===" | tee -a "$RESULTS_LOG"
      output_prefix="$OUTPUT_DIR/wt_bs${bs}_w${workers}_r${r}_$TS"
      start_time=$(date +%s)

      cmd=("$BINARY" \
        --input-dir "$INPUT_DIR" \
        --pattern "$PATTERN" \
        --text-cols content \
        --tokenizer "$TOKENIZER" \
        --output-prefix "$output_prefix" \
        --batch-size "$bs" \
        --workers "$workers" \
        --metrics-interval 0)
      if [ "$NO_WRITE" = "true" ]; then
        cmd+=(--no-write)
      fi

      log_file="$OUTPUT_DIR/run_bs${bs}_w${workers}_r${r}.log"
      if ! timeout "$TIMEOUT_SEC" "${cmd[@]}" 2>&1 | tee "$log_file"; then
        echo "  运行超时/失败: bs=$bs workers=$workers run#$r" | tee -a "$RESULTS_LOG"
      fi

      end_time=$(date +%s)
      wall_time=$((end_time - start_time))

      # 解析配置行（读取/分词/写入 worker 配比）
      cfg_line=$(grep "Multi-stage pipeline configuration" "$log_file" | head -1 || true)
      read_workers=$(echo "$cfg_line" | grep -o 'read_workers=[0-9]*' | cut -d'=' -f2)
      tokenize_workers=$(echo "$cfg_line" | grep -o 'tokenize_workers=[0-9]*' | cut -d'=' -f2)
      write_workers=$(echo "$cfg_line" | grep -o 'write_workers=[0-9]*' | cut -d'=' -f2)

      # 解析 summary 行
      summary_line=$(grep "run summary" "$log_file" | tail -1 || true)
      if [ -n "$summary_line" ]; then
        elapsed_secs=$(echo "$summary_line" | grep -o 'elapsed_secs="[0-9.]*"' | cut -d'"' -f2)
        overall_tokens_per_sec=$(echo "$summary_line" | grep -o 'overall_tokens_per_sec="[0-9.]*"' | cut -d'"' -f2)
        overall_records_per_sec=$(echo "$summary_line" | grep -o 'overall_records_per_sec="[0-9.]*"' | cut -d'"' -f2)
        read_avg_mb_per_sec=$(echo "$summary_line" | grep -o 'read_avg_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
        convert_avg_mb_per_sec=$(echo "$summary_line" | grep -o 'convert_avg_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
        total_tokens=$(echo "$summary_line" | grep -o 'tokens_total=[0-9]*' | cut -d'=' -f2)
        total_records=$(echo "$summary_line" | grep -o 'records_total=[0-9]*' | cut -d'=' -f2)

        echo "$workers,$bs,$elapsed_secs,$overall_tokens_per_sec,$overall_records_per_sec,$read_avg_mb_per_sec,$convert_avg_mb_per_sec,$total_tokens,$total_records,${read_workers:-},${tokenize_workers:-},${write_workers:-}" >> "$RESULTS_CSV"

        echo "  用时=${elapsed_secs}s, tokens/s=${overall_tokens_per_sec}, convert_MB/s=${convert_avg_mb_per_sec}, 墙钟=${wall_time}s" | tee -a "$RESULTS_LOG"
      else
        echo "$workers,$bs,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,${read_workers:-},${tokenize_workers:-},${write_workers:-}" >> "$RESULTS_CSV"
        echo "  错误: 未找到 summary 行 (可能超时)" | tee -a "$RESULTS_LOG"
      fi

      # 清理
      rm -f "${output_prefix}.bin" "${output_prefix}.idx" 2>/dev/null || true
    done
  done
done

echo "=== 测试完成 ===" | tee -a "$RESULTS_LOG"
echo "结束时间: $(date)" | tee -a "$RESULTS_LOG"
echo "结果文件: $RESULTS_CSV" | tee -a "$RESULTS_LOG"
echo "日志文件: $RESULTS_LOG" | tee -a "$RESULTS_LOG"

# 生成简单图表脚本（基于新的指标）
if command -v python3 >/dev/null 2>&1; then
    cat > "$OUTPUT_DIR/plot_results.py" << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python3 plot_results.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
if not os.path.exists(csv_file):
    print(f"File not found: {csv_file}")
    sys.exit(1)

# 读取数据
df = pd.read_csv(csv_file)
df = df[df['elapsed_secs'] != 'ERROR']  # 过滤错误行
df = df.astype({'workers': int, 'batch_size': int, 'elapsed_secs': float,
                'overall_tokens_per_sec': float, 'overall_records_per_sec': float,
                'read_avg_mb_per_sec': float, 'convert_avg_mb_per_sec': float})

# 创建图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 1. Workers vs overall tokens/s （按不同batch_size分组）
for bs, g in df.groupby('batch_size'):
    ax1.plot(g['workers'], g['overall_tokens_per_sec'], 'o-', label=f'bs={bs}')
ax1.set_xlabel('Workers')
ax1.set_ylabel('Overall Tokens/s')
ax1.set_title('Workers vs Tokens/s (by batch_size)')
ax1.grid(True)
ax1.legend()

# 2. Batch_size vs overall tokens/s （按不同workers分组）
for w, g in df.groupby('workers'):
    ax2.plot(g['batch_size'], g['overall_tokens_per_sec'], 'o-', label=f'w={w}')
ax2.set_xlabel('Batch Size')
ax2.set_ylabel('Overall Tokens/s')
ax2.set_title('Batch Size vs Tokens/s (by workers)')
ax2.grid(True)
ax2.legend()

# 3. Workers vs Convert Speed
for bs, g in df.groupby('batch_size'):
    ax3.plot(g['workers'], g['convert_avg_mb_per_sec'], 'o-', label=f'bs={bs}')
ax3.set_xlabel('Workers')
ax3.set_ylabel('Convert Speed (MB/s)')
ax3.set_title('Workers vs Convert Speed')
ax3.grid(True)
ax3.legend()

# 4. Workers vs Elapsed Time
for bs, g in df.groupby('batch_size'):
    ax4.plot(g['workers'], g['elapsed_secs'], 'o-', label=f'bs={bs}')
ax4.set_xlabel('Workers')
ax4.set_ylabel('Elapsed Time (s)')
ax4.set_title('Workers vs Elapsed Time')
ax4.grid(True)
ax4.legend()

plt.tight_layout()
output_file = csv_file.replace('.csv', '_plot.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"图表已保存到: {output_file}")

# 显示最优配置（按 tokens/s 最大）
best_idx = df['overall_tokens_per_sec'].idxmax()
best_workers = df.loc[best_idx, 'workers']
best_bs = df.loc[best_idx, 'batch_size']
best_tps = df.loc[best_idx, 'overall_tokens_per_sec']
print(f"\n最优配置: workers={best_workers}, batch_size={best_bs}, tokens/s={best_tps}")
EOF

    echo "" | tee -a "$RESULTS_LOG"
    echo "生成图表脚本: $OUTPUT_DIR/plot_results.py" | tee -a "$RESULTS_LOG"
    echo "运行命令: python3 $OUTPUT_DIR/plot_results.py $RESULTS_CSV" | tee -a "$RESULTS_LOG"
fi

echo ""
echo "测试完成！结果保存在:"
echo "  CSV: $RESULTS_CSV"
echo "  LOG: $RESULTS_LOG"

