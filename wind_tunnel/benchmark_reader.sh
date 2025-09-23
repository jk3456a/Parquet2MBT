#!/bin/bash

# Reader扩展性对比测试脚本
# 1. 在纯读模式下，测试不同数量Reader Worker的理论IO上限
# 2. 在带载模式下(固定120个Tokenizer)，测试其对最终性能(tokens/s)的贡献
set -euo pipefail

# --- 配置 ---
INPUT_DIR="${INPUT_DIR:-/home/lizhen/dataset/zh__CCI4.0-M2-Base-v1-newest_zh_cc-high-loss0__2025091500}"
PATTERN="${PATTERN:-*.parquet}"
TOKENIZER="${TOKENIZER:-../testdata/tokenizer/tokenizer.json}"
OUTPUT_DIR="${OUTPUT_DIR:-../testdata/benchmark/reader_scaling}"
BINARY="${BINARY:-../target/release/parquet2mbt}"
TIMEOUT_SEC="${TIMEOUT_SEC:-30}"
REPEAT="${REPEAT:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-true}"

# 测试的Read Worker数量范围
READ_WORKER_COUNTS=(${READ_WORKER_COUNTS:-1 2 3 4 5 6 7 8})
# 在带载测试中，固定的Tokenizer数量
FIXED_TOKENIZE_WORKERS=120

# --- 脚本主体 ---
mkdir -p "$OUTPUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)

# 增量测试：检查是否存在现有结果文件
EXISTING_CSV=""
if [ "$SKIP_EXISTING" = "true" ]; then
  # 查找数据最完整的CSV文件（行数最多的）
  EXISTING_CSV=""
  MAX_LINES=0
  for csv_file in "$OUTPUT_DIR"/reader_scaling_*.csv; do
    if [ -f "$csv_file" ]; then
      line_count=$(wc -l < "$csv_file" 2>/dev/null || echo 0)
      if [ "$line_count" -gt "$MAX_LINES" ]; then
        MAX_LINES=$line_count
        EXISTING_CSV="$csv_file"
      fi
    fi
  done
  
  if [ -n "$EXISTING_CSV" ] && [ -f "$EXISTING_CSV" ]; then
    echo "发现现有结果文件: $EXISTING_CSV (包含 $MAX_LINES 行数据)"
    echo "将跳过已测试的参数组合，进行增量测试"
    RESULTS_CSV="$EXISTING_CSV"
    RESULTS_LOG="${EXISTING_CSV%.csv}.log"
  else
    RESULTS_CSV="$OUTPUT_DIR/reader_scaling_${TS}.csv"
    RESULTS_LOG="$OUTPUT_DIR/reader_scaling_${TS}.log"
    echo "test_mode,read_workers,tokenize_workers,write_workers,total_workers,batch_size,uptime_secs,overall_tokens_per_sec,overall_records_per_sec,overall_read_mb_per_sec,overall_convert_mb_per_sec,interval_tokens_per_sec,interval_records_per_sec,interval_read_mb_per_sec,interval_convert_mb_per_sec,interval_tokenize_input_mb_per_sec,tokens_total,records_total" > "$RESULTS_CSV"
  fi
else
  RESULTS_CSV="$OUTPUT_DIR/reader_scaling_${TS}.csv"
  RESULTS_LOG="$OUTPUT_DIR/reader_scaling_${TS}.log"
  echo "test_mode,read_workers,tokenize_workers,write_workers,total_workers,batch_size,uptime_secs,overall_tokens_per_sec,overall_records_per_sec,overall_read_mb_per_sec,overall_convert_mb_per_sec,interval_tokens_per_sec,interval_records_per_sec,interval_read_mb_per_sec,interval_convert_mb_per_sec,interval_tokenize_input_mb_per_sec,tokens_total,records_total" > "$RESULTS_CSV"
fi

# 日志文件头部信息
if [ "$SKIP_EXISTING" = "true" ] && [ -f "$RESULTS_LOG" ]; then
  echo "" | tee -a "$RESULTS_LOG"
  echo "=== Reader扩展性对比测试增量继续 ===" | tee -a "$RESULTS_LOG"
  echo "继续时间: $(date)" | tee -a "$RESULTS_LOG"
  echo "READ_WORKER测试范围: ${READ_WORKER_COUNTS[*]}" | tee -a "$RESULTS_LOG"
  echo "" | tee -a "$RESULTS_LOG"
else
  echo "=== Reader扩展性对比测试 ===" | tee "$RESULTS_LOG"
  echo "开始时间: $(date)" | tee -a "$RESULTS_LOG"
  echo "INPUT_DIR=$INPUT_DIR" | tee -a "$RESULTS_LOG"
  echo "READ_WORKER测试范围: ${READ_WORKER_COUNTS[*]}" | tee -a "$RESULTS_LOG"
  echo "固定TOKENIZE_WORKERS(带载模式): $FIXED_TOKENIZE_WORKERS" | tee -a "$RESULTS_LOG"
  echo "" | tee -a "$RESULTS_LOG"
fi

# 函数：检查参数组合是否已测试
is_already_tested() {
  local test_mode=$1
  local read_workers=$2
  if [ -f "$RESULTS_CSV" ] && [ "$SKIP_EXISTING" = "true" ]; then
    # 检查CSV文件中是否已存在该配置
    grep -q "^$test_mode,$read_workers," "$RESULTS_CSV" 2>/dev/null
  else
    return 1  # 不跳过
  fi
}

for read_workers in "${READ_WORKER_COUNTS[@]}"; do
  for r in $(seq 1 "$REPEAT"); do
    
    # --- 测试模式A: 纯读 (Read-Only) ---
    if is_already_tested "read_only" "$read_workers"; then
      echo "=== 跳过已测试 [纯读模式] read_workers=$read_workers (run#$r) ===" | tee -a "$RESULTS_LOG"
    else
      echo "=== 测试 [纯读模式] read_workers=$read_workers (run#$r) ===" | tee -a "$RESULTS_LOG"
      output_prefix_ro="$OUTPUT_DIR/ro_r${read_workers}_r${r}_$TS"
    
      cmd_ro=("$BINARY" \
        --input-dir "$INPUT_DIR" --pattern "$PATTERN" --text-cols content \
        --tokenizer "$TOKENIZER" --output-prefix "$output_prefix_ro" \
        --batch-size 8192 --workers "$read_workers" \
        --metrics-interval 5 --no-tokenize --no-write)

      log_file_ro="$OUTPUT_DIR/run_ro_r${read_workers}_r${r}.log"
      timeout --foreground "$TIMEOUT_SEC" "${cmd_ro[@]}" &> "$log_file_ro" || true
      
      metrics_line_ro=$(grep -F 'component="metrics"' "$log_file_ro" | tail -1 || true)
      if [ -n "$metrics_line_ro" ]; then
        interval_read_mb_per_sec_ro=$(echo "$metrics_line_ro" | grep -o 'interval_read_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
        echo "read_only,$read_workers,0,0,$read_workers,8192,$(echo "$metrics_line_ro" | grep -o 'uptime_secs=[0-9]*' | cut -d'=' -f2),0,0,0,0,0,0,${interval_read_mb_per_sec_ro:-0},0,0,0,0" >> "$RESULTS_CSV"
        echo "  [纯读] 结果: interval_read_MB/s = ${interval_read_mb_per_sec_ro:-ERROR}" | tee -a "$RESULTS_LOG"
      else
        echo "read_only,$read_workers,0,0,$read_workers,8192,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS_CSV"
        echo "  [纯读] 错误: 未找到 metrics" | tee -a "$RESULTS_LOG"
      fi
    fi

    # --- 测试模式B: 带载 (Read + Tokenize) ---
    if is_already_tested "read_tokenize" "$read_workers"; then
      echo "=== 跳过已测试 [带载模式] read_workers=$read_workers (run#$r) ===" | tee -a "$RESULTS_LOG"
    else
      echo "=== 测试 [带载模式] read_workers=$read_workers tokenize_workers=$FIXED_TOKENIZE_WORKERS (run#$r) ===" | tee -a "$RESULTS_LOG"
      total_workers=$((read_workers + FIXED_TOKENIZE_WORKERS + 1))
      output_prefix_rt="$OUTPUT_DIR/rt_r${read_workers}_t${FIXED_TOKENIZE_WORKERS}_r${r}_$TS"

      cmd_rt=("$BINARY" \
        --input-dir "$INPUT_DIR" --pattern "$PATTERN" --text-cols content \
        --tokenizer "$TOKENIZER" --output-prefix "$output_prefix_rt" \
        --batch-size 8192 --workers "$total_workers" \
        --read-workers "$read_workers" --tokenize-workers "$FIXED_TOKENIZE_WORKERS" --write-workers 1 \
        --metrics-interval 5 --no-write)

      log_file_rt="$OUTPUT_DIR/run_rt_r${read_workers}_t${FIXED_TOKENIZE_WORKERS}_r${r}.log"
      timeout --foreground "$TIMEOUT_SEC" "${cmd_rt[@]}" &> "$log_file_rt" || true

      metrics_line_rt=$(grep -F 'component="metrics"' "$log_file_rt" | tail -1 || true)
      if [ -n "$metrics_line_rt" ]; then
        interval_read_mb_per_sec_rt=$(echo "$metrics_line_rt" | grep -o 'interval_read_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
        interval_tokens_per_sec_rt=$(echo "$metrics_line_rt" | grep -o 'interval_tokens_per_sec="[0-9.]*"' | cut -d'"' -f2)
        interval_tokenize_input_mb_per_sec_rt=$(echo "$metrics_line_rt" | grep -o 'interval_tokenize_input_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
        interval_convert_mb_per_sec_rt=$(echo "$metrics_line_rt" | grep -o 'interval_convert_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
        echo "read_tokenize,$read_workers,$FIXED_TOKENIZE_WORKERS,1,$total_workers,8192,$(echo "$metrics_line_rt" | grep -o 'uptime_secs=[0-9]*' | cut -d'=' -f2),0,0,0,${interval_convert_mb_per_sec_rt:-0},${interval_tokens_per_sec_rt:-0},0,${interval_read_mb_per_sec_rt:-0},${interval_convert_mb_per_sec_rt:-0},${interval_tokenize_input_mb_per_sec_rt:-0},0,0" >> "$RESULTS_CSV"
        echo "  [带载] 结果: tokenize_input=${interval_tokenize_input_mb_per_sec_rt:-ERROR}MB/s, tokenize_output=${interval_convert_mb_per_sec_rt:-ERROR}MB/s, tokens=${interval_tokens_per_sec_rt:-ERROR}/s" | tee -a "$RESULTS_LOG"
      else
        echo "read_tokenize,$read_workers,$FIXED_TOKENIZE_WORKERS,1,$total_workers,8192,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS_CSV"
        echo "  [带载] 错误: 未找到 metrics" | tee -a "$RESULTS_LOG"
      fi
    fi
    echo "" | tee -a "$RESULTS_LOG"
  done
done

echo "=== 测试完成 ===" | tee -a "$RESULTS_LOG"
echo "结果文件: $RESULTS_CSV" | tee -a "$RESULTS_LOG"

# 生成新的绘图脚本
if command -v python3 >/dev/null 2>&1; then
    PLOTTER_PY="$OUTPUT_DIR/plot_results.py"
    cat > "$PLOTTER_PY" << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python3 plot_results.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
df = pd.read_csv(csv_file)

df_ro = df[df['test_mode'] == 'read_only'].copy()
df_rt = df[df['test_mode'] == 'read_tokenize'].copy()

# 只对需要的列进行NaN处理
df_ro = df_ro.dropna(subset=['interval_read_mb_per_sec'])
df_rt = df_rt.dropna(subset=['interval_tokenize_input_mb_per_sec', 'interval_tokens_per_sec'])

fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))
fig.suptitle('Reader Performance: Pure I/O vs. With Tokenizer Load')

# Main plot: I/O Throughput Comparison
ax1.set_xlabel('Number of Read Workers')
ax1.set_ylabel('I/O Throughput (MB/s)', color='blue')
ax1.plot(df_ro['read_workers'], df_ro['interval_read_mb_per_sec'], 'o-', color='deepskyblue', linewidth=2, markersize=8, label='Pure I/O (Read-Only)')
ax1.plot(df_rt['read_workers'], df_rt['interval_tokenize_input_mb_per_sec'], 's-', color='red', linewidth=2, markersize=8, label='With Tokenizer Load')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Find and annotate the optimal point
if not df_rt.empty:
    best_idx = df_rt['interval_tokens_per_sec'].idxmax()
    best_workers = df_rt.loc[best_idx, 'read_workers']
    best_input_speed = df_rt.loc[best_idx, 'interval_tokenize_input_mb_per_sec']
    ax1.annotate(f'Optimal: {best_workers} readers\n{best_input_speed:.1f} MB/s input',
                 xy=(best_workers, best_input_speed),
                 xytext=(best_workers, best_input_speed * 0.8),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=8),
                 ha='center', va='top')

plt.tight_layout()
plt.savefig(csv_file.replace('.csv', '_plot.png'), dpi=150)
print(f"Plot saved to: {csv_file.replace('.csv', '_plot.png')}")

# Summary
print("\n=== Reader Performance Analysis ===")
if not df_rt.empty and not df_ro.empty:
    best_row = df_rt.loc[best_idx]
    pure_io_at_best = df_ro[df_ro['read_workers'] == best_workers]['interval_read_mb_per_sec'].iloc[0] if len(df_ro[df_ro['read_workers'] == best_workers]) > 0 else 0
    
    print(f"Optimal configuration: {best_workers} read workers")
    print(f"  - Pure I/O throughput: {pure_io_at_best:.1f} MB/s")
    print(f"  - With tokenizer load: {best_row['interval_tokenize_input_mb_per_sec']:.1f} MB/s")
    
    if pure_io_at_best > 0:
        efficiency = (best_row['interval_tokenize_input_mb_per_sec'] / pure_io_at_best) * 100
        print(f"  - Tokenizer efficiency: {efficiency:.1f}% of pure I/O capacity")
    
    best_tps = best_row['interval_tokens_per_sec']
    print(f"  - Token processing rate: {best_tps/1e6:.1f}M tokens/s")
EOF

    echo "运行命令: python3 $PLOTTER_PY $RESULTS_CSV" | tee -a "$RESULTS_LOG"
    python3 "$PLOTTER_PY" "$RESULTS_CSV"
fi

