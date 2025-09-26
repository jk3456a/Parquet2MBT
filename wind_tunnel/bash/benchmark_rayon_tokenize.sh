#!/bin/bash

# Rayon Tokenize 对比测试脚本：对比传统多线程 vs rayon 内部并行化
set -euo pipefail

# 默认配置
INPUT_DIR="${INPUT_DIR:-/home/lizhen/dataset/zh__CCI4.0-M2-Base-v1-newest_zh_cc-high-loss0__2025091500}"
PATTERN="${PATTERN:-*.parquet}"
TOKENIZER="${TOKENIZER:-../testdata/tokenizer/tokenizer.json}"
OUTPUT_DIR="${OUTPUT_DIR:-../testdata/benchmark/rayon_tokenize_comparison}"
BINARY="${BINARY:-../target/release/parquet2mbt}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
REPEAT="${REPEAT:-1}"

# 固定最优配置（基于之前的实验结果）
FIXED_READ_WORKERS=4
FIXED_WRITE_WORKERS=2
BATCH_SIZE=8192

# 测试不同的 tokenize worker 数量
TOKENIZE_WORKER_COUNTS=(${TOKENIZE_WORKER_COUNTS:-120 122 126 128})

mkdir -p "$OUTPUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)

# 增量测试：查找现有的最新CSV文件
SKIP_EXISTING="${SKIP_EXISTING:-true}"
EXISTING_CSV=""
if [ "$SKIP_EXISTING" = "true" ]; then
    # 找到行数最多的CSV文件（最完整的数据）
    EXISTING_CSV=$(find "$OUTPUT_DIR" -name "rayon_tokenize_comparison_*.csv" -type f 2>/dev/null | \
        xargs -I {} sh -c 'echo "$(wc -l < "{}")" "{}"' | \
        sort -nr | head -1 | cut -d' ' -f2- || echo "")
    
    if [ -n "$EXISTING_CSV" ] && [ -f "$EXISTING_CSV" ]; then
        echo "找到现有测试数据: $EXISTING_CSV ($(wc -l < "$EXISTING_CSV") 行)"
        RESULTS_CSV="$EXISTING_CSV"
        RESULTS_LOG="${EXISTING_CSV%.csv}.log"
        echo "将追加到现有文件，跳过已测试的配置"
    else
        RESULTS_CSV="$OUTPUT_DIR/rayon_tokenize_comparison_${TS}.csv"
        RESULTS_LOG="$OUTPUT_DIR/rayon_tokenize_comparison_${TS}.log"
        echo "未找到现有数据，创建新文件"
        # CSV 表头
        echo "tokenize_mode,tokenize_workers,total_workers,uptime_secs,overall_tokens_per_sec,overall_records_per_sec,overall_read_mb_per_sec,overall_convert_mb_per_sec,interval_tokens_per_sec,interval_records_per_sec,interval_read_mb_per_sec,interval_convert_mb_per_sec,tokens_total,records_total" > "$RESULTS_CSV"
    fi
else
    RESULTS_CSV="$OUTPUT_DIR/rayon_tokenize_comparison_${TS}.csv"
    RESULTS_LOG="$OUTPUT_DIR/rayon_tokenize_comparison_${TS}.log"
    # CSV 表头
    echo "tokenize_mode,tokenize_workers,total_workers,uptime_secs,overall_tokens_per_sec,overall_records_per_sec,overall_read_mb_per_sec,overall_convert_mb_per_sec,interval_tokens_per_sec,interval_records_per_sec,interval_read_mb_per_sec,interval_convert_mb_per_sec,tokens_total,records_total" > "$RESULTS_CSV"
fi

# 检查是否已经测试过某个配置
is_already_tested() {
    local mode="$1"
    local workers="$2"
    local run="$3"
    
    if [ ! -f "$RESULTS_CSV" ]; then
        return 1  # 文件不存在，未测试过
    fi
    
    # 检查CSV中是否已有该配置的数据
    grep -q "^${mode},${workers}," "$RESULTS_CSV" 2>/dev/null
}

echo "=== Rayon Tokenize 对比测试 ===" | tee "$RESULTS_LOG"
echo "开始时间: $(date)" | tee -a "$RESULTS_LOG"
echo "INPUT_DIR=$INPUT_DIR" | tee -a "$RESULTS_LOG"
echo "TOKENIZER=$TOKENIZER" | tee -a "$RESULTS_LOG"
echo "固定配置: READ_WORKERS=$FIXED_READ_WORKERS, WRITE_WORKERS=$FIXED_WRITE_WORKERS, BATCH_SIZE=$BATCH_SIZE" | tee -a "$RESULTS_LOG"
echo "TOKENIZE_WORKERS测试范围: ${TOKENIZE_WORKER_COUNTS[*]}" | tee -a "$RESULTS_LOG"
echo "测试模式: 传统多线程 vs Rayon内部并行化" | tee -a "$RESULTS_LOG"
echo "" | tee -a "$RESULTS_LOG"

for tokenize_workers in "${TOKENIZE_WORKER_COUNTS[@]}"; do
  for r in $(seq 1 "$REPEAT"); do
    total_workers=$((FIXED_READ_WORKERS + tokenize_workers + FIXED_WRITE_WORKERS))
    
    # 测试1: 传统多线程模式
    if is_already_tested "traditional" "$tokenize_workers" "$r"; then
        echo "跳过已测试配置: [传统多线程] tokenize_workers=$tokenize_workers (run#$r)" | tee -a "$RESULTS_LOG"
    else
        echo "=== 测试 [传统多线程] tokenize_workers=$tokenize_workers (总workers=$total_workers) (run#$r) ===" | tee -a "$RESULTS_LOG"
    output_prefix="$OUTPUT_DIR/traditional_t${tokenize_workers}_r${r}_$TS"
    start_time=$(date +%s)

    cmd_traditional=("$BINARY" \
      --input-dir "$INPUT_DIR" \
      --pattern "$PATTERN" \
      --text-cols content \
      --tokenizer "$TOKENIZER" \
      --output-prefix "$output_prefix" \
      --batch-size "$BATCH_SIZE" \
      --workers "$total_workers" \
      --read-workers "$FIXED_READ_WORKERS" \
      --tokenize-workers "$tokenize_workers" \
      --write-workers "$FIXED_WRITE_WORKERS" \
      --metrics-interval 10 \
      --no-write)

    log_file="$OUTPUT_DIR/traditional_t${tokenize_workers}_r${r}.log"
    set +e
    timeout --foreground --signal=INT --kill-after=5s "$TIMEOUT_SEC" "${cmd_traditional[@]}" &> "$log_file" &
    run_pid=$!
    wait "$run_pid"
    run_rc=$?
    set -e
    
    if [ $run_rc -ne 0 ]; then
      echo "  运行超时/失败: traditional tokenize_workers=$tokenize_workers run#$r (rc=$run_rc)" | tee -a "$RESULTS_LOG"
    fi

    end_time=$(date +%s)
    wall_time=$((end_time - start_time))

    # 解析结果
    metrics_line=$(grep -F 'component="metrics"' "$log_file" | tail -1 || true)
    if [ -n "$metrics_line" ]; then
      uptime_secs=$(echo "$metrics_line" | grep -o 'uptime_secs=[0-9]*' | cut -d'=' -f2)
      overall_tokens_per_sec=$(echo "$metrics_line" | grep -o 'overall_tokens_per_sec="[0-9.]*"' | cut -d'"' -f2)
      overall_records_per_sec=$(echo "$metrics_line" | grep -o 'overall_records_per_sec="[0-9.]*"' | cut -d'"' -f2)
      overall_read_mb_per_sec=$(echo "$metrics_line" | grep -o 'overall_read_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
      overall_convert_mb_per_sec=$(echo "$metrics_line" | grep -o 'overall_convert_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
      interval_tokens_per_sec=$(echo "$metrics_line" | grep -o 'interval_tokens_per_sec="[0-9.]*"' | cut -d'"' -f2)
      interval_records_per_sec=$(echo "$metrics_line" | grep -o 'interval_records_per_sec="[0-9.]*"' | cut -d'"' -f2)
      interval_read_mb_per_sec=$(echo "$metrics_line" | grep -o 'interval_read_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
      interval_convert_mb_per_sec=$(echo "$metrics_line" | grep -o 'interval_convert_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
      tokens_total=$(echo "$metrics_line" | grep -o 'tokens_total=[0-9]*' | cut -d'=' -f2)
      records_total=$(echo "$metrics_line" | grep -o 'records_total=[0-9]*' | cut -d'=' -f2)

      echo "traditional,$tokenize_workers,$total_workers,${uptime_secs:-},${overall_tokens_per_sec:-},${overall_records_per_sec:-},${overall_read_mb_per_sec:-},${overall_convert_mb_per_sec:-},${interval_tokens_per_sec:-},${interval_records_per_sec:-},${interval_read_mb_per_sec:-},${interval_convert_mb_per_sec:-},${tokens_total:-},${records_total:-}" >> "$RESULTS_CSV"

      echo "  [传统] 稳定(uptime=${uptime_secs:-?}s): tokens/s=${overall_tokens_per_sec:-?}, interval_tokens/s=${interval_tokens_per_sec:-?}, 墙钟=${wall_time}s" | tee -a "$RESULTS_LOG"
    else
      echo "traditional,$tokenize_workers,$total_workers,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS_CSV"
      echo "  [传统] 错误: 未找到 metrics snapshot 行" | tee -a "$RESULTS_LOG"
    fi
    fi  # 结束传统模式的 if 判断

    # 测试2: Rayon内部并行化模式
    if is_already_tested "rayon" "$tokenize_workers" "$r"; then
        echo "跳过已测试配置: [Rayon并行化] tokenize_workers=$tokenize_workers (run#$r)" | tee -a "$RESULTS_LOG"
    else
        echo "=== 测试 [Rayon并行化] tokenize_workers=$tokenize_workers (总workers=$total_workers) (run#$r) ===" | tee -a "$RESULTS_LOG"
    output_prefix="$OUTPUT_DIR/rayon_t${tokenize_workers}_r${r}_$TS"
    start_time=$(date +%s)

    # 设置 RAYON_NUM_THREADS 环境变量
    # 合理分配：总核心数 / tokenize_workers，避免过度订阅
    TOTAL_CORES=128
    RAYON_THREADS_PER_WORKER=$((TOTAL_CORES / tokenize_workers))
    if [ $RAYON_THREADS_PER_WORKER -lt 1 ]; then
        RAYON_THREADS_PER_WORKER=1
    elif [ $RAYON_THREADS_PER_WORKER -gt 4 ]; then
        RAYON_THREADS_PER_WORKER=4  # 限制最大值，避免单个batch占用过多资源
    fi
    export RAYON_NUM_THREADS=$RAYON_THREADS_PER_WORKER
    echo "  设置 RAYON_NUM_THREADS=$RAYON_THREADS_PER_WORKER (每个tokenize worker内部使用$RAYON_THREADS_PER_WORKER个线程)" | tee -a "$RESULTS_LOG"

    cmd_rayon=("$BINARY" \
      --input-dir "$INPUT_DIR" \
      --pattern "$PATTERN" \
      --text-cols content \
      --tokenizer "$TOKENIZER" \
      --output-prefix "$output_prefix" \
      --batch-size "$BATCH_SIZE" \
      --workers "$total_workers" \
      --read-workers "$FIXED_READ_WORKERS" \
      --tokenize-workers "$tokenize_workers" \
      --write-workers "$FIXED_WRITE_WORKERS" \
      --metrics-interval 10 \
      --no-write \
      --use-rayon-tokenize)

    log_file="$OUTPUT_DIR/rayon_t${tokenize_workers}_r${r}.log"
    set +e
    timeout --foreground --signal=INT --kill-after=5s "$TIMEOUT_SEC" "${cmd_rayon[@]}" &> "$log_file" &
    run_pid=$!
    wait "$run_pid"
    run_rc=$?
    set -e
    
    if [ $run_rc -ne 0 ]; then
      echo "  运行超时/失败: rayon tokenize_workers=$tokenize_workers run#$r (rc=$run_rc)" | tee -a "$RESULTS_LOG"
    fi

    end_time=$(date +%s)
    wall_time=$((end_time - start_time))

    # 解析结果
    metrics_line=$(grep -F 'component="metrics"' "$log_file" | tail -1 || true)
    if [ -n "$metrics_line" ]; then
      uptime_secs=$(echo "$metrics_line" | grep -o 'uptime_secs=[0-9]*' | cut -d'=' -f2)
      overall_tokens_per_sec=$(echo "$metrics_line" | grep -o 'overall_tokens_per_sec="[0-9.]*"' | cut -d'"' -f2)
      overall_records_per_sec=$(echo "$metrics_line" | grep -o 'overall_records_per_sec="[0-9.]*"' | cut -d'"' -f2)
      overall_read_mb_per_sec=$(echo "$metrics_line" | grep -o 'overall_read_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
      overall_convert_mb_per_sec=$(echo "$metrics_line" | grep -o 'overall_convert_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
      interval_tokens_per_sec=$(echo "$metrics_line" | grep -o 'interval_tokens_per_sec="[0-9.]*"' | cut -d'"' -f2)
      interval_records_per_sec=$(echo "$metrics_line" | grep -o 'interval_records_per_sec="[0-9.]*"' | cut -d'"' -f2)
      interval_read_mb_per_sec=$(echo "$metrics_line" | grep -o 'interval_read_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
      interval_convert_mb_per_sec=$(echo "$metrics_line" | grep -o 'interval_convert_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
      tokens_total=$(echo "$metrics_line" | grep -o 'tokens_total=[0-9]*' | cut -d'=' -f2)
      records_total=$(echo "$metrics_line" | grep -o 'records_total=[0-9]*' | cut -d'=' -f2)

      echo "rayon,$tokenize_workers,$total_workers,${uptime_secs:-},${overall_tokens_per_sec:-},${overall_records_per_sec:-},${overall_read_mb_per_sec:-},${overall_convert_mb_per_sec:-},${interval_tokens_per_sec:-},${interval_records_per_sec:-},${interval_read_mb_per_sec:-},${interval_convert_mb_per_sec:-},${tokens_total:-},${records_total:-}" >> "$RESULTS_CSV"

      echo "  [Rayon] 稳定(uptime=${uptime_secs:-?}s): tokens/s=${overall_tokens_per_sec:-?}, interval_tokens/s=${interval_tokens_per_sec:-?}, 墙钟=${wall_time}s" | tee -a "$RESULTS_LOG"
    else
      echo "rayon,$tokenize_workers,$total_workers,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS_CSV"
      echo "  [Rayon] 错误: 未找到 metrics snapshot 行" | tee -a "$RESULTS_LOG"
    fi
    fi  # 结束 Rayon 模式的 if 判断

    unset RAYON_NUM_THREADS
    echo "" | tee -a "$RESULTS_LOG"
  done
done

echo "=== 测试完成 ===" | tee -a "$RESULTS_LOG"
echo "结束时间: $(date)" | tee -a "$RESULTS_LOG"
echo "结果文件: $RESULTS_CSV" | tee -a "$RESULTS_LOG"
echo "日志文件: $RESULTS_LOG" | tee -a "$RESULTS_LOG"

# 生成对比图表脚本
if command -v python3 >/dev/null 2>&1; then
    cat > "$OUTPUT_DIR/plot_comparison.py" << 'EOF'
#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python3 plot_comparison.py <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]
if not os.path.exists(csv_file):
    print(f"File not found: {csv_file}")
    sys.exit(1)

# 读取数据
df = pd.read_csv(csv_file)
df = df[df['uptime_secs'] != 'ERROR']  # 过滤错误行
df = df.astype({'tokenize_workers': int, 'total_workers': int, 'uptime_secs': float,
                'overall_tokens_per_sec': float, 'interval_tokens_per_sec': float})

# 分离两种模式的数据
df_traditional = df[df['tokenize_mode'] == 'traditional']
df_rayon = df[df['tokenize_mode'] == 'rayon']

# 创建对比图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 1. Overall Tokens/s 对比
ax1.plot(df_traditional['tokenize_workers'], df_traditional['overall_tokens_per_sec'], 
         'o-', color='blue', linewidth=2, markersize=8, label='Traditional Multi-thread')
ax1.plot(df_rayon['tokenize_workers'], df_rayon['overall_tokens_per_sec'], 
         's-', color='red', linewidth=2, markersize=8, label='Rayon Internal Parallel')
ax1.set_xlabel('Tokenize Workers')
ax1.set_ylabel('Overall Tokens/s')
ax1.set_title('Tokenization Performance Comparison')
ax1.grid(True, alpha=0.3)
ax1.legend()

# 2. Interval Tokens/s 对比
ax2.plot(df_traditional['tokenize_workers'], df_traditional['interval_tokens_per_sec'], 
         'o-', color='blue', linewidth=2, markersize=8, label='Traditional Multi-thread')
ax2.plot(df_rayon['tokenize_workers'], df_rayon['interval_tokens_per_sec'], 
         's-', color='red', linewidth=2, markersize=8, label='Rayon Internal Parallel')
ax2.set_xlabel('Tokenize Workers')
ax2.set_ylabel('Interval Tokens/s (Stable State)')
ax2.set_title('Stable State Performance Comparison')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
output_file = csv_file.replace('.csv', '_comparison_plot.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Comparison plot saved to: {output_file}")

# 性能分析
print(f"\n=== Rayon vs Traditional Tokenization Analysis ===")
for workers in sorted(df['tokenize_workers'].unique()):
    trad_perf = df_traditional[df_traditional['tokenize_workers'] == workers]['overall_tokens_per_sec']
    rayon_perf = df_rayon[df_rayon['tokenize_workers'] == workers]['overall_tokens_per_sec']
    
    if not trad_perf.empty and not rayon_perf.empty:
        trad_val = trad_perf.iloc[0]
        rayon_val = rayon_perf.iloc[0]
        improvement = ((rayon_val - trad_val) / trad_val) * 100
        print(f"  {workers} workers: Traditional={trad_val:.0f}T/s, Rayon={rayon_val:.0f}T/s, Improvement={improvement:+.1f}%")

# 找出最佳配置
best_trad = df_traditional.loc[df_traditional['overall_tokens_per_sec'].idxmax()]
best_rayon = df_rayon.loc[df_rayon['overall_tokens_per_sec'].idxmax()]

print(f"\n=== Best Configurations ===")
print(f"Traditional Best: {best_trad['tokenize_workers']} workers, {best_trad['overall_tokens_per_sec']:.0f} tokens/s")
print(f"Rayon Best: {best_rayon['tokenize_workers']} workers, {best_rayon['overall_tokens_per_sec']:.0f} tokens/s")

overall_improvement = ((best_rayon['overall_tokens_per_sec'] - best_trad['overall_tokens_per_sec']) / best_trad['overall_tokens_per_sec']) * 100
print(f"Overall Best Improvement: {overall_improvement:+.1f}%")
EOF

    echo "" | tee -a "$RESULTS_LOG"
    echo "生成对比图表脚本: $OUTPUT_DIR/plot_comparison.py" | tee -a "$RESULTS_LOG"
    echo "运行命令: python3 $OUTPUT_DIR/plot_comparison.py $RESULTS_CSV" | tee -a "$RESULTS_LOG"
fi

echo ""
echo "测试完成！结果保存在:"
echo "  CSV: $RESULTS_CSV"
echo "  LOG: $RESULTS_LOG"
echo ""
echo "使用方法对比:"
echo "  传统模式: 无需额外参数"
echo "  Rayon模式: 添加 --use-rayon-tokenize 参数"
echo "  环境变量: RAYON_NUM_THREADS 控制 rayon 并行度"
