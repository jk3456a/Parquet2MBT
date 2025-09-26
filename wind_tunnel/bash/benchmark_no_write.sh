#!/bin/bash

# Pipeline优化实验脚本：基于最佳配置(128 workers, 8192 batch_size)测试不同的read/tokenize worker比例
set -euo pipefail

# 默认配置（可通过环境变量或命令行覆盖）
INPUT_DIR="${INPUT_DIR:-/home/lizhen/dataset/zh__CCI4.0-M2-Base-v1-newest_zh_cc-high-loss0__2025091500}"
PATTERN="${PATTERN:-*.parquet}"
TOKENIZER="${TOKENIZER:-../testdata/tokenizer/tokenizer.json}"
OUTPUT_DIR="${OUTPUT_DIR:-../testdata/benchmark/read_tokenize_ratio}"
BINARY="${BINARY:-../target/release/parquet2mbt}"
NO_WRITE="${NO_WRITE:-true}"
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
REPEAT="${REPEAT:-1}"

# 固定最佳配置
TOTAL_WORKERS=128
BATCH_SIZE=8192
WRITE_WORKERS=1

# Pipeline比例测试：read worker数量通过/2不断缩减
# 从默认的25个read workers开始，逐步减少到1个
READ_WORKER_COUNTS=(${READ_WORKER_COUNTS:-25 20 16 12 10 8 6 4 3 2 1})

# 增量测试支持：检查已完成的测试
SKIP_EXISTING="${SKIP_EXISTING:-true}"

mkdir -p "$OUTPUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)

# 增量测试：检查是否存在现有结果文件
EXISTING_CSV=""
if [ "$SKIP_EXISTING" = "true" ]; then
  # 查找最新的pipeline结果文件
  EXISTING_CSV=$(find "$OUTPUT_DIR" -name "pipeline_opt_*.csv" -type f | sort | tail -1)
  if [ -n "$EXISTING_CSV" ] && [ -f "$EXISTING_CSV" ]; then
    echo "发现现有结果文件: $EXISTING_CSV"
    echo "将跳过已测试的参数组合，进行增量测试"
    RESULTS_CSV="$EXISTING_CSV"
    RESULTS_LOG="${EXISTING_CSV%.csv}.log"
  else
    RESULTS_CSV="$OUTPUT_DIR/pipeline_opt_${TS}.csv"
    RESULTS_LOG="$OUTPUT_DIR/pipeline_opt_${TS}.log"
    # CSV 表头（包含pipeline配置信息）
    echo "read_workers,tokenize_workers,write_workers,total_workers,batch_size,uptime_secs,overall_tokens_per_sec,overall_records_per_sec,overall_read_mb_per_sec,overall_convert_mb_per_sec,interval_tokens_per_sec,interval_records_per_sec,interval_read_mb_per_sec,interval_convert_mb_per_sec,tokens_total,records_total" > "$RESULTS_CSV"
  fi
else
  RESULTS_CSV="$OUTPUT_DIR/pipeline_opt_${TS}.csv"
  RESULTS_LOG="$OUTPUT_DIR/pipeline_opt_${TS}.log"
  # CSV 表头（包含pipeline配置信息）
  echo "read_workers,tokenize_workers,write_workers,total_workers,batch_size,uptime_secs,overall_tokens_per_sec,overall_records_per_sec,overall_read_mb_per_sec,overall_convert_mb_per_sec,interval_tokens_per_sec,interval_records_per_sec,interval_read_mb_per_sec,interval_convert_mb_per_sec,tokens_total,records_total" > "$RESULTS_CSV"
fi


# 日志文件头部信息（仅在新文件时写入）
if [ "$SKIP_EXISTING" = "true" ] && [ -f "$RESULTS_LOG" ]; then
  echo "" | tee -a "$RESULTS_LOG"
  echo "=== Pipeline优化增量测试继续 ===" | tee -a "$RESULTS_LOG"
  echo "继续时间: $(date)" | tee -a "$RESULTS_LOG"
  echo "READ_WORKERS: ${READ_WORKER_COUNTS[*]}" | tee -a "$RESULTS_LOG"
  echo "TOTAL_WORKERS: $TOTAL_WORKERS, BATCH_SIZE: $BATCH_SIZE, WRITE_WORKERS: $WRITE_WORKERS" | tee -a "$RESULTS_LOG"
  echo "" | tee -a "$RESULTS_LOG"
else
  echo "=== Pipeline优化实验：read/tokenize worker比例优化 ===" | tee "$RESULTS_LOG"
  echo "开始时间: $(date)" | tee -a "$RESULTS_LOG"
  echo "INPUT_DIR=$INPUT_DIR" | tee -a "$RESULTS_LOG"
  echo "TOKENIZER=$TOKENIZER" | tee -a "$RESULTS_LOG"
  echo "NO_WRITE=$NO_WRITE TIMEOUT_SEC=$TIMEOUT_SEC REPEAT=$REPEAT" | tee -a "$RESULTS_LOG"
  echo "固定配置: TOTAL_WORKERS=$TOTAL_WORKERS, BATCH_SIZE=$BATCH_SIZE, WRITE_WORKERS=$WRITE_WORKERS" | tee -a "$RESULTS_LOG"
  echo "READ_WORKERS测试范围: ${READ_WORKER_COUNTS[*]}" | tee -a "$RESULTS_LOG"
  echo "" | tee -a "$RESULTS_LOG"
fi

# 函数：检查参数组合是否已测试
is_already_tested() {
  local read_workers=$1
  if [ -f "$RESULTS_CSV" ] && [ "$SKIP_EXISTING" = "true" ]; then
    # 检查CSV文件中是否已存在该read_workers配置（排除ERROR行）
    grep -q "^$read_workers," "$RESULTS_CSV" 2>/dev/null
  else
    return 1  # 不跳过
  fi
}

for read_workers in "${READ_WORKER_COUNTS[@]}"; do
  for r in $(seq 1 "$REPEAT"); do
    # 计算tokenize workers数量
    tokenize_workers=$((TOTAL_WORKERS - read_workers - WRITE_WORKERS))
    
    # 检查是否已测试过该参数组合
    if is_already_tested "$read_workers"; then
      echo "=== 跳过已测试 read_workers=$read_workers tokenize_workers=$tokenize_workers (run#$r) ===" | tee -a "$RESULTS_LOG"
      continue
    fi
    
    echo "=== 测试 read_workers=$read_workers tokenize_workers=$tokenize_workers write_workers=$WRITE_WORKERS (run#$r) ===" | tee -a "$RESULTS_LOG"
    output_prefix="$OUTPUT_DIR/pipeline_r${read_workers}_t${tokenize_workers}_w${WRITE_WORKERS}_r${r}_$TS"
    start_time=$(date +%s)

    cmd=("$BINARY" \
      --input-dir "$INPUT_DIR" \
      --pattern "$PATTERN" \
      --text-cols content \
      --tokenizer "$TOKENIZER" \
      --output-prefix "$output_prefix" \
      --batch-size "$BATCH_SIZE" \
      --workers "$TOTAL_WORKERS" \
      --read-workers "$read_workers" \
      --tokenize-workers "$tokenize_workers" \
      --write-workers "$WRITE_WORKERS" \
      --metrics-interval 10)
    if [ "$NO_WRITE" = "true" ]; then
      cmd+=(--no-write)
    fi

    log_file="$OUTPUT_DIR/run_r${read_workers}_t${tokenize_workers}_w${WRITE_WORKERS}_r${r}.log"
      # 不再用 tee 管道，避免 timeout 信号只送到管道首进程导致残留子进程
      # 前台运行会导致 Ctrl-C 直接打断脚本而无法清理；
      # 这里将 timeout 放到后台并用 trap 转发信号，实现优雅退出
      set +e
      timeout --foreground --signal=INT --kill-after=5s "$TIMEOUT_SEC" "${cmd[@]}" &> "$log_file" &
      run_pid=$!
      on_int() {
        echo "  捕获 SIGINT，正在优雅停止子进程(pid=$run_pid) ..." | tee -a "$RESULTS_LOG"
        kill -INT "$run_pid" 2>/dev/null || true
        sleep 1
        kill -TERM "$run_pid" 2>/dev/null || true
      }
      on_term() {
        echo "  捕获 SIGTERM，正在优雅停止子进程(pid=$run_pid) ..." | tee -a "$RESULTS_LOG"
        kill -TERM "$run_pid" 2>/dev/null || true
        sleep 1
        kill -KILL "$run_pid" 2>/dev/null || true
      }
      trap on_int INT
      trap on_term TERM
      wait "$run_pid"
      run_rc=$?
      trap - INT TERM
      set -e
      if [ $run_rc -ne 0 ]; then
        echo "  运行超时/失败: read_workers=$read_workers tokenize_workers=$tokenize_workers run#$r (rc=$run_rc)" | tee -a "$RESULTS_LOG"
      fi

      end_time=$(date +%s)
      wall_time=$((end_time - start_time))

      # 简短展示末尾日志便于人工核验
      tail -n 5 "$log_file" | sed 's/^/    /' | tee -a "$RESULTS_LOG"

      # 解析配置行（读取/分词/写入 worker 配比）- 验证实际配置
      cfg_line=$(grep "Multi-stage pipeline configuration" "$log_file" | head -1 || true)
      actual_read_workers=$(echo "$cfg_line" | grep -o 'read_workers=[0-9]*' | cut -d'=' -f2)
      actual_tokenize_workers=$(echo "$cfg_line" | grep -o 'tokenize_workers=[0-9]*' | cut -d'=' -f2)
      actual_write_workers=$(echo "$cfg_line" | grep -o 'write_workers=[0-9]*' | cut -d'=' -f2)

      # 解析最后一条 metrics snapshot（稳定阶段）
      metrics_line=$(grep -F 'component="metrics"' "$log_file" | tail -1 || true)
      # 若本地日志仍未命中，回退到总日志（stdout 已经打印过）
      if [ -z "${metrics_line:-}" ] && [ -f "$RESULTS_LOG" ]; then
        metrics_line=$(grep -F 'component="metrics"' "$RESULTS_LOG" | tail -1 || true)
      fi
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

        echo "${actual_read_workers:-$read_workers},${actual_tokenize_workers:-$tokenize_workers},${actual_write_workers:-$WRITE_WORKERS},$TOTAL_WORKERS,$BATCH_SIZE,${uptime_secs:-},${overall_tokens_per_sec:-},${overall_records_per_sec:-},${overall_read_mb_per_sec:-},${overall_convert_mb_per_sec:-},${interval_tokens_per_sec:-},${interval_records_per_sec:-},${interval_read_mb_per_sec:-},${interval_convert_mb_per_sec:-},${tokens_total:-},${records_total:-}" >> "$RESULTS_CSV"

        echo "  稳定(uptime=${uptime_secs:-?}s): tokens/s=${overall_tokens_per_sec:-?}, interval_tokens/s=${interval_tokens_per_sec:-?}, convert_MB/s=${overall_convert_mb_per_sec:-?}, 墙钟=${wall_time}s" | tee -a "$RESULTS_LOG"
      else
        echo "${actual_read_workers:-$read_workers},${actual_tokenize_workers:-$tokenize_workers},${actual_write_workers:-$WRITE_WORKERS},$TOTAL_WORKERS,$BATCH_SIZE,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS_CSV"
        echo "  错误: 未找到 metrics snapshot 行 (可能超时)" | tee -a "$RESULTS_LOG"
      fi

      # 清理
      rm -f "${output_prefix}.bin" "${output_prefix}.idx" 2>/dev/null || true
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
df = df[df['uptime_secs'] != 'ERROR']  # 过滤错误行
df = df.astype({'read_workers': int, 'tokenize_workers': int, 'write_workers': int,
                'total_workers': int, 'batch_size': int, 'uptime_secs': float,
                'overall_tokens_per_sec': float, 'overall_records_per_sec': float,
                'overall_read_mb_per_sec': float, 'overall_convert_mb_per_sec': float})

# 创建图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 1. Read Workers vs overall tokens/s
ax1.plot(df['read_workers'], df['overall_tokens_per_sec'], 'o-', color='blue', linewidth=2, markersize=8)
ax1.set_xlabel('Read Workers')
ax1.set_ylabel('Overall Tokens/s')
ax1.set_title('Read Workers vs Tokens/s (固定128总workers, 8192 batch_size)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max(df['read_workers']) + 2)

# 2. Tokenize Workers vs overall tokens/s
ax2.plot(df['tokenize_workers'], df['overall_tokens_per_sec'], 'o-', color='green', linewidth=2, markersize=8)
ax2.set_xlabel('Tokenize Workers')
ax2.set_ylabel('Overall Tokens/s')
ax2.set_title('Tokenize Workers vs Tokens/s')
ax2.grid(True, alpha=0.3)

# 3. Read/Tokenize比例 vs Convert Speed
ratio = df['read_workers'] / df['tokenize_workers']
ax3.plot(ratio, df['overall_convert_mb_per_sec'], 'o-', color='red', linewidth=2, markersize=8)
ax3.set_xlabel('Read/Tokenize Workers 比例')
ax3.set_ylabel('Convert Speed (MB/s)')
ax3.set_title('Worker比例 vs Convert Speed')
ax3.grid(True, alpha=0.3)

# 4. Read Workers vs interval性能对比
ax4.plot(df['read_workers'], df['overall_tokens_per_sec'], 'o-', label='Overall', linewidth=2, markersize=8)
ax4.plot(df['read_workers'], df['interval_tokens_per_sec'], 's-', label='Interval', linewidth=2, markersize=6)
ax4.set_xlabel('Read Workers')
ax4.set_ylabel('Tokens/s')
ax4.set_title('Overall vs Interval Performance')
ax4.grid(True, alpha=0.3)
ax4.legend()

plt.tight_layout()
output_file = csv_file.replace('.csv', '_plot.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"图表已保存到: {output_file}")

# 显示最优配置（按 tokens/s 最大）
best_idx = df['overall_tokens_per_sec'].idxmax()
best_read = df.loc[best_idx, 'read_workers']
best_tokenize = df.loc[best_idx, 'tokenize_workers']
best_tps = df.loc[best_idx, 'overall_tokens_per_sec']
print(f"\n最优Pipeline配置: read_workers={best_read}, tokenize_workers={best_tokenize}, tokens/s={best_tps}")
print(f"最优比例: read/tokenize = {best_read}/{best_tokenize} = {best_read/best_tokenize:.3f}")
EOF

    echo "" | tee -a "$RESULTS_LOG"
    echo "生成图表脚本: $OUTPUT_DIR/plot_results.py" | tee -a "$RESULTS_LOG"
    echo "运行命令: python3 $OUTPUT_DIR/plot_results.py $RESULTS_CSV" | tee -a "$RESULTS_LOG"
fi

echo ""
echo "测试完成！结果保存在:"
echo "  CSV: $RESULTS_CSV"
echo "  LOG: $RESULTS_LOG"

