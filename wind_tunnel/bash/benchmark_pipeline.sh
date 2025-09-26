#!/bin/bash

# Write Worker优化实验脚本：锁定Reader=4，测试不同Write Worker数量的性能
set -euo pipefail

# 默认配置（可通过环境变量或命令行覆盖）
INPUT_DIR="${INPUT_DIR:-/home/lizhen/dataset/zh__CCI4.0-M2-Base-v1-newest_zh_cc-high-loss0__2025091500}"
PATTERN="${PATTERN:-*.parquet}"
TOKENIZER="${TOKENIZER:-../testdata/tokenizer/tokenizer.json}"
OUTPUT_DIR="${OUTPUT_DIR:-../testdata/benchmark/write_worker_optimization}"
BINARY="${BINARY:-../target/release/parquet2mbt}"
NO_WRITE="${NO_WRITE:-false}"  # 启用写入以测试Write Worker性能
TIMEOUT_SEC="${TIMEOUT_SEC:-120}"
REPEAT="${REPEAT:-1}"

# 固定最佳Reader配置
FIXED_READ_WORKERS=4
BATCH_SIZE=8192

# Write Worker数量测试范围
WRITE_WORKER_COUNTS=(${WRITE_WORKER_COUNTS:-1 2 3 4 5 6 7 8})

# 获取CPU核心数
TOTAL_CPU_CORES=$(nproc)

# 增量测试支持：检查已完成的测试
SKIP_EXISTING="${SKIP_EXISTING:-true}"

mkdir -p "$OUTPUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)

# 增量测试：检查是否存在现有结果文件
EXISTING_CSV=""
if [ "$SKIP_EXISTING" = "true" ]; then
  # 查找最新的write worker优化结果文件
  EXISTING_CSV=$(find "$OUTPUT_DIR" -name "write_worker_opt_*.csv" -type f | sort | tail -1)
  if [ -n "$EXISTING_CSV" ] && [ -f "$EXISTING_CSV" ]; then
    echo "发现现有结果文件: $EXISTING_CSV"
    echo "将跳过已测试的参数组合，进行增量测试"
    RESULTS_CSV="$EXISTING_CSV"
    RESULTS_LOG="${EXISTING_CSV%.csv}.log"
  else
    RESULTS_CSV="$OUTPUT_DIR/write_worker_opt_${TS}.csv"
    RESULTS_LOG="$OUTPUT_DIR/write_worker_opt_${TS}.log"
    # CSV 表头（包含pipeline配置信息）
    echo "read_workers,tokenize_workers,write_workers,total_workers,batch_size,uptime_secs,overall_tokens_per_sec,overall_records_per_sec,overall_read_mb_per_sec,overall_convert_mb_per_sec,interval_tokens_per_sec,interval_records_per_sec,interval_read_mb_per_sec,interval_convert_mb_per_sec,tokens_total,records_total" > "$RESULTS_CSV"
  fi
else
  RESULTS_CSV="$OUTPUT_DIR/write_worker_opt_${TS}.csv"
  RESULTS_LOG="$OUTPUT_DIR/write_worker_opt_${TS}.log"
  # CSV 表头（包含pipeline配置信息）
  echo "read_workers,tokenize_workers,write_workers,total_workers,batch_size,uptime_secs,overall_tokens_per_sec,overall_records_per_sec,overall_read_mb_per_sec,overall_convert_mb_per_sec,interval_tokens_per_sec,interval_records_per_sec,interval_read_mb_per_sec,interval_convert_mb_per_sec,tokens_total,records_total" > "$RESULTS_CSV"
fi


# 日志文件头部信息（仅在新文件时写入）
if [ "$SKIP_EXISTING" = "true" ] && [ -f "$RESULTS_LOG" ]; then
  echo "" | tee -a "$RESULTS_LOG"
  echo "=== Write Worker优化增量测试继续 ===" | tee -a "$RESULTS_LOG"
  echo "继续时间: $(date)" | tee -a "$RESULTS_LOG"
  echo "WRITE_WORKERS: ${WRITE_WORKER_COUNTS[*]}" | tee -a "$RESULTS_LOG"
  echo "固定配置: READ_WORKERS=$FIXED_READ_WORKERS, BATCH_SIZE=$BATCH_SIZE, CPU_CORES=$TOTAL_CPU_CORES" | tee -a "$RESULTS_LOG"
  echo "" | tee -a "$RESULTS_LOG"
else
  echo "=== Write Worker优化实验：锁定Reader=4，测试Write Worker最佳数量 ===" | tee "$RESULTS_LOG"
  echo "开始时间: $(date)" | tee -a "$RESULTS_LOG"
  echo "INPUT_DIR=$INPUT_DIR" | tee -a "$RESULTS_LOG"
  echo "TOKENIZER=$TOKENIZER" | tee -a "$RESULTS_LOG"
  echo "NO_WRITE=$NO_WRITE TIMEOUT_SEC=$TIMEOUT_SEC REPEAT=$REPEAT" | tee -a "$RESULTS_LOG"
  echo "固定配置: READ_WORKERS=$FIXED_READ_WORKERS, BATCH_SIZE=$BATCH_SIZE, CPU_CORES=$TOTAL_CPU_CORES" | tee -a "$RESULTS_LOG"
  echo "WRITE_WORKERS测试范围: ${WRITE_WORKER_COUNTS[*]}" | tee -a "$RESULTS_LOG"
  echo "Worker分配逻辑: tokenize_workers = CPU_CORES - read_workers - write_workers" | tee -a "$RESULTS_LOG"
  echo "" | tee -a "$RESULTS_LOG"
fi

# 函数：检查参数组合是否已测试
is_already_tested() {
  local write_workers=$1
  if [ -f "$RESULTS_CSV" ] && [ "$SKIP_EXISTING" = "true" ]; then
    # 检查CSV文件中是否已存在该write_workers配置（排除ERROR行）
    grep -q "^$FIXED_READ_WORKERS,[0-9]*,$write_workers," "$RESULTS_CSV" 2>/dev/null
  else
    return 1  # 不跳过
  fi
}

for write_workers in "${WRITE_WORKER_COUNTS[@]}"; do
  for r in $(seq 1 "$REPEAT"); do
    # 计算tokenize workers数量：总CPU核心数 - 固定read workers - 当前write workers
    tokenize_workers=$((TOTAL_CPU_CORES - FIXED_READ_WORKERS - write_workers))
    total_workers=$((FIXED_READ_WORKERS + tokenize_workers + write_workers))
    
    # 检查tokenize_workers是否合理（至少1个）
    if [ $tokenize_workers -lt 1 ]; then
      echo "=== 跳过 write_workers=$write_workers (tokenize_workers=$tokenize_workers < 1) ===" | tee -a "$RESULTS_LOG"
      continue
    fi
    
    # 检查是否已测试过该参数组合
    if is_already_tested "$write_workers"; then
      echo "=== 跳过已测试 write_workers=$write_workers tokenize_workers=$tokenize_workers (run#$r) ===" | tee -a "$RESULTS_LOG"
      continue
    fi
    
    echo "=== 测试 read_workers=$FIXED_READ_WORKERS tokenize_workers=$tokenize_workers write_workers=$write_workers (总workers=$total_workers) (run#$r) ===" | tee -a "$RESULTS_LOG"
    output_prefix="$OUTPUT_DIR/write_opt_r${FIXED_READ_WORKERS}_t${tokenize_workers}_w${write_workers}_r${r}_$TS"
    start_time=$(date +%s)

    cmd=("$BINARY" \
      --input-dir "$INPUT_DIR" \
      --pattern "$PATTERN" \
      --text-cols content \
      --tokenizer "$TOKENIZER" \
      --output-prefix "$output_prefix" \
      --batch-size "$BATCH_SIZE" \
      --workers "$total_workers" \
      --read-workers "$FIXED_READ_WORKERS" \
      --tokenize-workers "$tokenize_workers" \
      --write-workers "$write_workers" \
      --metrics-interval 10)
    if [ "$NO_WRITE" = "true" ]; then
      cmd+=(--no-write)
    fi

    log_file="$OUTPUT_DIR/run_r${FIXED_READ_WORKERS}_t${tokenize_workers}_w${write_workers}_r${r}.log"
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
        echo "  运行超时/失败: write_workers=$write_workers tokenize_workers=$tokenize_workers run#$r (rc=$run_rc)" | tee -a "$RESULTS_LOG"
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

        echo "${actual_read_workers:-$FIXED_READ_WORKERS},${actual_tokenize_workers:-$tokenize_workers},${actual_write_workers:-$write_workers},$total_workers,$BATCH_SIZE,${uptime_secs:-},${overall_tokens_per_sec:-},${overall_records_per_sec:-},${overall_read_mb_per_sec:-},${overall_convert_mb_per_sec:-},${interval_tokens_per_sec:-},${interval_records_per_sec:-},${interval_read_mb_per_sec:-},${interval_convert_mb_per_sec:-},${tokens_total:-},${records_total:-}" >> "$RESULTS_CSV"

        echo "  稳定(uptime=${uptime_secs:-?}s): tokens/s=${overall_tokens_per_sec:-?}, interval_tokens/s=${interval_tokens_per_sec:-?}, convert_MB/s=${overall_convert_mb_per_sec:-?}, 墙钟=${wall_time}s" | tee -a "$RESULTS_LOG"
      else
        echo "${actual_read_workers:-$FIXED_READ_WORKERS},${actual_tokenize_workers:-$tokenize_workers},${actual_write_workers:-$write_workers},$total_workers,$BATCH_SIZE,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS_CSV"
        echo "  错误: 未找到 metrics snapshot 行 (可能超时)" | tee -a "$RESULTS_LOG"
      fi

      # 清理（分片产物）
      rm -f "${output_prefix}".shard_*.bin "${output_prefix}".shard_*.idx 2>/dev/null || true
  done
done

echo "=== 测试完成 ===" | tee -a "$RESULTS_LOG"
echo "结束时间: $(date)" | tee -a "$RESULTS_LOG"
echo "结果文件: $RESULTS_CSV" | tee -a "$RESULTS_LOG"
echo "日志文件: $RESULTS_LOG" | tee -a "$RESULTS_LOG"

# 生成Write Worker优化图表脚本
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
                'overall_read_mb_per_sec': float, 'overall_convert_mb_per_sec': float,
                'interval_tokens_per_sec': float, 'interval_records_per_sec': float,
                'interval_read_mb_per_sec': float, 'interval_convert_mb_per_sec': float})

# 创建图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# 1. Write Workers vs Overall Tokens/s
ax1.plot(df['write_workers'], df['overall_tokens_per_sec'], 'o-', color='blue', linewidth=2, markersize=8)
ax1.set_xlabel('Write Workers')
ax1.set_ylabel('Overall Tokens/s')
ax1.set_title('Write Workers vs Tokens/s (Fixed: Read=4, Dynamic Tokenize)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, max(df['write_workers']) + 1)

# 2. Write Workers vs Tokenize Workers (显示资源分配)
ax2.plot(df['write_workers'], df['tokenize_workers'], 'o-', color='green', linewidth=2, markersize=8)
ax2.set_xlabel('Write Workers')
ax2.set_ylabel('Tokenize Workers')
ax2.set_title('Write Workers vs Available Tokenize Workers')
ax2.grid(True, alpha=0.3)

# 3. Write Workers vs Convert Speed
ax3.plot(df['write_workers'], df['overall_convert_mb_per_sec'], 'o-', color='red', linewidth=2, markersize=8)
ax3.set_xlabel('Write Workers')
ax3.set_ylabel('Convert Speed (MB/s)')
ax3.set_title('Write Workers vs Convert Speed')
ax3.grid(True, alpha=0.3)

# 4. Write Workers vs Write Efficiency (Tokens/s per Write Worker)
write_efficiency = df['overall_tokens_per_sec'] / df['write_workers']
ax4.plot(df['write_workers'], write_efficiency, 'o-', color='purple', linewidth=2, markersize=8)
ax4.set_xlabel('Write Workers')
ax4.set_ylabel('Write Efficiency (Tokens/s per Write Worker)')
ax4.set_title('Write Worker Efficiency')
ax4.grid(True, alpha=0.3)

# 标注最优点
best_idx = df['overall_tokens_per_sec'].idxmax()
best_write = df.loc[best_idx, 'write_workers']
best_tps = df.loc[best_idx, 'overall_tokens_per_sec']
ax1.annotate(f'Peak: {best_write}W, {best_tps:.0f}T/s', 
             xy=(best_write, best_tps), xytext=(best_write+0.5, best_tps*0.95),
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
             fontsize=10, color='red', weight='bold')

plt.tight_layout()
output_file = csv_file.replace('.csv', '_plot.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Write Worker optimization plot saved to: {output_file}")

# 显示最优配置和分析
best_tokenize = df.loc[best_idx, 'tokenize_workers']
best_total = df.loc[best_idx, 'total_workers']
best_convert = df.loc[best_idx, 'overall_convert_mb_per_sec']
best_efficiency = write_efficiency.iloc[best_idx]

print(f"\n=== Write Worker Optimization Results ===")
print(f"Optimal Configuration:")
print(f"  Read Workers: 4 (fixed)")
print(f"  Write Workers: {best_write}")
print(f"  Tokenize Workers: {best_tokenize}")
print(f"  Total Workers: {best_total}")
print(f"  Performance: {best_tps:.0f} tokens/s")
print(f"  Convert Speed: {best_convert:.2f} MB/s")
print(f"  Write Efficiency: {best_efficiency:.0f} tokens/s per write worker")

# 分析Write Worker扩展性
print(f"\n=== Write Worker Scaling Analysis ===")
for i, row in df.iterrows():
    eff = write_efficiency.iloc[i]
    print(f"  {int(row['write_workers'])}W: {row['overall_tokens_per_sec']:.0f}T/s, {eff:.0f}T/s/W, {int(row['tokenize_workers'])}T")
EOF

    echo "" | tee -a "$RESULTS_LOG"
    echo "生成图表脚本: $OUTPUT_DIR/plot_results.py" | tee -a "$RESULTS_LOG"
    echo "运行命令: python3 $OUTPUT_DIR/plot_results.py $RESULTS_CSV" | tee -a "$RESULTS_LOG"
fi

echo ""
echo "测试完成！结果保存在:"
echo "  CSV: $RESULTS_CSV"
echo "  LOG: $RESULTS_LOG"

