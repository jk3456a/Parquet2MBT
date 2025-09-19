#!/bin/bash

# Worker 并行度性能测试脚本
# 测试不同 workers 数量对 tokenization 性能的影响

set -e

# 配置参数
INPUT_DIR="/cache/lizhen/repos/DataPlat/Sstable/Parquet2MBT/testdata/data"
TOKENIZER="/cache/lizhen/repos/DataPlat/Sstable/Parquet2MBT/testdata/tokenizer/tokenizer.json"
OUTPUT_DIR="/cache/lizhen/repos/DataPlat/Sstable/Parquet2MBT/testdata/benchmark"
BINARY="./target/release/parquet2mbt"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 结果文件
RESULTS_CSV="$OUTPUT_DIR/workers_benchmark_$(date +%Y%m%d_%H%M%S).csv"
RESULTS_LOG="$OUTPUT_DIR/workers_benchmark_$(date +%Y%m%d_%H%M%S).log"

# CSV 表头
echo "workers,rayon_threads,elapsed_secs,tokenize_ms_total,tokenize_pct,tokens_per_sec,read_mb_per_sec,convert_mb_per_sec,total_tokens,total_records" > "$RESULTS_CSV"

echo "=== Worker 并行度性能测试 ===" | tee "$RESULTS_LOG"
echo "开始时间: $(date)" | tee -a "$RESULTS_LOG"
echo "测试配置:" | tee -a "$RESULTS_LOG"
echo "  INPUT_DIR: $INPUT_DIR" | tee -a "$RESULTS_LOG"
echo "  TOKENIZER: $TOKENIZER" | tee -a "$RESULTS_LOG"
echo "  OUTPUT_DIR: $OUTPUT_DIR" | tee -a "$RESULTS_LOG"
echo "" | tee -a "$RESULTS_LOG"

# 测试不同的 worker 数量
WORKER_COUNTS=(4 8 12 16 20 24 28 32 40 48 56 64)

for workers in "${WORKER_COUNTS[@]}"; do
    echo "=== 测试 workers=$workers, RAYON_NUM_THREADS=$workers ===" | tee -a "$RESULTS_LOG"
    
    # 输出文件前缀
    output_prefix="$OUTPUT_DIR/test_w${workers}_$(date +%s)"
    
    # 运行测试
    start_time=$(date +%s)
    
    # 设置环境变量并运行
    RAYON_NUM_THREADS=$workers timeout 300s "$BINARY" \
        --input-dir "$INPUT_DIR" \
        --pattern "*.parquet" \
        --text-cols content \
        --tokenizer "$TOKENIZER" \
        --output-prefix "$output_prefix" \
        --batch-size 32768 \
        --workers $workers \
        --metrics-interval 0 \
        --no-write 2>&1 | tee "$OUTPUT_DIR/run_w${workers}.log"
    
    end_time=$(date +%s)
    wall_time=$((end_time - start_time))
    
    # 从输出中提取关键指标
    log_file="$OUTPUT_DIR/run_w${workers}.log"
    
    if [ -f "$log_file" ]; then
        # 提取 summary 行的指标
        summary_line=$(grep "run summary" "$log_file" | tail -1)
        
        if [ -n "$summary_line" ]; then
            # 使用 grep 和 sed 提取各个指标
            elapsed_secs=$(echo "$summary_line" | grep -o 'elapsed_secs=[0-9.]*' | cut -d'=' -f2)
            tokenize_ms=$(echo "$summary_line" | grep -o 'tokenize_ms_total="[0-9.]*"' | cut -d'"' -f2)
            tokenize_pct=$(echo "$summary_line" | grep -o 'tokenize_pct="[0-9.]*"' | cut -d'"' -f2)
            tokens_per_sec=$(echo "$summary_line" | grep -o 'convert_avg_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
            read_mb_per_sec=$(echo "$summary_line" | grep -o 'read_avg_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
            convert_mb_per_sec=$(echo "$summary_line" | grep -o 'convert_avg_mb_per_sec="[0-9.]*"' | cut -d'"' -f2)
            total_tokens=$(echo "$summary_line" | grep -o 'tokens_total=[0-9]*' | cut -d'=' -f2)
            total_records=$(echo "$summary_line" | grep -o 'records_total=[0-9]*' | cut -d'=' -f2)
            
            # 写入 CSV
            echo "$workers,$workers,$elapsed_secs,$tokenize_ms,$tokenize_pct,$tokens_per_sec,$read_mb_per_sec,$convert_mb_per_sec,$total_tokens,$total_records" >> "$RESULTS_CSV"
            
            # 写入日志
            echo "  结果: elapsed=${elapsed_secs}s, tokenize_ms=${tokenize_ms}, tokenize_pct=${tokenize_pct}%" | tee -a "$RESULTS_LOG"
            echo "  性能: tokens/s=${tokens_per_sec}, convert_mb/s=${convert_mb_per_sec}" | tee -a "$RESULTS_LOG"
        else
            echo "  错误: 未找到 summary 行" | tee -a "$RESULTS_LOG"
            echo "$workers,$workers,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS_CSV"
        fi
    else
        echo "  错误: 日志文件不存在" | tee -a "$RESULTS_LOG"
        echo "$workers,$workers,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR,ERROR" >> "$RESULTS_CSV"
    fi
    
    echo "  墙钟时间: ${wall_time}s" | tee -a "$RESULTS_LOG"
    echo "" | tee -a "$RESULTS_LOG"
    
    # 清理输出文件（如果使用了 --no-write 就不会有文件）
    rm -f "${output_prefix}.bin" "${output_prefix}.idx" 2>/dev/null || true
    
    # 短暂休息避免系统过热
    sleep 2
done

echo "=== 测试完成 ===" | tee -a "$RESULTS_LOG"
echo "结束时间: $(date)" | tee -a "$RESULTS_LOG"
echo "结果文件: $RESULTS_CSV" | tee -a "$RESULTS_LOG"
echo "日志文件: $RESULTS_LOG" | tee -a "$RESULTS_LOG"

# 生成简单的性能报告
echo "" | tee -a "$RESULTS_LOG"
echo "=== 性能摘要 ===" | tee -a "$RESULTS_LOG"
echo "按 tokenize 时间排序 (越小越好):" | tee -a "$RESULTS_LOG"

# 跳过表头，按 tokenize_ms_total 排序
tail -n +2 "$RESULTS_CSV" | grep -v ERROR | sort -t',' -k4 -n | head -5 | while IFS=',' read -r workers rayon elapsed tokenize_ms tokenize_pct tokens_per_sec read_mb convert_mb total_tokens total_records; do
    echo "  workers=$workers: tokenize=${tokenize_ms}ms (${tokenize_pct}%), convert=${convert_mb}MB/s" | tee -a "$RESULTS_LOG"
done

echo "" | tee -a "$RESULTS_LOG"
echo "详细结果请查看: $RESULTS_CSV" | tee -a "$RESULTS_LOG"

# 如果有 python，生成简单的图表脚本
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
df = df.astype({'workers': int, 'elapsed_secs': float, 'tokenize_ms_total': float, 'tokenize_pct': float})

# 创建图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# 1. Workers vs Tokenize Time
ax1.plot(df['workers'], df['tokenize_ms_total'], 'bo-')
ax1.set_xlabel('Workers')
ax1.set_ylabel('Tokenize Time (ms)')
ax1.set_title('Workers vs Tokenization Time')
ax1.grid(True)

# 2. Workers vs Total Time
ax2.plot(df['workers'], df['elapsed_secs'], 'ro-')
ax2.set_xlabel('Workers')
ax2.set_ylabel('Total Time (seconds)')
ax2.set_title('Workers vs Total Elapsed Time')
ax2.grid(True)

# 3. Workers vs Convert Speed
ax3.plot(df['workers'], df['convert_mb_per_sec'], 'go-')
ax3.set_xlabel('Workers')
ax3.set_ylabel('Convert Speed (MB/s)')
ax3.set_title('Workers vs Convert Speed')
ax3.grid(True)

# 4. Workers vs Tokenize Percentage
ax4.plot(df['workers'], df['tokenize_pct'], 'mo-')
ax4.set_xlabel('Workers')
ax4.set_ylabel('Tokenize Percentage (%)')
ax4.set_title('Workers vs Tokenize Time Percentage')
ax4.grid(True)

plt.tight_layout()
output_file = csv_file.replace('.csv', '_plot.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"图表已保存到: {output_file}")

# 显示最优配置
best_idx = df['tokenize_ms_total'].idxmin()
best_workers = df.loc[best_idx, 'workers']
best_time = df.loc[best_idx, 'tokenize_ms_total']
print(f"\n最优配置: workers={best_workers}, tokenize_time={best_time}ms")
EOF

    echo "" | tee -a "$RESULTS_LOG"
    echo "生成图表脚本: $OUTPUT_DIR/plot_results.py" | tee -a "$RESULTS_LOG"
    echo "运行命令: python3 $OUTPUT_DIR/plot_results.py $RESULTS_CSV" | tee -a "$RESULTS_LOG"
fi

echo ""
echo "测试完成！结果保存在:"
echo "  CSV: $RESULTS_CSV"
echo "  LOG: $RESULTS_LOG"

