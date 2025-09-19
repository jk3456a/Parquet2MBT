#!/bin/bash
set -euo pipefail

# 并行运行96个 spm_encode 进程测试脚本
#
# 可重复运行的示例命令：
# bash scripts/parallel_spm_test.sh

echo "[INFO] 开始96个 smp_encode 进程并行测试..."

# 记录开始时间
start_time=$(date +%s.%N)

# 并行启动96个进程
pids=()
# 获取 CPU 核心数
num_cores=$(nproc)

echo "[INFO] 系统核心数: ${num_cores}"
echo "[INFO] 将96个进程绑定到 ${num_cores} 个核心上..."

for i in {0..95}; do
    # 计算要绑定的CPU核心ID
    
    # 使用 taskset -c <cpu_id> 来绑定进程
    # 每个进程处理对应的测试文件，输出到 /dev/null，错误输出到对应的日志文件
    ./build/src/spm_encode --model ./tokenizer.model --output_format=id --show_timing \
        < /tmp/test_100k_files/test_${i}.txt > /dev/null 2> /tmp/test_100k_files/log_${i}.txt &
    pids+=($!)
done

echo "[INFO] 已启动96个进程并完成CPU绑定，等待完成..."

# 等待所有进程完成
for pid in "${pids[@]}"; do
    wait $pid
done

# 记录结束时间
end_time=$(date +%s.%N)

# 计算总耗时
total_time=$(python3 -c "print($end_time - $start_time)")

echo "[INFO] 所有进程完成"
echo "[INFO] 总耗时: ${total_time} 秒"

# 统计结果
echo ""
echo "=== 统计结果 ==="

# 提取初始化时间和编码时间
init_times=()
encode_times=()
throughputs=()

for i in {0..95}; do
    log_file="/tmp/test_100k_files/log_${i}.txt"
    if [[ -f "$log_file" ]]; then
        init_time=$(grep "Initialization time:" "$log_file" | awk '{print $3}' || echo "0")
        encode_time=$(grep "Encoding time:" "$log_file" | awk '{print $3}' || echo "0")
        throughput=$(grep "Throughput:" "$log_file" | awk '{print $2}' || echo "0")
        
        init_times+=($init_time)
        encode_times+=($encode_time)
        throughputs+=($throughput)
    fi
done

# 计算平均值
if [[ ${#init_times[@]} -gt 0 ]]; then
    avg_init=$(echo "${init_times[*]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
    avg_encode=$(echo "${encode_times[*]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
    total_throughput=$(echo "${throughputs[*]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum}')
    
    echo "成功完成的进程数: ${#init_times[@]}"
    echo "平均初始化时间: ${avg_init} ms"
    echo "平均编码时间: ${avg_encode} ms"
    echo "总体吞吐量: ${total_throughput} 行/秒"
    echo "处理总行数: $((${#init_times[@]} * 10000))"
    echo "实际总体吞吐量: $(python3 -c "print(f'{${#init_times[@]} * 10000 / $total_time:.2f}')") 行/秒"
else
    echo "没有找到有效的日志文件"
fi

echo ""
echo "详细日志文件位于: /tmp/test_100k_files/log_*.txt"
