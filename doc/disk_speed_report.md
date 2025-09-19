## 磁盘 I/O 性能简报（顺序读写）

- **结论**：顺序写入约 3.9 GB/s；顺序读取约 1.1 GB/s。
- **测试时间**：2025-09-18

### 测试结果

| 指标 | 吞吐 | 耗时 | 样本大小 | 块大小 | 并发 |
| --- | --- | --- | --- | --- | --- |
| 顺序写入（O_DIRECT） | 3.9 GB/s | 0.273 s | 1 GiB | 1 MiB | 单线程 |
| 顺序读取（O_DIRECT） | 1.1 GB/s | 0.994 s | 1 GiB | 1 MiB | 单线程 |

### 环境与硬件

- **设备**：Intel SSDPF2KX153T1（NVMe SSD），容量约 14 TB（挂载分区 12 TB）
- **挂载**：`/cache`（ext4，选项：rw,relatime）
- **I/O 调度器**：none（多队列），nr_requests=1023，read_ahead_kb=128
- **逻辑/物理块**：512 B / 512 B；**文件系统块**：4 KiB
- **内核**：Linux 5.15.0-94-generic
- **磁盘使用率**：约 99%（可用约 220 GiB）

### 测试方法（可复现）

- 工具：`dd`
- 写入命令：
  ```bash
  dd if=/dev/zero of=/cache/lizhen/repos/DataPlat/Sstable/Parquet2MBT/disk_speed_test.tmp bs=1M count=1024 oflag=direct
  ```
- 读取命令：
  ```bash
  dd if=/cache/lizhen/repos/DataPlat/Sstable/Parquet2MBT/disk_speed_test.tmp of=/dev/null bs=1M iflag=direct
  ```
- 说明：O_DIRECT 绕过页缓存，更贴近裸顺序 I/O 能力；测试后临时文件已删除。

### 说明与后续

- 以上为单线程顺序读写结果；实际业务还会受随机访问、并发度、压缩/解压、网络等因素影响。
- 如需更全面的评估，可采用 `fio` 在不同块大小（4 KiB ~ 1 MiB）、队列深度（Q1 ~ Q64）以及随机/混合读写下生成吞吐与 P95/P99 延迟曲线。


