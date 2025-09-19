# Parquet2MBT 产品需求文档（PRD）

## 1. 背景与目标

将指定目录下的 Parquet 数据集批量转换为 Megatron 训练可直接加载的二进制索引格式（简称 MBT：Megatron Binary Tokenized dataset，输出 `.bin/.idx`）。

- **使用场景**：LLM/自回归模型预训练、继续训练、微调等离线数据准备。
- **实时性定义（本项目阶段）**：单次命令对指定目录内全部 Parquet 文件完成转换，吞吐接近本机顺序磁盘读写能力（I/O 上限）。
- **兼容性**：输出 `.bin/.idx` 必须与 Megatron-LM 的 `IndexedDataset` 加载器兼容（以 Megatron-LM 当前源码为准）。

## 2. 范围（Scope）

输入与输出：
- **输入**：
  - 本地目录（递归扫描）。
  - 支持 `--pattern` 过滤（如 `*.parquet`）。
  - 支持读取 Parquet 指定列（如 `--text-cols title,content`）。
  - 支持行组/批次流式读取，避免整表载入内存。
- **处理**：
  - 反序列化（Parquet→Arrow）。
  - 提取关键字段、轻量预处理（去首尾空白、可配置连接符）。
  - 批量分词（HuggingFace Tokenizers Rust 实现），支持 `encode_batch`。
  - 文档边界策略（按行/按文件）。
- **输出**：
  - `--output-prefix` 对应的 `prefix.bin` 与 `prefix.idx`。
  - 原子落盘（先写临时文件再重命名）。
  - 可选输出一个 `prefix.meta.json`（运行统计与配置快照）。

非目标（Out of Scope，本期不做）：
- 集群/调度框架（Ray/Spark/Kafka 等）。
- 复杂数据血缘、审计链路。
- 分布式断点续传（本期仅单机断点能力）。

## 3. 非功能与指标（SLO）

- **性能**：
  - 吞吐接近顺序磁盘读上限，目标≥80%（受压缩解码与分词 CPU 影响）。
  - Tokenization 为主要 CPU 瓶颈；充分利用多核（并行度≈CPU核数-2）。
- **内存**：
  - 流式处理，峰值内存可配置，默认<1–2 GiB（与批大小相关）。
- **可靠性**：
  - 单条/单文件失败不影响整体；失败样本计数与日志。
  - 幂等与断点：已完成输出不重复写入；支持 `--resume` 跳过已完成文件。
- **可观测性**：
  - 关键指标：读写吞吐（MB/s、tokens/s）、队列深度、批处理时延、错误率、CPU 使用率。
  - 结构化日志（JSON 行）与进度条（可关闭）。

## 4. 总体架构

单机多线程流水线，Reader→Tokenizer Pool→Writer，通过有界通道形成背压，确保写入顺序与原子性。

```
+--------------+      +-------------------+      +--------------+
|   Scanner    | ---> |  Reader (Arrow)   | ---> |  Tokenizers  |
|  (walk dir)  |      |  rowgroup batches |  \   |  Pool (N)    |
+--------------+      +-------------------+   \->+--------------+
                                            \           |
                                             \----------v-------> Writer (.bin/.idx)
```

关键设计：
- 读：按 Row Group/RecordBatch 拉取，仅投影需要的列，降低反序列化与内存压力。
- 分词：批量 `encode_batch`，每个工作线程独立 `Tokenizer` 实例，避免锁竞争。
- 写：单线程顺序写 `.bin`，并维护文档边界信息；结束生成 `.idx`。
- 顺序：需要文件级顺序时携带任务序号在 Writer 端重排；默认可无序以最大化吞吐。

### 分阶段路线图（Pipeline 演进）

- Stage1：单机单一 pipeline
  - 单进程内多线程流水线（Reader→Tokenizers→Writer）。
  - 目标：打通功能与稳定性，吞吐≥80% 顺序磁盘读。
- Stage2：单机多进程 pipeline
  - Supervisor 进程 + 多个 Worker 进程，每个 Worker 独立完整流水线。
  - 进程间可通过文件队列/命名管道/本地消息队列协调（可选）。
  - 目标：隔离故障、CPU 亲和与 NUMA 利用，更佳可运维性与可重启性。
- Stage3：容器化与 Pod 级 pipeline
  - 提供 Docker 镜像与 K8s 部署清单；每个 Pod 运行 1..N 条流水线。
  - 配置 via 环境变量/ConfigMap/CLI；持久化卷挂载输入/输出；暴露 Prometheus 指标与健康检查。
  - 目标：水平扩展与运维标准化。

## 5. 数据模型与预处理

- 列选择：`--text-cols a,b,c`；未指定时尝试按启发式匹配（`text`/`content`/`doc`）。
- 多列拼接：使用 `--concat-sep`（默认 `"\n"`）进行连接。
- 文档边界：
  - `row`：每行/记录为一篇文档（默认）。
  - `file`：每个文件拼接为一篇文档。
- 过滤：`--min-chars`、`--max-chars`、`--min-tokens`、`--max-tokens`（过滤或切分）。
- 特殊符号：`--bos-id`、`--eos-id`、`--add-special-tokens`（默认不额外添加）。

## 6. Tokenizer

- 使用 HuggingFace `tokenizers`（Rust 实现）。
- 加载方式：
  - 本地 `--tokenizer path.json|.model`；
  - 或 `--tokenizer hf://repo_name`（可选）。
- 批处理：`--batch-size` 控制 `encode_batch` 的条数（建议 16–64K，视文本平均长度与内存而定）。
- 并行（两种模式，二选一或按需组合）：
  - 内部并行（tokenizers 内建）：启用 `tokenizers` crate 的 `parallel` feature（内部使用 `rayon`），通过 `RAYON_NUM_THREADS` 控制线程数；`encode_batch` 在库内并行。
  - 外层并行（进程外线程池）：本工具维护 `N` 个分词 worker 线程（`--workers`），每线程持有独立 `Tokenizer` 实例，批次切分为 `N` 份并行调用 `encode_batch`，Writer 端合并顺序。
  - 选择策略：
    - `auto`（推荐缺省）：优先外层并行，禁用内部并行，避免“双重并行”争抢 CPU；在低核数或内存紧张场景可切到内部并行。
    - `internal`：仅启用内部并行；
    - `external`：仅启用外层分词线程池；
    - `both`：同时启用（一般不推荐，仅用于实验）。
  - 相关开关（见 CLI）：`--tok-parallel <auto|internal|external|both>`、`--workers`、环境变量 `RAYON_NUM_THREADS`。
  - 批次切分：`--batch-shards <INT>` 将 `encode_batch` 输入均分为若干 shard，喂给多个 worker；`auto` 时等于 `min(workers, ceil(batch_size/target_docs_per_shard))`。
  - 预热：启动时为每个 worker 做一次小批次预热，降低首次延迟。
- dtype：根据词表大小和 `--dtype`（`u16|u32|auto`）决定写入 `.bin` 的元素类型。

### 特殊 Token 处理与校验

- 目标：确保训练/推理端所依赖的特殊 token（如 `<|im_start|>`、`<|im_end|>`、`<|tool_call|>`、`<|execute_start|>`、`<|execute_end|>`、`<|fim_prefix|>`、`<|fim_middle|>`、`<|fim_suffix|>`，以及 BOS/EOS）在所加载的 `tokenizer.json|.model` 中存在且 ID 与期望一致；不一致时给出明确策略与可观测告警。
- 配置与开关：
  - `--add-special-tokens`：是否在编码时自动添加 BOS/EOS（默认关闭）。
  - `--bos-id`、`--eos-id`：当 `--add-special-tokens` 开启时用于显式指定 ID；未指定则按 tokenizer 默认。
  - `--special-tokens-json <PATH>`：JSON 文件，形如 `{ "<|im_start|>": 73441, "<|im_end|>": 73440, ... }`，用于声明“期望的特殊 token → 期望的 ID”。
  - `--strict-special-ids`：严格模式；若发现声明的 token 缺失或 ID 不匹配，直接报错退出（配合 `--fail-fast`）。默认关闭时仅告警并继续。
- 校验流程（启动时一次性）：
  1) 载入 tokenizer；
  2) 若提供 `--special-tokens-json`，遍历其中每个 token：
     - 若 `token_to_id(token)` 返回 `None`：
       - 严格模式：报错退出（提示修复方式：更新 `tokenizer.json` 的 `added_tokens` 或使用正确的 HF repo）。
       - 非严格模式：记录 `WARN`，并在 `meta.json` 写入实际缺失列表；继续执行。
     - 若返回的 `id != 期望 id`：
       - 严格模式：报错退出。
       - 非严格模式：记录 `WARN`，并在 `meta.json` 写入不一致映射（`expected_id`/`actual_id`）。
- 说明：HF `tokenizers` 运行期 `add_special_tokens` 会为新增 token 重新分配 ID，通常无法在运行时“指定固定 ID”；若需要完全对齐固定 ID，应从源头修正 `tokenizer.json`/词表（或使用发布方提供的原版 tokenizer）。
- 元数据：在可选输出的 `prefix.meta.json` 中写入 `special_tokens_check` 段，包含 `missing`、`mismatched`、`strict`、`tokenizer_path` 等，便于审计。

## 7. 输出格式（Megatron `.bin/.idx`）

- `.bin`：紧凑存放全部 token ids（小端，`u16/u32`）；无分隔符。
- `.idx`：文档边界与元信息索引文件。
  - 要求与 Megatron-LM `IndexedDatasetBuilder` 生成的 `.idx` 完全一致（版本、dtype、偏移/长度数组结构、字节序等以 Megatron-LM 当前源码为准）。
  - 实现上记录每篇文档 token 数，构建累积偏移（或等价结构），并写入必要 header/metadata。
- 兼容性验收：使用 Megatron-LM 的 `IndexedDataset` 成功加载并正确遍历所有样本。

## 8. CLI 设计

必选/常用：
- `--input-dir <PATH>`：输入目录（递归）。
- `--pattern <GLOB>`：文件匹配（默认 `*.parquet`）。
- `--output-prefix <PATH>`：输出前缀（生成 `<prefix>.bin/.idx`）。
- `--text-cols <COLS>`：逗号分隔的文本列名（如 `title,content`）。

可选：
- `--recursive`（默认开启）。
- `--doc-boundary <row|file>`（默认 `row`）。
- `--concat-sep <STR>`（默认换行）。
- `--tokenizer <PATH|HF_REPO>`（必需其一）。
- `--batch-size <INT>`、`--workers <INT>`、`--queue-cap <INT>`。
- `--dtype <u16|u32|auto>`（默认 `auto`）。
- `--min-chars`、`--max-chars`、`--min-tokens`、`--max-tokens`。
- `--bos-id`、`--eos-id`、`--add-special-tokens`。
- `--keep-order`：保持文件级处理顺序（牺牲吞吐）。
- `--resume`：跳过已完成文件。
- `--fail-fast`：遇错即停；默认记录并继续。
- `--tmp-dir <PATH>`：临时文件目录。
- `--log-level <trace|debug|info|warn|error>`，`--progress`（默认开）。
 - `--metrics-exporter <none|prometheus|stdout>`（默认 `stdout`）。
 - `--metrics-interval <SECONDS>`（默认 5）。
 - `--special-tokens-json <PATH>`、`--strict-special-ids`（见“特殊 Token 处理与校验”）。
 - 并行相关：
   - `--tok-parallel <auto|internal|external|both>`（默认 `auto`）。
   - `--workers <INT>`：外层分词线程池大小（`external/auto` 模式有效）。
   - `--batch-shards <INT>`：一次批次对分词池的切分份数（默认 `auto`）。
   - `--rowgroup-parallel <INT>`（可选）：同一文件按 RowGroup 解码并行度（默认 1；旋转盘不建议开太大）。
   - 环境：`RAYON_NUM_THREADS`（控制内部并行线程数）。

## 9. 工程结构（Rust）

```
Sstable/Parquet2MBT/
  ├── doc/
  │   └── PRD.md
  ├── Cargo.toml                # 依赖：anyhow, clap, serde, serde_json, tracing, tokio(可选), arrow, parquet, tokenizers
  └── src/
      ├── main.rs               # 入口：解析 CLI，装配 Pipeline
      ├── cli/mod.rs            # clap 参数定义与校验
      ├── config/mod.rs         # 配置结构体、serde 解析
      ├── scanner/mod.rs        # 目录扫描与过滤
      ├── reader/mod.rs         # Parquet→Arrow 流式读取（投影列、batch 拉取）
      ├── preprocessor/mod.rs   # 文本清洗、拼接、过滤
      ├── tokenizer/mod.rs      # HF tokenizers 封装，encode_batch
      ├── writer/mod.rs         # .bin 追加写与 .idx 构建（原子 rename）
      ├── index/mod.rs          # 索引结构与序列化写入（与 Megatron 对齐）
      └── pipeline/mod.rs       # 通道、背压与调度（crossbeam/rayon）
```

依赖建议：
- `anyhow`, `thiserror`（错误处理）
- `clap`（CLI）
- `serde`, `serde_json`（配置与元数据）
- `tracing`, `tracing-subscriber`（日志/指标）
- `rayon`, `crossbeam-channel`（并行与背压）
- `arrow`, `parquet`（读取）
- `tokenizers`（分词）

## 10. 并发与背压

### 分层并行策略

- Reader 并行（同文件 RowGroup）：
  - 读取文件元数据后，将 RowGroup 建模为解码任务；并发度由 `--rowgroup-parallel` 控制。
  - 每个任务仅解码投影列（ProjectionMask），降低反序列化与内存。
  - 风险：多点随机读对 HDD 不友好；SSD 建议 2–4 观察吞吐再调。

- Tokenizer 并行（二选一）：
  - 外层分词线程池（推荐）：
    - 结构：`reader → (work-queue) → tokenizer workers (N) → merge → writer`
    - 每 worker 独立 `Tokenizer` 实例，避免锁竞争；`--workers` 控制并发。
    - 批次切分：`--batch-shards` 控制将一个 RecordBatch 的文本划分为多个 shard 以提高并行度与负载均衡；超长文可单独成 shard。
    - 顺序：默认 Writer 端按“文件序 + 批内序 + shard 序”重排复原（`--keep-order`）；关闭保持顺序可最大化吞吐。
  - 内部并行（HF tokenizers `parallel` 特性）：
    - 通过 `RAYON_NUM_THREADS` 控内并行线程数。
    - 与外层并行互斥（`--tok-parallel internal`），避免双层争抢。

- Writer：单线程顺序写 `.bin`，避免磁盘随机写；记录 `sequence_lengths` 构建 `.idx`。

### 背压与内存

- 通道：`reader → tokenizer → writer` 均使用有界队列（`--queue-cap`）；达到上限时 Reader/Tokenizer 阶段阻塞，防止内存膨胀。
- 内存估算：
  - 文本缓冲 ≈ `batch_size * avg_chars_per_doc`；
  - Token 缓冲 ≈ `batch_size * avg_tokens_per_doc * (2|4)`；
  - 建议容量：`queue-cap * max(文本缓冲, token缓冲)` 控制在可接受范围内。

### 长尾与切分

- 超长文（极端长）会拖慢批次：开启 `--batch-shards` 或启用“长文单独成 shard”，必要时配合 `--max-tokens` 截断。
- 可选 `--truncate`：对训练无损场景，直接截长保速。

### 线程与亲和性（可选）

- 进程内：`--workers` ≈ `cpu_cores-2`，避免与 Reader/Writer/系统抢核。
- 进程间（Stage2）：每进程 1 条完整流水线，配合 taskset/NUMA 亲和；通过多进程水平扩展。

### 观测与调参

- 指标：增加按阶段累计时间与占比（reader/preprocess/tokenize/write/index），并在结束时打印 summary（平均带宽、tokens 总数、各阶段占比）。
- 常见调参顺序：增大 `--workers` → 调整 `--batch-size`/`--batch-shards` → 视磁盘类型设置 `--rowgroup-parallel`。

## 11. 可靠性与幂等

- 文件粒度状态：开始、进行中、完成、失败；`--resume` 时跳过已完成。
- 单条记录失败：计数与日志，不中断整体。
- 原子写入：`*.bin.tmp`/`*.idx.tmp` 完成后 `rename` 覆盖。

## 12. 监控与运维（Monitoring & Operations）

### 日志（Logging）

- 形式：结构化 JSON 行日志（默认到 stdout，可定向到文件），配合 `jq`/ELK 收集。
- 关键字段（示例）：
  - `ts`、`level`、`component`（`scanner|reader|tokenizer|writer|pipeline`）、`event`
  - `file_path`、`row_group`、`batch_size`、`bytes_read`、`records`、`tokens`
  - `latency_ms`（批处理耗时）、`queue_depth_reader`、`queue_depth_tok`
  - `cpu_pct`、`mem_rss_bytes`、`disk_read_MBps`、`disk_write_MBps`（若启用资源采样）
  - `error`（错误摘要）、`error_count`
- 进度条：默认启用（可 `--progress` 关闭），显示累计 MB、records、tokens 与 ETA。

### 指标（Metrics）

- 导出方式：
  - `stdout`：每 `--metrics-interval` 秒打印一行简要指标（便于离线运行查看）。
  - `prometheus`：内置 HTTP 端口暴露 `/metrics`（Stage3 容器化重点）。
- 指标项（建议命名）：
  - 吞吐：`parquet2mbt_bytes_read_total`、`parquet2mbt_tokens_total`、`parquet2mbt_records_total`
  - 速率：`parquet2mbt_read_mb_per_sec`、`parquet2mbt_tokens_per_sec`
  - 延迟：`parquet2mbt_batch_latency_ms`（Histogram：p50/p90/p99）
  - 队列：`parquet2mbt_queue_depth_reader`、`parquet2mbt_queue_depth_tokenizer`
  - 资源：`parquet2mbt_cpu_pct`、`parquet2mbt_mem_rss_bytes`、`parquet2mbt_disk_read_mb_per_sec`
  - 错误：`parquet2mbt_error_total{type=...}`、`parquet2mbt_failed_records_total`
- 采样频率：默认 5s，可配置 `--metrics-interval`。

### 健康检查与运维

- 健康：
  - 启动自检：配置合法性、tokenizer 可加载、输出目录可写、特殊 token 校验。
  - 运行健康：队列阻塞检测（超过阈值持续 X 秒告警）、连续错误阈值告警。
- 断点续转与幂等：已完成文件记录到状态存储；`--resume` 时跳过；状态与统计写入 `prefix.meta.json`。
- 资源采样：可选启用（Linux `procfs`/`sysinfo`），采集 CPU/内存/磁盘 I/O，写入日志与/或指标。
- 报警对接：预留 Prometheus 告警与日志关键字匹配（如 `WARN special_token_mismatch`、`ERROR batch_decode_failed`）。

## 13. 风险与缓解

- Tokenizer 线程安全与复制开销：为每线程构建独立实例，避免共享锁；冷启动预热。
- Parquet 压缩解码（snappy/zstd）成为瓶颈：可增大批大小或提升 workers 数；必要时启用更高压缩级硬件/库。
- `.idx` 细节兼容风险：实现严格对齐 Megatron-LM 当前 `IndexedDatasetBuilder`；上线前以其加载器做端到端验证。

## 14. 里程碑与验收

- 阶段映射：
  - Stage1：M1–M3
  - Stage2：M4
  - Stage3：M5

- M0：PRD 落地（本次）。
- M1：最小可用路径（单文件→`.bin/.idx`），CLI 子集。
- M2：目录与并行处理、断点续、指标日志。
- M3：性能冲刺（≥80% 磁盘顺序读吞吐），大规模测试。
- M4：多进程 pipeline（Supervisor/Workers）、稳定性与边界完善，错误处理与回收策略。
- M5：容器化（Docker 镜像）与 K8s 清单，打包与版本发布（静态二进制）。

验收标准：
- 通过 Megatron-LM `IndexedDataset` 成功加载输出数据集；随机抽样校验 token 数与边界正确。
- 在代表性数据上达到目标吞吐与资源占用边界。

## 15. 示例命令

```
parquet2mbt \
  --input-dir /data/corpus \
  --pattern "*.parquet" \
  --text-cols title,content \
  --tokenizer ./tokenizer.json \
  --output-prefix /data/out/corpus \
  --batch-size 32768 \
  --workers 32 \
  --dtype auto \
  --doc-boundary row \
  --resume
```

## 16. 未来扩展

- 文件系统适配：S3/HDFS/OSS/GCS。
- 分布式执行（多机并发，集中式去重与指标聚合）。
- 更丰富的预处理算子：正则清洗、HTML 去标、语言检测、去重等（保持模块化可插拔）。


