# CALO 当前模拟器实现说明

本文档只描述当前仓库中已经落地的实现，不引用论文中的理想化表述。

## 0. 这份文档回答什么

这份文档回答三个问题：

1. 当前 simulator 代码到底实现了什么。
2. 当前 simulator 的目标边界是什么。
3. 当前 simulator 离“完全真实”还差什么。

先把边界说清楚：

- 当前目标是做一个**由真实 OpenWhisk profile 校准的 research-grade simulator**。
- 当前目标不是做云厂商内部细节一比一复刻的 digital twin。
- 当前目标也不是把 PPO 训练直接改成全程真实云在线训练。

所以这里的“把假的变成真的”，更准确地说是：

- 把 warm runtime、cold overhead、burst slowdown、TTL 这些关键量，尽量从真实平台采出来；
- 再让 simulator 以这些真实校准表为主，而不是以手写公式为主；
- 最后在这个基础上再做 CALO 的训练和对比实验。

## 0.5 当前阶段判断

如果只问一句“现在这个 simulator 开发到哪一步了”，截至 `2026-04-08`，最准确的判断是：

- 它已经不是启发式 toy simulator。
- 它已经是一个**以真实 OpenWhisk profile 为主驱动的 calibrated simulator**。
- 它还不是完整真实部署闭环，也不是公有云级 digital twin。

当前可以把状态拆成三层来理解。

### 0.5.1 已完成

- 训练环境结构已经升级为批量窗口仿真，而不是单步假 latency 采样。
- `79` 维状态接口已经稳定；`ActionSpace` 现支持 `full_48` 与 `calibrated_x64_8` 两种 preset，当前主实验配置默认使用后者。
- CodeBERT + PCA 特征链真实存在。
- `x64` 路径下，`stage2_prime` 已收口为完整的 full24 聚合目录：
  - `warm_profile.csv`：120 行
  - `burst_profile.csv`：360 行
  - `cold_profile.csv`：362 行
- `arm64` 路径下，`stage1` 也已收口为完整的 full24 聚合目录：
  - `warm_profile.csv`：120 行
  - `burst_profile.csv`：360 行
  - `cold_profile.csv`：232 行
  - 但这套目录当前主要覆盖 cold feasibility 与 burst slowdown，并未对每个组合额外补正 `idle-gap > 0` 的 TTL 搜索网格；因此 `estimate_ttl_sec()` 对 `arm64` 上缺少正 idle-gap 证据的组合会保守回退到默认 `600s`
- 其中 `512MB/x64/300s`、`1024MB/x64/300s`、`2048MB/x64/300s` 与 `3008MB/x64/300s` 的 5 个 CALO benchmark 都已完成真实 idle-gap TTL 标定，当前统一估计为 `900s`。
- 另外，`x64/timeout=120s` 的 5 个 CALO benchmark 不仅已补齐真实 `warm + enforced cold + burst` 标定，还已通过自适应 idle-gap 搜索把 5 个 benchmark 在 `512/1024/2048/3008MB` 上的 TTL 全部推进到真实推断的 `900s`。

### 0.5.2 进行中

- `x64` 与 `arm64` 的真实 profile 已全部落盘；下一阶段主要是把这些 profile 回灌到 surrogate、simulator 对照和论文证据中。
- 断点续采逻辑已经打通，`--keep-existing-output` 会跳过已成功 invocation，只补跑缺失点。
- 主实验配置已经稳定接到 `stage2_prime`，因此后续 TTL 扩展会直接增强现有 simulator，而不是另起一套新环境。

### 0.5.3 尚未完成

- 真实执行路径仍缺少“动作下发到 OpenWhisk并把真实指标写回 RL 环境”的闭环。

所以，当前最合理的结论不是“它已经完全真实”，而是：

- **相对以前：** 它已经从“假环境”跨到了“有真实校准证据支撑的环境”。
- **相对顶会系统论文要求：** 它还缺更完整的 TTL / timeout / arm64 矩阵，以及真实在线闭环。

相关代码入口：

- `rl_optimizer/environment.py`
- `rl_optimizer/dynamic_environment.py`
- `rl_optimizer/sim_clock.py`
- `rl_optimizer/container_pool.py`
- `rl_optimizer/service_model.py`
- `rl_optimizer/workload_sources.py`
- `rl_optimizer/load_monitor.py`
- `rl_optimizer/state_space.py`
- `rl_optimizer/action_space.py`
- `rl_optimizer/llm_analyzer.py`

## 1. 总体结构

当前 CALO 的训练环境已经不再是“单步采样一个假 latency”的旧写法，而是一个三层结构：

1. `ServerlessFunctionEnv`
   - 负责动作解释、配置更新、奖励计算、状态更新。
   - 每个 `step()` 执行的是一个时间窗内的一批请求，而不是单个请求。
2. `DynamicLoadEnvironment`
   - 负责给基础环境提供逐步变化的 workload window。
   - 当前支持 `synthetic` 和 `azure_trace` 两种 workload source。
3. 仿真后端组件
   - `SimulationClock`：维护模拟时间。
   - `ContainerPool`：维护 warm container、TTL 驱逐和排队。
   - `CalibratedServiceModel`：采样 warm runtime 和 cold overhead。

因此，当前实现已经从“完全手写启发式 Gym 环境”升级到了“批量窗口 + 容器池 + 可校准服务时间模型”的结构。但它仍然不是完整的生产级数字孪生。

如果你只想快速验证“当前运行的到底是不是这套真实校准 simulator”，建议优先看两件事：

1. `python-venv/bin/python run_dynamic_experiment.py --config config/rl_experiments/full_suite.json --smoke-test --smoke-steps 2`
   - 输出中应出现 `calibration=openwhisk_calibration_x64_stage2_prime`
   - 输出中应出现 `ttl_512_x64_300=900s`
2. `results/openwhisk_calibration_x64_stage2_prime/`
   - `warm_profile.csv=120`、`burst_profile.csv=360`、`cold_profile.csv=362`
3. `results/openwhisk_calibration_arm64_stage1/`
   - `warm_profile.csv=120`、`burst_profile.csv=360`、`cold_profile.csv=232`

## 2. 单步执行流程

当前单步执行发生在 `ServerlessFunctionEnv.step()`，流程如下：

1. 接收一个离散动作 `action`。
2. 通过 `ActionSpace` 将动作映射为 `(memory, architecture, timeout)`。
3. 将配置写入 `StateSpace.config_context`。
4. 读取 `DynamicLoadEnvironment` 预先设置的 `WorkloadStep`。
5. 对该时间窗内的所有到达请求执行批量仿真：
   - 由 `CalibratedServiceModel` 采样 warm runtime。
   - 由 `CalibratedServiceModel` 采样 cold overhead。
   - 由 `ContainerPool` 决定请求命中 warm container、触发 cold start，还是进入排队。
6. 汇总该时间窗的聚合指标：
   - mean / p50 / p95 / p99 latency
   - mean / total cost
   - success rate
   - cold-start rate
   - timeout rate
   - queue ratio
   - peak concurrency
7. 用这批事件更新 `LoadMonitor`、`PerformanceHistory` 和 `ConfigurationContext`。
8. 计算 reward，提取新的 79 维状态。

所以，当前实现确实已经显式模拟了“请求到达 -> 容器分配 -> 完成/超时”的批量路径。

## 3. 状态空间

当前状态空间在 `StateSpace` 中定义，总维度仍然是 79，分解如下：

- 32 维代码特征
- 10 维负载特征
- 5 维函数类别 one-hot
- 5 维历史性能特征
- 27 维配置上下文特征

### 3.1 代码特征

代码特征来自 `CodeBERTAnalyzer`：

1. 使用 `microsoft/codebert-base` 提取 768 维 `[CLS]` embedding。
2. 在 benchmark 目录的 Python 源文件及其滑动窗口代码片段上拟合真实 PCA，再降到 32 维。
3. 对 PCA 输出做 L2 normalization。

当前一次实测中：

- PCA 语料样本数为 48
- 累计解释方差约为 0.9973

因此：

- “用了 CodeBERT” 这件事是真的。
- “做了 PCA 降维” 现在在代码里已经成立。

需要注意的是：

- PCA 是基于当前仓库中的 benchmark 语料拟合的，不是基于大规模外部代码语料单独训练的降维器。
- 如果缺少已保存的 PCA 状态，而外部调用又绕过了 `BenchmarkFeatureExtractor` 的两阶段流程，代码仍可能退回到简单截断路径。因此论文和实验应以实际 benchmark 提取路径为准。

### 3.2 负载特征

`LoadMonitor.extract_features()` 输出 10 维特征，包括：

- 当前 QPS
- QPS 趋势
- burst 指标
- 近期峰值并发
- 冷启动概率估计
- 平均响应时间
- 小时特征
- 星期特征
- 负载波动性
- 容器 warmth

这里和旧版本不同的一点是：`LoadMonitor` 现在可以由模拟时钟驱动，而不是强制依赖 `time.time()`。当前 `ServerlessFunctionEnv` 会在批量事件回放时把模拟时间写进去。

## 4. 动作空间

动作空间现在已经不是“只能固定 48 个动作”的硬编码实现，而是同一个 `ActionSpace` 类支持多个 preset：

- `full_48`
  - 内存：`128/256/512/1024/2048/3008`
  - 架构：`x64/arm64`
  - 超时：`60/120/300/900`
  - 总计 `6 x 2 x 4 = 48`
- `calibrated_x64_8`
  - 内存：`512/1024/2048/3008`
  - 架构：`x64`
  - 超时：`120/300`
  - 总计 `4 x 1 x 2 = 8`

当前 `config/rl_experiments/full_suite.json`、`config/rl_experiments/tuning_uploader_recognition.json` 和 `config/rl_experiments/azure_trace_tuned.json` 都显式设置了 `action_space.preset=calibrated_x64_8`。这意味着当前主实验 headline 仍默认不跨到 `128/256MB`、`60/900s` 和 `arm64` 这些 widened 组合。

不过从 `2026-04-08` 起，训练入口与 `CalibratedServiceModel` 已支持 `calibration_dirs={x64: ..., arm64: ...}` 的多目录加载。也就是说，`full_48` 配置现在已经可以在同一次 simulator 运行中同时读取 `results/openwhisk_calibration_x64_stage2_prime/` 与 `results/openwhisk_calibration_arm64_stage1/`，而不是被单个 `calibration_dir` 限制在某一侧架构上。

因此，更准确的说法应当是：

- `79-D state` 这件事在代码里仍然成立；
- `48-action space` 仍然作为可选全空间存在；
- `full_48` widened action-space 现在已经可以消费双架构真实 profile；
- 但当前主实验 headline 仍运行在“真实校准证据最强”的 `8-action calibrated subset` 上。

## 5. 当前性能模拟是怎么做的

当前默认不会调用真实云平台，而是走 `_execute_simulated_batch()`。

### 5.1 workload window

每个 step 先拿到一个 `WorkloadStep`，其中包含：

- `arrival_count`
- `step_duration_sec`
- `source_name`
- `load_value`
- `minute_of_day`
- `mean_invocations_per_minute`
- `std_invocations_per_minute`
- `max_invocations_per_minute`
- `burstiness_hint`

随后，环境不会再把这批请求均匀铺到整个时间窗，而是按 burstiness-aware 的方式在窗内采样到达时间：

- 平滑 workload 会近似均匀随机分布
- bursty workload 会先采样 bin 权重，再把请求聚簇到少量子区间
- burstiness 由 `mean/std/max` 统计量和 source 提供的 `burstiness_hint` 联合估计

### 5.2 warm runtime

warm runtime 来自 `CalibratedServiceModel.sample_warm_runtime_ms()`。

如果存在校准文件 `warm_profile.csv`：

- 从最接近的 `(benchmark, memory, timeout)` 记录中读取统计量。
- 再从拟合的 log-normal 分布里采样。
- 如果同时存在 `burst_profile.csv`，还会根据窗口到达率估算 expected concurrency，并对 warm runtime 叠加并发退化倍率。

如果不存在校准文件：

- 回退到启发式基线：
  - benchmark 前缀 -> 手工 `base_latency`
  - `memory_factor = sqrt(128 / memory_mb)`
  - `arch_factor = 0.95` for `arm64`
  - 再乘随机噪声

所以当前 warm runtime 已经是“优先查校准表，再按 burst 压力修正，否则退回启发式公式”。

### 5.3 cold overhead

cold overhead 来自 `CalibratedServiceModel.sample_cold_overhead_ms()`。

如果存在 `cold_profile.csv`：

- 按 `(benchmark, memory, timeout, idle_gap)` 找最近记录。
- 用记录中的 `estimated_cold_overhead_ms` 和 `estimated_p95_cold_overhead_ms` 采样。

如果不存在：

- 回退到 `Uniform(500, 1500) ms`。

### 5.4 container pool

`ContainerPool` 负责：

- warm container TTL 驱逐
- idle container 复用
- 容量不足时排队
- 新建 container 时标记 cold start

当前规则是：

1. 请求到达前先驱逐 idle time 超过 TTL 的 container。
2. 若存在空闲 warm container，则直接复用。
3. 若没有空闲 container 且池子未满，则创建新 container，记为 cold start。
4. 若池子已满，则请求排队到最早可用的 container。

这意味着当前冷启动已经不再是“直接按概率伯努利采样”，而是和 TTL、并发、池子容量绑定。

### 5.5 burst pressure

如果存在 `burst_profile.csv`：

- `CalibratedServiceModel` 会先用 warm mean latency 和当前 arrival rate 估算 expected concurrency。
- 再从最接近的 `(benchmark, memory, timeout, concurrency)` 记录中读取 slowdown。
- 用 slowdown 的 mean / p95 关系生成额外的 runtime 放大因子。

如果不存在 `burst_profile.csv`：

- 会退回到一个较保守的启发式退化项。
- 退化幅度与 expected concurrency 成正相关，但不会无限放大。

这一步的作用是把“高并发下函数本身会变慢”这件事显式建模出来，而不仅仅依赖队列等待时间。

### 5.6 timeout / success / cost

- timeout：若 latency 超过当前 timeout，就按 timeout 截断。
- success rate：由 timeout flag 的补集得到，不再有旧版固定 2% 的随机失败率。
- cost：按 Lambda 风格 GB-second 计费近似计算，x64 与 arm64 单价不同，并加上固定请求成本。

## 6. 动态负载是怎么做的

动态负载现在由 `DynamicLoadEnvironment` 驱动，不再直接手工往 `LoadMonitor` 塞假请求。

### 6.1 synthetic workload

`SyntheticWorkloadSource` 仍然支持四种模式：

- `sine`
- `spike`
- `decay`
- `random`

但输出已经变成 `WorkloadStep`，由基础环境真正执行一批请求。

对于 synthetic source：

- `load_change_freq` 控制 workload pattern 的刷新步长。
- 同一刷新区间内会复用同一个 synthetic 负载级别。

### 6.2 Azure trace workload

`AzureTraceWorkloadSource` 可以直接读取：

- `external_data/azure_functions/processed/azure_functions_2019_topk_minute_profiles.csv`
- `external_data/azure_functions/processed/azure_functions_2019_function_summary.csv`

它会：

1. 先根据 benchmark 画像在 Azure summary 上选取一个相对匹配的 function profile。
2. 按 `minute_of_day` 读取该分钟的 `mean_invocations/std/max_invocations`。
3. 根据 `step_duration_sec` 折算成本 step 的到达强度。
4. 同时把分钟级 mean/std/max 调用统计折算为 `WorkloadStep`，供 simulator 在窗口内生成更真实的到达簇。
4. 按 benchmark 的目标并发度自动计算缩放倍率。
5. 用 Poisson 采样该 step 的 `arrival_count`。

当前还额外支持两个降采样参数：

- `arrival_scale`
- `max_arrivals_per_step`

以及一组 profile 选择参数：

- `profile_selection`
- `selection_pool_size`
- `target_concurrency`

这是为了避免公共 trace 的热点函数在 15 秒窗口内直接产生几千次调用，同时也避免所有 benchmark 都随机重放同一种热点 profile。

因此，当前动态负载已经支持“真实公开 trace 驱动”，但默认训练配置仍然使用 synthetic source。

## 7. 冷启动模型

当前冷启动有两层含义，需要区分：

### 7.1 仿真真值

仿真里的 cold start 真值由 `ContainerPool` 决定：

- container 过期被驱逐
- 新请求到来时没有可复用 warm container
- 且池子还有扩容空间

这部分是真正参与 latency/cost 计算的 cold-start 事件。

### 7.2 状态特征中的冷启动概率

`LoadMonitor.get_cold_start_probability()` 仍然是一个统计估计量：

- 基于历史 cold/warm 比例
- 再结合 `last_request_time` 和 `container_ttl`

它只是状态特征，不是仿真器的底层判定逻辑。

这两者不能混为一谈。

## 8. 真实执行路径目前是什么状态

代码里仍然保留 `enable_real_execution=True` 的路径，但训练闭环依旧没有打通：

1. `_execute_real()` 会尝试调用 `sebs.py benchmark invoke ...`。
2. 但动作中的 memory、architecture、timeout 还没有真正下发到部署端。
3. 命令返回后的真实指标解析也没有完成。
4. 当前实现最终仍然会回退到 `_execute_simulated_batch()`。

因此，仓库里目前已经存在“真实 OpenWhisk 调用”和“真实校准采集”两条链路，但仍然不存在“真实部署执行 -> 回写真实指标 -> 直接用于训练/评估”的闭环。

需要区分三件事：

1. `sebs.py benchmark invoke ... --deployment openwhisk`
   - 这条真实部署链路现在已经在当前机器上跑通。
   - 已实测通过 `110.dynamic-html`、`120.uploader`、`130.crud-api`。
2. `scripts/collect_openwhisk_calibration.py`
   - 这条真实校准采集链路现在也已经从 preflight 推进到可产数。
   - 已实测跑通过 `warm` smoke，真实生成了 `calibration_invocations.csv`、`warm_profile.csv` 和 `metadata.json`。
   - 当前保留两套真实 `x64` 校准目录，以及一套已收口的 `arm64` 最小目录：
     - `results/openwhisk_calibration_x64_stage1_v2/`：历史累计基线目录，保留早期 warm/cold/burst 真实 profile；其中一部分高并发 burst 行是在未启用 priming 的旧方法下采集的，适合作为回退和对比基线。
     - `results/openwhisk_calibration_x64_stage2_prime/`：当前默认 `x64` 目录。`warm_profile.csv` 含 120 行，覆盖 5 个 CALO benchmark 在 `60/120/300/900s x 128/256/512/1024/2048/3008MB` 下的 full24 warm 聚合；`burst_profile.csv` 含 360 行，覆盖上述 full24 配置在并发 `1/2/4` 下的 burst 聚合；`cold_profile.csv` 含 362 行，包含 enforced cold 与非负 idle-gap 的混合聚合结果。低内存 `411.image-recognition` 的不可行组合现已以 `profile_source=prewarm_only` / `success_rate=0.0` 保留在 profile 中，而不是被静默丢弃。
     - `results/openwhisk_calibration_arm64_stage1/`：当前默认 `arm64` 目录。`warm_profile.csv` 含 120 行，`burst_profile.csv` 含 360 行，`cold_profile.csv` 含 232 行，同样对应 5 个 CALO benchmark 的 full24 聚合结果。基于这批数据，`411.image-recognition` 在 ARM64/OpenWhisk 上的可行性边界已经扩展为：`128/256MB` 在 cold 与 burst 路径上持续 failure-only，`512MB` 及以上在四个 timeout 下稳定成功；同时其余 4 个 benchmark 的 full24 warm/cold/burst 也已真实落盘。
   - `stage2_prime` 中 `timeout=120s` 的 `warm + enforced cold + burst` 已补齐 5 个 CALO benchmark 的完整矩阵；其中 `120.uploader` 的补采最终把 `2048MB@concurrency=2` 从 `9/10` 收敛到 `10/10` 成功样本。后续又通过 `config/calibration/openwhisk_calibration_x64_stage2_idle_gap_timeout120_adaptive.json` 对 `311.compression` 与 `411.image-recognition` 追加关键 `600/900s` 观测，使 5 个 benchmark 在 `x64/timeout=120s` 的四个 memory 档位当前都能真实推断为 `TTL=900s`。
   - OpenWhisk standalone 默认把 action memory 上限限制在 `512MB`。当前仓库已新增 `config/openwhisk_standalone_memory_1024.conf` 与 `config/openwhisk_standalone_memory_4096.conf`，通过官方 `--config-file` 覆盖入口把 `maxActionMemory` 真实放宽到了 `1024MB` 和 `4096MB`，并已分别用 `/api/v1/namespaces/_/limits` 与真实 `110.dynamic-html` probe 验证。
   - 为了让高内存配置切换真正可用，`sebs/openwhisk/openwhisk.py` 中的 `update_function_configuration()` 已修正为在更新 limits 时一并重发 `--docker` 和 code package，避免 `wsk action update` 在部分 benchmark 上报 `exec undefined`。
   - 为了让 profile 在单个 warm 点缺失时仍可用，`scripts/collect_openwhisk_calibration.py` 现已在聚合 `cold_profile.csv` 和 `burst_profile.csv` 时支持“最近 warm 配置回退”。这让例如 `210.thumbnailer@3008MB` 这类 warm 缺行点仍能得到可用的冷启动 overhead 估计，而不是直接落成 `NaN`。
   - 另外，burst 聚合已经优先使用 measured 中 `success=True && cold_start=False` 的 warm-only 样本来计算 slowdown，避免把 measured 轮次里混入的冷启动直接双重记入 simulator。
   - 针对 burst 采集方法本身，仓库已新增 `prime_concurrency_rounds`。在 `results/openwhisk_burst_probe_base/` 与 `results/openwhisk_burst_probe_prime/` 的对照 probe 中，相同 benchmark/memory/concurrency 条件下，measured burst 的 cold-start 比例从 `25%` 降到了 `0%`。`stage2_prime` 的正式 burst 表就是按这个带 prime 的方法重采出来的。
3. `ServerlessFunctionEnv(enable_real_execution=True)`
   - 这条 RL 训练时的真实执行路径仍未完成动作下发和结果回写闭环。
   - 当前依旧不能直接用来做真实系统训练。

## 9. 当前模拟器最大的局限

### 9.1 目前还缺“完整校准数据”

虽然现在代码已经支持：

- `warm_profile.csv`
- `cold_profile.csv`
- `burst_profile.csv`

并且当前仓库已经具备真实 OpenWhisk 校准采集能力。当前已确认的真实进展是：

- OpenWhisk 单机部署可用
- MinIO 路径可用
- Scylla/Alternator 路径可用
- `collect_openwhisk_calibration.py` 的 `warm` smoke 已真实产出 CSV

但这还不是完整校准矩阵。当前仍缺：

- `210.thumbnailer@3008MB` 这类高内存不稳定点的进一步复核
- 更多 timeout 档位的真实校准
- `arm64` 在 `411` 之外 benchmark 的 cold/burst 全矩阵

不过，主实验配置已经不再默认空跑：

- `config/rl_experiments/full_suite.json`
- `config/rl_experiments/tuning_uploader_recognition.json`
- `config/rl_experiments/azure_trace_tuned.json`

这三份配置现在都显式指向 `results/openwhisk_calibration_x64_stage2_prime`，因此训练和 smoke test 会优先使用当前真实 profile，而不是无条件退回启发式 warm/cold/burst 模型。

但要注意，`stage2_prime` 的 TTL 校准虽然已经把 `x64/timeout=300s` 的四个 memory tier 全部补齐，但仍不是所有 memory/timeout 组合都已经覆盖。当前默认目录已经不再是“只有 enforced cold、TTL 一律退回默认值”的状态，而是：

- `cold_profile.csv` 在 `x64` 默认目录下现有 362 行，对应 full24 组合上的 enforced cold 与非负 idle-gap 混合证据；在 `arm64` 默认目录下现有 232 行，对应同样的 full24 组合但分布更偏向于“够用的阈值识别”，而不是把每个组合都采满同一数量的正 idle-gap 点。
- 对 `x64` 来说，当前真实 TTL 证据已经不再只局限于 `300s` 与 `120s` 的局部矩阵，而是已经被扩展到这轮 full24 聚合目录；其中 `411.image-recognition@128/256MB` 的 failure-only rows 也能被 `CalibratedServiceModel` 直接消费为 exact infeasible hint。
- 对 `arm64` 来说，当前 full24 目录已经足以提供 exact feasibility 和 burst slowdown 约束，但 TTL 推断只会使用正 `idle_gap_sec` 的证据；如果某个组合只有 `idle_gap=0` 的冷启动记录，当前实现会保守回退到默认 TTL，而不会误判成 `TTL=0`。
- 对尚未精确标定的其它 memory/timeout 组合，当前实现会先走“同 benchmark 最近邻 TTL 回退”；若某组数据只有 warm gap、还没有首次 cold threshold，则阈值会被下界到 `max(default_ttl_sec, max_observed_gap + 1)`，避免把“目前观测到一直 warm”的 profile 误退回到更小的默认 TTL。
- 对 `results/openwhisk_calibration_arm64_stage1`，当前已经可以直接被 `CalibratedServiceModel` 读取并参与 exact feasibility 约束。已实测验证 `411.image-recognition` 在 `arm64` 下的 `warm` / `burst` exact hint 会稳定给出：`128/256MB -> is_feasible=False, profile_source=warmup_only/prewarm_only`，`512MB+ -> is_feasible=True, profile_source=measured`。这意味着 widened action-space 的 ARM64 feasibility 边界已经不只是“采到了 CSV”，而是已经能被 simulator 在线消费。

### 9.2 还不是完整离散事件模拟器

当前实现已经有：

- 请求到达序列
- container pool
- TTL eviction
- 排队
- batch 级请求计数与失败计数回写

但仍然缺少：

- provider-specific keep-alive 分布
- 多租户干扰
- 后端 I/O 争用
- region / network jitter
- 更细粒度的实例并发模型

### 9.3 Azure trace 只解决 arrival，不解决反事实性能

Azure 公共数据适合驱动：

- 到达率
- workload shape
- burstiness

但它不能直接回答：

- 同一个请求在 512MB / 1024MB / 2048MB 下的真实 latency 差异
- arm64 vs x64 的真实性能差异
- cold start overhead 在不同配置下的真实变化

这些仍然需要 OpenWhisk 或其他真实平台的标定数据。

另外，当前 minute-level profile 文件只覆盖预处理阶段选出的 Top-K 活跃函数，因此 benchmark-aware 选择仍然是在“Top-K 子集里找最像的画像”，而不是在完整 Azure 函数全集上做精确匹配。

### 9.4 README / 论文仍可能比代码更理想化

尤其要注意以下未完全兑现的点：

- 真实公有云闭环验证
- LightGBM / GP 等真正由实测数据拟合出的性能 surrogate

描述系统时必须以当前代码为准。

## 10. 当前模拟器适合做什么

当前版本适合：

- 验证 PPO / baseline 训练与评估流程
- 比较 reward shaping 的方向性影响
- 在统一 workload replay 下比较策略响应速度
- 为后续 OpenWhisk 标定数据接入预留结构

## 11. 当前模拟器还不适合做什么

当前版本还不适合直接支撑以下强结论：

- CALO 在真实 Lambda / OpenWhisk 上显著优于某个基线
- 当前冷启动结果可直接迁移到生产云
- arm64/x64 的收益量化已经可信
- CVaR / tail latency 的绝对数值具备期刊级系统证据强度

## 12. 当前最合理的下一步

如果要继续把“假的变成真的”，当前优先级建议如下：

1. 在当前 `stage2_prime` 上跑一轮对比实验，直接量化“启发式 simulator”和“校准后 simulator”在 latency / cold-start / reward 分布上的差异。
2. 决定是否回补 `timeout=120s` 剩余的 19 个低信息量点；它们已不影响 TTL 判定，但会影响表格规整性与后续论文叙事。
3. 扩更多 timeout 档位，而不是只在 `120s/300s` 两档上继续堆细节。
4. 再开始补 `arm64` 校准，而不是继续扩大 synthetic pattern 的复杂度。

## 13. 一句话结论

当前 CALO 模拟器已经从“手写启发式采样器”升级到了“模拟时钟 + workload window + container pool + 可校准服务时间模型”的框架。

而且主训练配置已经真正接上了 `results/openwhisk_calibration_x64_stage2_prime` 这套真实 OpenWhisk profile，且 simulator 已修正为：

- TTL 推断忽略 `idle_gap_sec=-1` 的 enforced 行
- cold overhead 采样优先使用 `enforced_update` 的真实冷启动行
- `512MB/x64/300s`、`1024MB/x64/300s`、`2048MB/x64/300s` 与 `3008MB/x64/300s` 的 5 个 CALO benchmark 已有真实 idle-gap TTL，当前统一估计为 `900s`
- `512/1024/2048/3008MB/x64/120s` 的 5 个 CALO benchmark 现在也已有足够的真实 idle-gap 证据，当前同样统一估计为 `900s`
- 未精确采样到的其它配置会优先走 benchmark 内最近邻 TTL，而不是直接退回默认值

但在真实系统证据层面，它目前仍处在“`x64` 的 warm/burst/cold-overhead 主矩阵已打通，且 `512/1024/2048/3008MB` 在 `x64/300s` 与 `x64/120s` 两档上的 TTL 都已进入真实推断，但更多 timeout 组合、`arm64` 与真实在线闭环仍待继续复核”的阶段。
