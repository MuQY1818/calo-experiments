# Current Pipeline And Experiments

本文档用于快速回答两个问题：

1. 现在这套 CALO pipeline 是怎么串起来的。
2. 到目前为止，仓库里已经做过哪些实验，哪些是当前主稿采用的口径，哪些只是补充或内部探针。

这不是教程文档，而是一份面向仓库维护者的 explanation + reference。

## 1. 先看哪些文件

建议按下面顺序看：

1. `results/README.md`
   当前最新的结果索引，优先告诉你“现在主看哪批结果”。
2. `CURRENT_PIPELINE_AND_EXPERIMENTS.md`
   当前这份总览，解释 pipeline 和实验版图。
3. `rl_optimizer/SIMULATOR_AUDIT.md`
   解释 simulator 当前真实实现，而不是论文理想化描述。
4. `config/rl_experiments/full_suite.json`
   当前稳定的默认训练配置，走 `calibrated_x64_8`。
5. `config/rl_experiments/full_suite_full48.json`
   当前 widened full-48 broad comparison 配置，走双架构校准。
6. `results/final_paper_data_inventory.json`
   论文打包用的冻结清单，适合做存在性核对，不总是代表最新 headline 口径。
7. `results/paper_ready_status.json`
   对 `final_paper_data_inventory.json` 的存在性检查结果。

当前需要特别注意一件事：

- `results/README.md` 反映的是“当前推荐查看的结果口径”。
- `results/final_paper_data_inventory.json` 更接近“论文包里冻结的结果清单”。
- 这两者有少量不同步的历史痕迹，尤其是 broad comparison 主路径。

如果你只想知道“现在主结果看哪里”，优先看 `results/README.md`。

## 2. 当前端到端 pipeline

当前 pipeline 可以分成七步。

### 2.1 Benchmark 与代码语义特征

- 输入 benchmark 来自 `benchmarks/` 和 `benchmarks-data/`。
- 代码语义特征由 `rl_optimizer/llm_analyzer.py` 提取。
- 当前使用 CodeBERT embedding，再做 PCA 降到 32 维。
- 状态空间总维度为 79，组成是：
  - 32 维代码特征
  - 10 维负载特征
  - 5 维函数类别
  - 5 维历史性能
  - 27 维上下文

### 2.2 真实 OpenWhisk 校准采集

- 校准采集入口是 `scripts/collect_openwhisk_calibration.py`。
- 采集对象是单机 OpenWhisk 上的真实 warm、cold、burst 行为。
- 当前权威校准目录是：
  - `results/openwhisk_calibration_x64_stage2_prime/`
  - `results/openwhisk_calibration_arm64_stage1/`

当前两套目录的覆盖是：

- `x64`
  - `warm_profile.csv`: 120 行
  - `cold_profile.csv`: 362 行
  - `burst_profile.csv`: 360 行
- `arm64`
  - `warm_profile.csv`: 120 行
  - `cold_profile.csv`: 232 行
  - `burst_profile.csv`: 360 行

### 2.3 校准后的 simulator

当前 simulator 不是旧的 toy heuristic 环境，而是由下面这条链路组成：

- `DynamicLoadEnvironment`
- `ServerlessFunctionEnv`
- `SimulationClock`
- `ContainerPool`
- `CalibratedServiceModel`

它的核心逻辑是：

1. workload source 生成一个 step 的请求窗口。
2. `ContainerPool` 决定 warm hit、cold start 和排队。
3. `CalibratedServiceModel` 从真实 profile 中采样 warm runtime、cold overhead、burst slowdown。
4. `StateSpace` 汇总出 79 维状态。
5. `ServerlessFunctionEnv` 计算 reward 并推进 episode。

更细的实现说明请直接看 `rl_optimizer/SIMULATOR_AUDIT.md`。

### 2.4 两条主要训练/评估轨

当前仓库实际上有两条主轨，不要混淆。

#### A. 稳定默认轨

- 配置文件：`config/rl_experiments/full_suite.json`
- 动作空间：`calibrated_x64_8`
- 校准输入：`results/openwhisk_calibration_x64_stage2_prime/`
- 用途：
  - 稳定训练
  - 默认 smoke test
  - x64 校准子空间上的 broad comparison

#### B. Widened full-48 轨

- 配置文件：`config/rl_experiments/full_suite_full48.json`
- 动作空间：`full_48`
- 校准输入：
  - `results/openwhisk_calibration_x64_stage2_prime/`
  - `results/openwhisk_calibration_arm64_stage1/`
- 用途：
  - 双架构 widened broad comparison
  - 当前 full-48 headline 图表与主结果

### 2.5 Broad comparison 与补充实验

训练与聚合的两个主要入口是：

- `run_dynamic_experiment.py`
  - 跑单个或小批量动态实验
  - 支持 `--resume`
  - 也支持 `--smoke-test`
- `calo_full_suite.py`
  - 聚合多 benchmark、多负载、多算法、多 seed
  - 输出总结果与论文图表

### 2.6 真机 fixed validation

- 入口脚本：`scripts/review_real_platform_validation.py`
- 当前关键目录：
  - `results/review_real_platform_validation_x64_full24_all5_live_20260413/`

这一步的作用不是训练 RL，而是：

- 在真实 OpenWhisk 上验证 x64 full24 动作集合的可部署性。
- 产出 `validation_summary.json`，供后续真机 online replay 做 feasible-action mask。

### 2.7 真机 online replay

- 入口脚本：`scripts/review_real_platform_online_ab.py`
- 主要配置在 `config/rl_experiments/review_real_platform_online_batch*.json`

当前真机 replay 的作用是：

- 用共享负载序列在真机上直接比较 `ppo`、`bayes_opt_online`、`default`。
- 利用 fixed-validation 结果对不可部署动作做 remap。
- 验证 simulator-trained policy 是否能迁移到真实平台。

## 3. 已完成的实验版图

下面按“主稿当前采用”“已完成补充实验”“内部/探索性结果”三层来整理。

## 4. 主稿当前采用的结果

### 4.1 Full-48 broad comparison

- 主要结果目录：
  - `results/full_suite_full48_bootstrap_shortlist_seed42/`
  - `results/full_suite_full48_bootstrap_shortlist_seed52/`
  - `results/full_suite_full48_bootstrap_shortlist_seed62/`
  - `results/full_suite_full48_bootstrap_shortlist_multiseed/full_suite_results.json`
- 对应配置：
  - `config/rl_experiments/full_suite_full48.json`
- 当前 top-line：
  - `CALO` mean reward: `0.5688`
  - `Online BO` mean reward: `0.5131`
  - `Greedy` mean reward: `-0.0517`
  - `Provider Default` mean reward: `-0.3037`
  - `Random` mean reward: `-30.5947`
- 当前 20 个 benchmark-pattern 聚合单元上，`CALO` 相对 `Online BO / Greedy / Default / Random` 为 `20/20` 全胜。

这是当前最应该被视为 headline broad comparison 的结果口径。

### 4.2 真机 fixed validation

- 主要结果目录：
  - `results/review_real_platform_validation_x64_full24_all5_live_20260413/`
- 关键文件：
  - `validation_summary.json`
  - `ranking_summary.json`
  - `validation_table.tex`
- 当前覆盖：
  - `5 benchmarks x 24 x64 actions = 120` 个 target
  - 已完整完成 `120/120`
- 当前结论：
  - `110.dynamic-html`、`120.uploader`、`210.thumbnailer`、`311.compression` 的 `24/24` x64 配置都完成了 warm/cold 采样。
  - `411.image-recognition` 的 `128MB` 和 `256MB` 共 `8` 个低内存 x64 组合在真机上持续 infeasible。
  - `512MB+` 的 `16` 个组合可以正常完成 warm/cold probe。

这一步是后续真机 adaptive replay 的 deployability 依据。

### 4.3 真机 adaptive-stress online replay

当前主稿采用的是更保守的 8-case 口径，而不是所有 exploratory run。

相关目录：

- `results/review_real_platform_online_adaptive_stress_bayes_focus_clean_20260414_fg/batch_summary.json`
- `results/review_real_platform_online_adaptive_stress_bayes_focus_411_decay_clean_20260415/batch_summary.json`
- `results/review_real_platform_online_adaptive_stress_bayes_focus_411_random_clean_20260415/batch_summary.json`
- `results/review_real_platform_online_411_seed52_anchorprime_rerun_20260414/summary.json`

当前主稿里的 8-case aggregate 结论是：

- 对 `Default` 的 mean raw reward delta: `+0.4831`
- 对 `Online BO` 的 mean raw reward delta: `+0.5319`
- 对 `Default` 的 steady-state delta: `+0.4893`
- 对 `Online BO` 的 steady-state delta: `+0.5459`
- 相对 `Online BO` 的 mean p95 latency reduction: `766.62 ms`

这批结果目前是“主稿采用的 strongest-baseline 真机在线证据”。

## 5. 已完成的补充实验

### 5.1 稳定默认 broad comparison

- 结果目录：
  - `results/full_suite_stage2_prime_seed42_rerun/`
  - `results/full_suite_stage2_prime_seed52_32d/`
  - `results/full_suite_stage2_prime_seed62_32d/`
  - `results/full_suite_stage2_prime_multiseed/aggregate_summary.json`
- 对应配置：
  - `config/rl_experiments/full_suite.json`
- 当前 aggregate：
  - 相对 `Online BO`: `43/60` 胜，mean raw reward delta `+0.0613`
  - 相对 `Default`: `60/60` 胜，mean raw reward delta `+0.5983`
  - 相对 `Greedy`: `54/60` 胜，mean raw reward delta `+0.3929`

这批结果仍然重要，因为它代表了“稳定默认 x64_8 轨”的主聚合结果，但它不是当前 full-48 headline 图表口径。

### 5.2 Load-Only ablation

- 结果目录：
  - `results/shared_ablation_90k_seed42/`
  - `results/shared_ablation_90k_seed43/`
  - `results/shared_ablation_90k_seed44/`
  - `results/shared_ablation_90k_aggregate/aggregate_summary.json`
- 结论：
- `CALO` 相对 `Load-Only CALO` 为 `11/15` 胜
  - mean raw reward delta 为 `+0.0855`

这批结果用于支撑“代码感知 + 负载感知”优于仅负载感知。

### 5.3 Focused full-48 widened study

- 结果目录：
  - `results/infeasible_boundary_full48_warmstart_checkpointed_conservative_411_broad_seed42/`
  - `results/infeasible_boundary_full48_warmstart_checkpointed_conservative_411_broad_seed52/`
  - `results/infeasible_boundary_full48_warmstart_checkpointed_conservative_411_broad_seed62/`
  - `results/infeasible_boundary_full48_conservative_broad_multiseed/aggregate_summary.json`
- 结论：
  - 相对 `Online BO`: `12/12` 胜
  - 相对 `Greedy`: `12/12` 胜
  - 相对 `Default`: `12/12` 胜

这批结果主要用于 widened action-space 的 focused evidence，不等同于 full-48 全局 broad comparison。

### 5.4 Reward sensitivity

- 结果目录：
  - `results/review_reward_sensitivity_default_multiseed/aggregate_summary.json`
  - `results/review_reward_sensitivity_light_multiseed/aggregate_summary.json`
  - `results/review_reward_sensitivity_heavy_multiseed/aggregate_summary.json`
  - `results/review_reward_sensitivity_weight5050_multiseed/aggregate_summary.json`
- 当前相对 `Online BO` 的总体结果：
  - `default`: `9/12` 胜，mean raw reward delta `+0.0955`
  - `light`: `8/12` 胜，mean raw reward delta `+0.1388`
  - `heavy`: `8/12` 胜，mean raw reward delta `+0.1271`
  - `weight5050`: `9/12` 胜，mean raw reward delta `+0.0817`

这批结果的用途是说明 reward 形状变化后，结论方向没有彻底反转。

### 5.5 Embedding sensitivity

- 结果目录：
  - `results/review_embedding_dim_16_multiseed/aggregate_summary.json`
  - `results/review_embedding_dim_32_multiseed/aggregate_summary.json`
  - `results/review_embedding_dim_64_multiseed/aggregate_summary.json`
- 当前这批 probe 的总体 mean reward：
  - `16-D`: `-0.1897`
  - `32-D`: `-0.2264`
  - `64-D`: `-0.0995`

这批实验是 compact probe，不是 full broad comparison。它们适合做敏感性说明，不适合作为 headline 主结果。

### 5.6 Feature attribution

- 结果目录：
  - `results/review_feature_attribution_multiseed/aggregate_summary.json`
  - `results/review_feature_attribution_quick/feature_attribution_summary.json`
  - `results/review_feature_attribution_seed52/feature_attribution_summary.json`
  - `results/review_feature_attribution_seed62/feature_attribution_summary.json`

这批结果的用途是 post-hoc probe：

- 看不同特征组被打乱或移除后 reward 如何变化。
- 当前 per-seed 细节中，`code`、`load`、`history`、`context` 都出现过可观影响。
- 但不同 seed 的排序并不完全一致，因此它更适合作为解释性证据，而不是强因果结论。

## 6. 内部或探索性结果

下面这些结果是有价值的，但不建议直接把它们当成“当前主稿唯一口径”。

### 6.1 411 guarded online replay summary

- 结果目录：
  - `results/review_real_platform_online_411_guarded_summary/summary.json`
- 当前 summary：
  - case 数：`7`
  - overall mean raw reward delta: `+1.5443`
  - overall mean steady-state reward delta: `+1.5461`
  - overall mean p95 latency reduction: `2343.11 ms`

这批结果更强，但更偏 `411.image-recognition` 单 benchmark 的 guarded evidence。当前更适合把它看成“强化证据”或“探索性更强的真机结果”，而不是主稿里最保守的主表口径。

### 6.2 `*_quick`、`*_smoke`、旧 probe 目录

例如：

- `results/review_reward_sensitivity_*_quick/`
- `results/review_embedding_dim_*_quick/`
- `results/review_feature_attribution_quick/`
- `results/openwhisk_burst_probe_base/`
- `results/openwhisk_burst_probe_prime/`
- 任何 `*_smoke` 目录

这些目录通常用于：

- 快速看趋势
- 做链路检查
- 验证脚本改动没有把实验跑坏

它们不是当前应优先引用的正式结论入口。

## 7. 当前最重要的“源事实”

如果你现在只想记住最关键的事实，可以记下面这些：

1. simulator 已经切成“真实 OpenWhisk 校准驱动”的实现，不再是纯启发式 toy 环境。
2. 当前仓库同时保留了稳定默认 `x64_8` 轨和 widened `full_48` 轨。
3. 当前 full-48 broad comparison 的 headline 结果在：
   - `results/full_suite_full48_bootstrap_shortlist_multiseed/full_suite_results.json`
4. 当前真机 fixed-validation 的 deployability 依据在：
   - `results/review_real_platform_validation_x64_full24_all5_live_20260413/validation_summary.json`
5. 当前主稿采用的真机 strongest-baseline 在线证据，是更保守的 8-case adaptive-stress 口径，而不是所有 exploratory guarded run。

## 8. 目前还没做完什么

当前还没完全闭合的地方主要有三类：

1. `inventory` 与“当前最新 headline 结果索引”之间还存在少量历史不同步。
2. 真正的 RL real-execution closed loop 还没有成为当前主路径。
3. `arm64` 真机 online replay 还不是当前主证据的一部分。

## 9. 推荐的排查顺序

如果后面你再问“现在到底跑到哪一步了”，建议按下面顺序排查：

1. 看 `results/README.md`
2. 看这份 `CURRENT_PIPELINE_AND_EXPERIMENTS.md`
3. 看 `results/paper_ready_status.json`
4. 看 `config/rl_experiments/full_suite.json` 和 `config/rl_experiments/full_suite_full48.json`
5. 看 `results/review_real_platform_validation_x64_full24_all5_live_20260413/`
6. 看 `results/review_real_platform_online_adaptive_stress_bayes_focus*.json`

这样基本就不会再被旧目录、quick run 和 exploratory run 混淆。
