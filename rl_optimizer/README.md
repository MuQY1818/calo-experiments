# RL Optimizer for Serverless Functions

基于强化学习的Serverless函数配置优化器

说明：本文件包含功能概览与历史设计描述。若需要查看当前仓库中模拟器的真实实现细节、执行路径与局限性，请优先阅读同目录下的 `SIMULATOR_AUDIT.md`。

## 已完成的功能 ✅

### 1. CodeBERT特征提取器（已优化）

使用CodeBERT自动提取函数语义特征，无需手工标注。

**特点**：
- 自动理解代码语义
- 基于 benchmark 语料拟合的 PCA 降维 + L2归一化
- 提取32维特征向量
- 支持缓存，避免重复计算
- GPU加速（如果可用）

**改进效果**（余弦相似度）：
- Scientific类内相似度: 0.9554（高度相似）
- Webapps类内相似度: -0.3226（多样性强）
- 跨类相似度: 0.0786（有效区分）

**使用方法**：

```bash
# 提取单个benchmark
python rl_optimizer/llm_analyzer.py --benchmark 110.dynamic-html

# 提取所有benchmark并可视化
python rl_optimizer/llm_analyzer.py --extract-all --visualize

# 自定义输出路径
python rl_optimizer/llm_analyzer.py --extract-all --output my_features.pkl
```

**编程接口**：

```python
from rl_optimizer.llm_analyzer import CodeBERTAnalyzer, BenchmarkFeatureExtractor

# 方式1：直接分析代码
analyzer = CodeBERTAnalyzer()
code = """
def handler(event):
    return {'status': 'ok'}
"""
features = analyzer.extract_features(code)  # 返回32维numpy数组

# 方式2：批量提取benchmark
extractor = BenchmarkFeatureExtractor()
features_dict = extractor.extract_all_benchmarks()
# 返回: {'110.dynamic-html': array([...]), ...}

# 加载已提取的特征
features_dict = extractor.load_features('.cache/codebert_embeddings/all_features.pkl')
```

### 2. 特征可视化

使用t-SNE降维到2D，直观展示不同类型函数的语义差异。

**结果分析**：
- Scientific函数聚类明显（501-504）
- Inference函数（411）独立性强
- Webapps函数内部差异较大
- CodeBERT成功捕捉到代码语义

查看可视化结果：
```bash
# 图片位置
results/figures/embeddings_tsne.png
```

### 3. 完整状态空间（79维）

整合所有特征的完整状态空间。

**组成**：
- LLM代码特征：32维 ✅
- 负载特征：10维 ✅
  - 当前QPS、趋势、突发检测
  - 并发请求数、冷启动概率
  - 响应时间、时间特征、容器warmth
- 函数类别：5维（one-hot） ✅
- 历史性能：5维 ✅
  - 平均延迟、成本、成功率
  - 冷启动比例、方差
- 上下文特征：27维 ✅
  - 当前配置（memory, arch, timeout）
  - 资源利用率、调用统计

**使用方法**：
```python
from rl_optimizer.state_space import StateSpace

state_space = StateSpace()
state_space.set_function('501.graph-pagerank')

# 模拟负载和性能数据
state_space.update_load(is_cold_start=False)
state_space.update_response(0.2)
state_space.update_performance(200, 0.0001, True, False)
state_space.update_configuration(512, 'x64', 120)

# 提取79维状态
state = state_space.extract_state()
```

### 4. 动作空间（48种配置）

离散动作空间，包含48种配置组合。

**维度**：
- Memory: 6个选项 [128, 256, 512, 1024, 2048, 3008] MB
- Architecture: 2个选项 [x64, arm64]
- Timeout: 4个选项 [60, 120, 300, 900] 秒
- 总计：6 × 2 × 4 = 48 种配置

**使用方法**：
```python
from rl_optimizer.action_space import ActionSpace

action_space = ActionSpace()

# 随机采样
action = action_space.sample()
config = action_space.get_configuration(action)
print(config)  # mem=512MB, arch=x64, timeout=120s

# 动作约束
mask = action_space.get_action_mask({'max_memory': 1024})
valid_actions = np.where(mask)[0]
```

### 5. 负载监控器

实时监控函数负载，提取10维负载特征。

**功能**：
- QPS追踪和趋势分析
- 突发流量检测
- 冷启动概率估计
- 容器warmth估计
- 响应时间统计

**使用方法**：
```python
from rl_optimizer.load_monitor import LoadMonitor

monitor = LoadMonitor()
monitor.record_request(is_cold_start=False)
monitor.record_response(0.3)

features = monitor.extract_features()  # 10维
stats = monitor.get_stats()
```

### 6. Gymnasium环境封装 ✅

已将SeBS封装成Gym环境，支持PPO训练。

**功能**：
- 标准Gym接口
- 支持真实执行和模拟模式
- Reward归一化（可选）
- 完整的观测空间（79维）和动作空间（48种）

**使用方法**：
```python
from rl_optimizer.environment import ServerlessFunctionEnv

env = ServerlessFunctionEnv(
    benchmark="110.dynamic-html",
    deployment='local',
    enable_real_execution=False,
    normalize_reward=False,
)

obs, info = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

### 7. 动态负载环境 ✅

支持4种动态负载模式的环境包装。

**负载模式**：
- `sine`: 正弦波周期性负载
- `spike`: 突发高峰负载
- `decay`: 逐渐衰减负载
- `random`: 随机波动负载

**特性**：
- Episode长度可配置
- 负载变化频率可调
- 配置切换惩罚（模拟冷启动成本）

**使用方法**：
```python
from rl_optimizer.dynamic_environment import DynamicLoadEnvironment

dynamic_env = DynamicLoadEnvironment(
    base_env,
    episode_length=100,
    load_change_freq=5,
    switch_penalty=0.05,
)
dynamic_env.load_pattern = "spike"
```

### 8. PPO算法实现 ✅

基于Stable-Baselines3的Pure PPO实现。

**特点**：
- 端到端深度强化学习
- 自动训练曲线绘制
- 支持长时间训练（300万步）
- 在线评估和统计

**使用方法**：
```python
from rl_optimizer.pure_ppo import PurePPO

ppo = PurePPO(
    env=dynamic_env,
    total_timesteps=3000000,
    ppo_kwargs={
        'learning_rate': 0.0003,
        'n_steps': 2048,
        'batch_size': 256,
    }
)

ppo.train()
results = ppo.evaluate(n_episodes=30)
```

### 9. Baseline算法 ✅

实现了4个对比baseline：
- `DefaultBaseline`: 固定1024MB配置
- `RandomBaseline`: 随机选择配置
- `GreedyBaseline`: 选择历史最优配置
- `OnlineBayesOptBaseline`: 在线贝叶斯优化

---

## 运行完整实验

### 快速测试（单benchmark, 推荐）

```bash
# 300万步训练，约22分钟
python calo_full_suite.py \
  --config config/rl_experiments/ppo_3m_test.json \
  --output-dir results_quick_test
```

### 完整评估套件

**实验规模**:
- 5个benchmarks (110, 120, 210, 311, 501)
- 4种负载模式 (sine, spike, decay, random)
- 5个算法 (PPO, Greedy, Bayes Opt, Default, Random)
- 3个随机种子 (42, 123, 456)
- 总计：300个实验

**运行命令**:
```bash
# 激活环境
. python-venv/bin/activate

# 运行完整实验（100万步，约12小时）
python calo_full_suite.py \
  --config config/rl_experiments/full_suite.json \
  --output-dir results_full_suite
```

**配置文件说明** (`config/rl_experiments/full_suite.json`):
```json
{
  "benchmarks": ["110.dynamic-html", "120.uploader", "210.thumbnailer",
                 "311.compression", "501.graph-pagerank"],
  "algorithms": ["ppo", "bayes_opt_online", "default", "random", "greedy"],
  "environment": {
    "episode_length": 100,
    "load_change_freq": 5,
    "switch_penalty": 0.05,
    "load_patterns": ["sine", "spike", "decay", "random"]
  },
  "training": {
    "total_timesteps": 1000000,
    "ppo_kwargs": {
      "learning_rate": 0.0003,
      "n_steps": 2048,
      "batch_size": 256
    }
  }
}
```

### 实验结果

最新完整实验结果：`results_full_suite/`

**总体性能**:
```
Algorithm            Mean Reward    vs Random    统计显著性
-----------------------------------------------------------
PPO                  -0.3799       +17.85%      ***
Greedy               -0.4067       +12.06%      **
Bayes Opt Online     -0.4159       +10.06%      *
Default              -0.4338       + 6.19%      *
Random (baseline)    -0.4624        0.00%       -
```

**6组关键统计图表**:

位于 `results_paper_full/figures/`:
1. `1_algorithm_comparison.pdf` - 算法总体性能对比
2. `2_load_pattern_comparison.pdf` - 不同负载模式下的性能
3. `3_benchmark_heatmap.pdf` - Benchmark × Algorithm 性能热图
4. `4_improvement_over_baseline.pdf` - 相对Random baseline的提升
5. `5_success_rate_comparison.pdf` - 请求成功率对比
6. `6_latency_cost_tradeoff.pdf` - 延迟-成本权衡散点图

**统计分析**:
- PPO vs Random: p=0.0002 ***（高度显著）
- PPO vs Default: p=0.0276 *（显著）
- PPO vs Greedy: p=0.4169（接近greedy性能）
- PPO vs Bayes Opt: p=0.1296（略优）

### 训练监控指标

训练过程中关注的关键指标：

1. **ep_rew_mean**: Episode平均reward
   - 越接近0越好
   - 应持续上升表明学习中

2. **entropy_loss**: 策略熵
   - 初期：-3.8左右（高探索）
   - 后期：-0.2左右（低探索，收敛）

3. **clip_fraction**: PPO clip比例
   - 初期：0.05-0.10
   - 收敛时：<0.02

4. **explained_variance**: Value function拟合质量
   - 应 >0.5
   - 越接近1越好

### 训练时间对比

| 训练步数 | 时间 | Mean Reward | vs Random | 建议 |
|---------|------|-------------|-----------|------|
| 50万    | ~3.5分钟 | -0.42 | +10% | 快速测试 |
| 100万   | ~7.5分钟 | -0.39 | +16% | 标准配置 |
| 300万   | ~22分钟  | -0.38 | +19% | 最佳性能 |

**推荐**: 100万步，性价比最高

---

## 项目结构

```
rl_optimizer/
├── __init__.py
├── README.md               # 本文件
├── llm_analyzer.py        # CodeBERT特征提取器 ✅
├── state_space.py         # 完整状态空间（79维） ✅
├── action_space.py        # 动作空间（48种配置） ✅
├── load_monitor.py        # 负载监控器 ✅
├── surrogate_model.py     # 高斯过程代理模型（待创建）
├── se_ppo.py              # Sample-Efficient PPO（待创建）
├── environment.py         # SeBS环境包装（待创建）
├── trainer.py             # 训练脚本（待创建）
└── evaluator.py           # 评估脚本（待创建）

tests/
├── test_llm_features.py   # LLM特征提取测试 ✅
└── test_state_action.py   # 状态/动作空间测试 ✅

results/
├── embeddings/
│   └── all_features.pkl   # 所有benchmark的特征 ✅
├── figures/
│   └── embeddings_tsne.png # t-SNE可视化图 ✅
├── models/                # 训练好的模型（待生成）
└── logs/                  # 训练日志（待生成）
```

---

## 依赖环境

已安装：
```bash
torch
transformers
scikit-learn
matplotlib
seaborn
```

待安装（下一阶段）：
```bash
stable-baselines3  # PPO实现
gymnasium          # RL环境
GPy                # 高斯过程
```

---

## 实验数据

### 当前状态

已成功提取11个benchmark的CodeBERT特征：

| Benchmark | 类型 | 特征维度 | 状态 |
|-----------|------|---------|------|
| 110.dynamic-html | Webapps | 32 | ✅ |
| 120.uploader | Webapps | 32 | ✅ |
| 130.crud-api | Webapps | 32 | ✅ |
| 210.thumbnailer | Multimedia | 32 | ✅ |
| 220.video-processing | Multimedia | 32 | ✅ |
| 311.compression | Utilities | 32 | ✅ |
| 411.image-recognition | Inference | 32 | ✅ |
| 501.graph-pagerank | Scientific | 32 | ✅ |
| 502.graph-mst | Scientific | 32 | ✅ |
| 503.graph-bfs | Scientific | 32 | ✅ |
| 504.dna-visualisation | Scientific | 32 | ✅ |

### 缓存信息

所有 embedding 默认缓存在 `.cache/codebert_embeddings/`：
- PCA 状态：`pca_32d.pkl`
- 原始特征缓存：`raw/*.pkl`
- 降维特征缓存：`reduced_32d/*.pkl`

重新运行不会重复计算，直接从缓存加载。

---

## 下一步计划

### 第1阶段：完整状态空间（本周）

1. 实现负载监控器（LoadMonitor）
2. 实现完整的状态提取器（ComprehensiveStateSpace）
3. 集成LLM特征、负载特征、上下文特征
4. 测试状态向量的生成

### 第2阶段：环境和基础RL（下周）

1. 实现动作空间（ActionSpace）
2. 封装SeBS环境（ServerlessFunctionEnv）
3. 实现简单的Random Policy验证流程
4. 实现基础PPO算法

### 第3阶段：代理模型和SE-PPO（第3周）

1. 实现高斯过程代理模型
2. 实现Sample-Efficient PPO
3. 训练和调试

### 第4阶段：实验和对比（第4周）

1. 实现所有baseline
2. 运行对比实验
3. 收集数据和分析

---

## 常见问题

### Q: CodeBERT模型在哪里下载？
A: 首次运行时，transformers会自动从HuggingFace下载到 `~/.cache/huggingface/`。大约500MB。

### Q: 可以在CPU上运行吗？
A: 可以，但会慢一些。推荐使用GPU（CUDA）。代码会自动检测并使用可用设备。

### Q: 特征向量的含义是什么？
A: 这是CodeBERT学到的代码语义表示，每一维没有明确的物理含义，但整体能区分不同类型的代码。

### Q: 为什么是32维而不是768维？
A: 768维是CodeBERT的原始输出。为了降低RL的状态空间复杂度，我们先在 benchmark 语料上拟合 PCA，再降到32维。可以通过修改 `embed_dim` 参数调整。

### Q: 如何更新特征？
A: 删除缓存文件后重新运行即可：
```bash
rm -rf .cache/codebert_embeddings/raw .cache/codebert_embeddings/reduced_32d .cache/codebert_embeddings/pca_32d.pkl
python rl_optimizer/llm_analyzer.py --extract-all
```

---

## 贡献者

- 基于SeBS项目构建
- 使用Microsoft CodeBERT模型
- 参考Gemini和GPT的建议设计方案

## 许可证

遵循SeBS项目的许可证
