# CALO Experiments

CALO (Code-Aware and Load-Aware Optimization) 实验框架，提取自 [serverless-benchmarks](https://github.com/...) 仓库。

包含：

- `rl_optimizer/` — 强化学习优化器核心（环境、状态/动作空间、PPO 训练、校准服务模型、工作负载模拟）
- `calo_full_suite.py` — 全量评估套件入口（多 benchmark × 算法 × 种子的网格实验）
- `run_dynamic_experiment.py` — 单次/小批量动态实验入口（支持 smoke test、断点续跑）
- `scripts/` — 实验脚本（OpenWhisk 校准采集、结果绘图、真机验证、Azure 数据预处理）
- `config/` — 配置（RL 实验配置、校准矩阵、平台部署配置）
- `tools/` — SeBS 工具（OpenWhisk 准备等）
- `dockerfiles/` — 函数 Docker 构建文件

## 快速开始

```bash
python-venv/bin/pip install -r requirements.txt
python-venv/bin/python run_dynamic_experiment.py --config config/rl_experiments/full_suite.json --smoke-test --smoke-steps 2
```

## 实验数据

大规模实验结果（results/、archive/、external_data/）存放在 `../calo-experiment-data/`，与本仓库分离以减小体积。

## 关联仓库

- `../serverless-benchmarks/` — SeBS 框架与 benchmark 函数实现
