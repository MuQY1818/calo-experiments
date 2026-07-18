# Repository Guidelines

## 沟通与代码规范

- 与维护者使用中文沟通，仓库公开文档可使用英文。
- Python 代码遵循 Google Style，使用四空格缩进、类型注解和完整 docstring。
- 不在源码、文档、日志或提交信息中加入 emoji。
- 重大功能、实验配置或公开数据变化必须同步更新本文件的更新记录。

## 开发环境

- v1 仅支持 Linux x86_64 和 Python `>=3.11,<3.12`。
- 始终在仓库内创建并使用 `python-venv`，不得向 Conda base 或系统 Python 安装依赖。
- 依赖和 `calo` 命令统一维护在 `pyproject.toml`。
- CodeBERT 固定为 `microsoft/codebert-base` revision
  `3b0952feddeffad0063f274080e3c23d75e7eb39`。修改模型或 revision 属于协议变更。

## 配置与实验协议

- `config/rl_experiments/paper_full48.json` 是唯一主配置，必须保持 79 维状态、48 actions、x64/ARM64 校准、32768 PPO steps 和 seeds 42/52/62。
- 九个 `supplementary_*.json` 记录历史 x64-only、8-action 补充实验，不得作为主配置默认值，也不得改写为 full-48 证据。
- 配置内相对路径从项目根目录解析。主配置缺少校准文件时必须立即失败，不得静默回退启发式服务模型。
- `--disable-calibration` 只用于明确标注的诊断；论文结果和 calibrated smoke 不得使用该参数。
- 不得把未随仓库发布的 Azure trace 路径或真实 OpenWhisk 基础设施配置写入公开配置。

## 工件维护

- `artifacts/claims.json` 定义论文 claim、来源、统计口径和容差；更新结果时必须同步更新来源说明和 `SHA256SUMS`。
- 每个 tracked artifact 文件不得超过 10 MiB，`artifacts/` 总量不得超过 32 MiB。
- 禁止提交 raw invocation CSV、partial、plot、日志、部署树、主机名、绝对路径、凭证、私钥、Azure 外部数据或符号链接。
- OpenWhisk 公开内容限于固定配置验证汇总、ranking 汇总及八个 replay case 的精简 summary/workload sequence。
- 未确认 canonical PPO checkpoint 前不得发布 checkpoint，也不得把本机未训练 policy 的 inference measurement 表述为论文硬件结果。
- `benchmarks/` 来源于 SeBS 并受 BSD-3-Clause 约束；作者自有代码和数据使用 MIT，归属变化必须同步 `THIRD_PARTY_NOTICES.md`。

## 复现与维护

- `calo verify` 默认仅验证冻结工件；`--measure-inference` 的 10,000 次 CPU forward pass 只报告本机数值。
- 核心测试只覆盖科研语义、公开命令和工件数值，不连接 OpenWhisk，也不运行完整三种子训练。
- 更新公开工件时必须重新运行 `calo verify` 和 calibrated smoke。

## 更新记录

- **2026-07-18：** 重写公开 README 首页，突出 CALO 论文身份、核心结果与两条复现路径；将快速安装、离线 claim 校验和双架构 20-case smoke 提前，新增主协议速查表、精简 CLI 参考、工件目录、补充实验分组、证据边界与可直接使用的软件引用，删除原先前半页过密的长段说明。未新增 CI、发布 workflow 或内部工程材料。
- **2026-07-18：** 进一步收窄公开代码面：移除未被 `calo` 入口使用的旧 baseline 层级、重复 greedy 实现和历史 compare 入口；将环境的真实 SeBS 配置改为显式可选参数，并把默认特征 benchmark 集合对齐到公开的五个 SeBS 基准。核心 PPO、Online BO、Default、Random、Greedy 路径及论文语义未改动。
- **2026-07-18：** 整理 CALO TCC v1.0.0 科研复现仓库。统一 `calo` 命令，冻结 79 维状态、48-action 双架构主协议、CodeBERT revision、三种子配置及论文数值校验；公开内容包含复现实验代码、冻结配置、精简工件、核心测试、引用和许可说明，不包含 Azure 原始数据、OpenWhisk 采集链、日志、论文 PDF 或未经确认的 PPO checkpoint。
