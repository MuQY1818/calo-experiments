# CALO Experiments

CALO Experiments provides a simulator and experiment framework for code-aware and
load-aware resource configuration of serverless functions. It models dynamic
workloads, warm-container reuse, cold starts, queueing, timeout failures, and
latency-cost trade-offs under discrete memory, architecture, and timeout choices.

The framework includes a Gymnasium environment, CodeBERT-based source-code
features, a PPO training path, and several baseline policies for controlled
comparison.

## Features

- Dynamic workload simulation with sine, spike, decay, random, and Azure
  trace-driven load sources.
- Stateful container-pool model for warm reuse, TTL eviction, cold starts, and
  queueing.
- Calibrated service-time model with optional OpenWhisk calibration tables and
  LightGBM surrogate fallback.
- Discrete resource-configuration action spaces for memory, CPU architecture,
  and timeout.
- PPO, online Bayesian optimization, greedy profiling, provider-default, and
  random baselines.
- JSON and Markdown experiment summaries for single-seed and multi-seed runs.

## Installation

Use an isolated Python environment. Python 3.11 or newer is recommended.

```bash
python -m venv python-venv
. python-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Full CALO training uses `torch`, `transformers`, and the
`microsoft/codebert-base` model. The smoke test below uses a lightweight
state-space stub and does not load CodeBERT.

## Quick Smoke Test

Run a short simulator check without PPO training or calibration files:

```bash
python run_dynamic_experiment.py \
  --config config/rl_experiments/full_suite.json \
  --smoke-test \
  --smoke-steps 2 \
  --disable-calibration
```

The command prints one `[Smoke]` line for each benchmark and load pattern. This
is the fastest way to check that the simulator, workload generator, action space,
and reward path are working.

## Run Experiments

Run one dynamic experiment with the default configuration:

```bash
python run_dynamic_experiment.py \
  --config config/rl_experiments/full_suite.json \
  --output-dir outputs/dynamic_seed42 \
  --seed 42 \
  --disable-calibration
```

Run the multi-seed suite:

```bash
python calo_full_suite.py \
  --config config/rl_experiments/full_suite.json \
  --output-dir outputs/full_suite \
  --disable-calibration
```

Resume an interrupted run with `--resume`. Aggregate completed result
directories with:

```bash
python run_dynamic_experiment.py \
  --aggregate-results outputs/full_suite/seed_42 outputs/full_suite/seed_52 \
  --aggregate-output-dir outputs/full_suite/aggregate
```

## Configuration

Experiment settings live under `config/rl_experiments/`.

- `full_suite.json` runs five benchmarks with synthetic load patterns.
- `azure_trace_tuned.json` uses processed Azure Functions workload profiles.

The main fields are:

- `benchmarks`: benchmark names to evaluate.
- `algorithms`: CALO and baseline policies to run.
- `environment`: episode length, workload patterns, reward weights, action
  space, and calibration options.
- `training`: PPO update settings.
- `evaluation`: number of evaluation episodes.
- `reproducibility`: random seeds.

The default `full_suite.json` uses the `calibrated_x64_8` action-space preset:

```text
{512, 1024, 2048, 3008} MB x {x64} x {120, 300} s
```

## Outputs

Single-seed runs write result files under the selected `--output-dir`:

- `dynamic_experiment_results.json`
- `dynamic_experiment_results.partial.json`
- `progress.json`
- `feature_attribution_summary.json`, when feature attribution is enabled

Multi-seed runs create one `seed_<seed>/` directory per seed and write aggregate
summaries under `aggregate/`.

## Optional Data

OpenWhisk calibration tables can be supplied with:

```bash
python run_dynamic_experiment.py \
  --config config/rl_experiments/full_suite.json \
  --override-calibration-dir /path/to/calibration \
  --output-dir outputs/calibrated_seed42 \
  --seed 42
```

The Azure trace configuration expects processed files under:

```text
external_data/azure_functions/processed/
```

Use `--disable-calibration` when calibration tables are unavailable.

## Project Layout

```text
.
├── benchmarks/              # SeBS-style benchmark source trees
├── config/rl_experiments/   # Experiment configurations
├── rl_optimizer/            # Simulator, state/action spaces, PPO, and baselines
├── calo_full_suite.py       # Multi-seed experiment wrapper
├── run_dynamic_experiment.py
└── requirements.txt
```

## License

This project is released under the MIT License. See `LICENSE`.
