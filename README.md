# CALO Experiments

CALO Experiments is a standalone simulator release for code-aware and load-aware resource configuration of serverless functions. It contains the CALO Gymnasium environment, dynamic workload simulator, CodeBERT-based source feature extraction, PPO training path, and baseline policies used for simulator comparisons.

The repository is intentionally scoped to reproducible simulator experiments. It does not include manuscript asset generation, cloud deployment scaffolding, or large experiment outputs.

## Repository Layout

- `rl_optimizer/`: simulator, state and action spaces, calibrated service model, PPO policy, and baselines.
- `benchmarks/`: minimal SeBS-style function source trees used for code feature extraction.
- `config/rl_experiments/`: runnable simulator configurations.
- `run_dynamic_experiment.py`: primary CLI for smoke tests, single-seed experiments, resume, and result aggregation.
- `calo_full_suite.py`: multi-seed wrapper that writes JSON and Markdown summaries only.

## Requirements

Use an isolated Python environment. Do not install dependencies into a global or base Conda environment.

The tested path uses Python 3.11 or newer. The smoke test does not require downloading the CodeBERT model because it uses a lightweight state-space stub. Full CALO training requires `torch`, `transformers`, and access to a local or downloadable `microsoft/codebert-base` model.

## Setup

```bash
python -m venv python-venv
. python-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Smoke Test

Run the simulator without PPO training or external calibration data:

```bash
python run_dynamic_experiment.py \
  --config config/rl_experiments/full_suite.json \
  --smoke-test \
  --smoke-steps 2 \
  --disable-calibration
```

Expected behavior: the command prints one `[Smoke]` line per benchmark and load pattern. It does not create plots.

## Run One Dynamic Experiment

This command trains CALO once per benchmark/load case for the seed selected from the config and writes JSON outputs under `outputs/dynamic_seed42`:

```bash
python run_dynamic_experiment.py \
  --config config/rl_experiments/full_suite.json \
  --output-dir outputs/dynamic_seed42 \
  --seed 42 \
  --disable-calibration
```

The main outputs are:

- `dynamic_experiment_results.json`
- `dynamic_experiment_results.partial.json`
- `progress.json`
- `feature_attribution_summary.json`, when feature attribution is enabled

## Run the Multi-Seed Suite

```bash
python calo_full_suite.py \
  --config config/rl_experiments/full_suite.json \
  --output-dir outputs/full_suite \
  --disable-calibration
```

The wrapper runs each seed into `outputs/full_suite/seed_<seed>/` and writes aggregate summaries to `outputs/full_suite/aggregate/aggregate_summary.json` and `aggregate_summary.md`.

## Optional Data

Large data is intentionally kept outside the repository.

- OpenWhisk calibration tables can be supplied with `--override-calibration-dir <path>` or by setting `environment.calibration_dir` in a config.
- Azure Functions traces are expected under `external_data/azure_functions/processed/` when using `config/rl_experiments/azure_trace_tuned.json`.
- Runtime outputs should be written under `outputs/`, `results/`, or another ignored directory.

When optional data is absent, use `--disable-calibration` and synthetic load patterns.

## Output Policy

This release writes machine-readable experiment artifacts only. Plotting scripts and plot export code have been removed from the repository.

## License

This project is released under the MIT License. See `LICENSE`.
