# CALO Experiments

Reproducibility artifact for the accepted IEEE Transactions on Cloud Computing
paper **"Code-Aware and Load-Aware Resource Configuration for Serverless
Functions via Deep Reinforcement Learning"** by Donghong Xu, Weijue Bu,
Zhouliang Ye, and Shilong Wu.

This v1.0.0 release freezes the accepted-paper experiment protocol: a
79-dimensional state, 48 resource-configuration actions, five SeBS benchmarks,
four workload families, dual x64/ARM64 calibration, and three random seeds. It
also publishes the curated result and OpenWhisk validation artifacts needed to
check the paper's reported values without retraining a model or accessing an
OpenWhisk deployment.

Use a GitHub source checkout and run reproduction commands from the repository
root so the published configurations, benchmarks, and artifacts remain
available. The Python package alone does not bundle the experiment assets.

## Accepted-Paper Results

The bundled claim definitions preserve the paper's aggregation convention.
`calo verify` checks these values against the published result files and their
SHA-256 manifest.

| Metric | CALO | Online BO |
| --- | ---: | ---: |
| Mean reward | 0.569 | 0.513 |
| Median reward | 0.644 | 0.579 |
| CVaR10 reward | 0.351 | 0.163 |

CALO's mean latency is 163.4612 ms and its mean invocation cost is
1.480487e-6 USD. The CALO minus Online BO mean-reward deltas for seeds 42, 52,
and 62 are +0.06124484, +0.04934411, and +0.05651807, respectively.

The fixed OpenWhisk validation covers 120 target configurations, of which 112
were deployable. Its median warm and cold prediction errors are 13.9% and
29.0%. The focused online replay set contains eight cases. Across those cases,
the mean raw-reward deltas are +0.4831 versus Provider Default and +0.5319
versus Online BO; the steady-state deltas are +0.4893 and +0.5459, and the mean
p95 latency reduction versus Online BO is 766.62 ms.

## Quick Verification

The authoritative v1 environment is Linux x86_64 with Python 3.11. Create the
environment inside the repository; do not install the project into a Conda base
environment.

```bash
python3.11 -m venv python-venv
. python-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0,<3"
python -m pip install -e .
```

Verify every bundled checksum and frozen paper claim:

```bash
calo verify
```

The first output line should be `CALO artifact verification: PASS`.

Run the calibrated smoke test over all five benchmarks and four workload
families:

```bash
calo smoke --config config/rl_experiments/paper_full48.json --steps 1
```

The smoke test executes exactly 20 cases. Each case reports a 79-dimensional
observation and 48 actions, and the complete run exercises both x64 and ARM64
calibration routes. Neither command downloads CodeBERT, trains PPO, connects to
OpenWhisk, or reads the external Azure trace dataset.

The final output line should be:

```text
Smoke passed: 20 cases; mode=calibrated; observation=79; actions=48.
```

To measure the policy architecture on the local CPU, append the optional flag:

```bash
calo verify --measure-inference
```

This performs exactly 10,000 single-observation forward passes after warmup and
reports local median and p95 latency. It does not compare the local machine with
the paper hardware and cannot change the frozen-claim verification result. No
trained PPO checkpoint is used for this measurement.

## Reproduce Experiments

Run the accepted-paper protocol for one seed:

```bash
calo run \
  --config config/rl_experiments/paper_full48.json \
  --output-dir outputs/paper_seed_42 \
  --seed 42
```

Run all three accepted-paper seeds and aggregate them:

```bash
calo suite \
  --config config/rl_experiments/paper_full48.json \
  --output-dir outputs/paper_suite \
  --seeds 42,52,62
```

Resume an interrupted `run` or `suite` with `--resume`. Existing result
directories can be aggregated independently:

```bash
calo aggregate \
  outputs/paper_suite/seed_42 \
  outputs/paper_suite/seed_52 \
  outputs/paper_suite/seed_62 \
  --output-dir outputs/paper_suite/aggregate
```

The main configuration fixes 32-dimensional code features, 32,768 PPO steps,
checkpoint selection, an oracle-shortlist bootstrap, and seeds 42, 52, and 62.
Relative paths in every published configuration are resolved from the project
root. Missing calibration files are errors. `--disable-calibration` is an
explicit diagnostic mode and does not reproduce the paper protocol.

Code features use `microsoft/codebert-base` at revision
`3b0952feddeffad0063f274080e3c23d75e7eb39`. A normal `run` or `suite` uses a
matching local cache when present and otherwise downloads that exact revision.
After one successful online load, use `--offline-model` to require cached files;
the command fails with preparation guidance if the pinned revision is absent.

## Resource Expectations

| Command | Work performed | External access |
| --- | --- | --- |
| `calo verify` | Checksums and statistics over the curated artifacts | None |
| `calo smoke --steps 1` | 20 calibrated simulator cases, one step each | None |
| `calo verify --measure-inference` | 10,000 CPU policy-shape forwards | None |
| `calo run --seed 42` | 20 benchmark/workload cases; 32,768 PPO steps for each CALO case plus baselines | CodeBERT download only when not cached |
| `calo suite` | Three complete single-seed runs and aggregation | CodeBERT download only when not cached |

Quick verification is intended for a laptop. A full seed is a research workload
and can require many CPU-hours; the exact wall time and peak memory depend on
the host. The three-seed suite is approximately three times the single-seed
work. Full training additionally needs space for the pinned CodeBERT cache and
generated checkpoints/results. Outputs and model files are ignored by Git.

## Published Configurations

`config/rl_experiments/paper_full48.json` is the only main protocol. It uses the
five benchmarks `110.dynamic-html`, `120.uploader`, `210.thumbnailer`,
`311.compression`, and `411.image-recognition`; the `sine`, `spike`, `decay`,
and `random` workload families; and the dual-architecture `full_48` action
catalog.

Nine supplementary configurations reproduce the recorded ablation and
sensitivity protocols:

- `supplementary_load_only.json`
- `supplementary_embedding_16.json`
- `supplementary_embedding_32.json`
- `supplementary_embedding_64.json`
- `supplementary_feature_attribution.json`
- `supplementary_reward_default.json`
- `supplementary_reward_light.json`
- `supplementary_reward_heavy.json`
- `supplementary_reward_balanced.json`

These supplementary protocols are all x64-only and use the historical
`calibrated_x64_8` action preset. They are runnable through `calo run` and
`calo suite`, but they are not substitutes for the main full-48,
dual-architecture protocol.

## Artifact Contents

```text
artifacts/
|-- calibration/{x64,arm64}/       Profiles and LightGBM service surrogates
|-- main_results/                  Three seeds and accepted-paper aggregate
|-- supplementary/                Ablation and sensitivity aggregates
|-- openwhisk_fixed_validation/    Target and ranking summaries
|-- replay/                        Eight summaries and workload sequences
|-- claims.json                    Machine-readable claim definitions
|-- provenance.json                Machine-readable source metadata
|-- provenance.md                  Human-readable evidence provenance
`-- SHA256SUMS                     Full artifact integrity manifest
```

The release excludes the 2.6 GB Azure dataset, original invocation records,
partial results, plots, collection logs, raw deployment trees, infrastructure
configuration, host identifiers, absolute source paths, and historical
archives. It also excludes the canonical PPO checkpoint because no confirmed
checkpoint was available for publication. The frozen configuration rebuilds a
model; the bundled final results verify the accepted-paper numbers.

## Evidence Boundary

The simulator results and supplementary aggregates can be regenerated from the
published configurations and curated calibration data. `calo verify` checks the
already published evidence; it is not a training run and does not claim to
reproduce machine-specific timing.

The OpenWhisk files document fixed-configuration validation and eight focused
online replay cases. They support the real-platform results reported by the
paper, but this repository does not include the deployment and data-collection
chain required to recollect them. No verification command connects to an
OpenWhisk cluster.

## Optional Tests

Run the core research tests in `python-venv`:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Citation And License

Use [CITATION.cff](CITATION.cff) to cite software release v1.0.0. A DOI is
intentionally not included until an official record exists.

Author-owned source code and curated data are released under the MIT License.
The vendored `benchmarks/` files originate from SeBS and remain subject to the
BSD-3-Clause license. See [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
