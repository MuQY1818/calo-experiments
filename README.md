<div align="center">

<h1>CALO</h1>

<p><strong>Code-Aware and Load-Aware Resource Configuration for Serverless Functions via Deep Reinforcement Learning</strong></p>

<p>Accepted by <em>IEEE Transactions on Cloud Computing</em></p>

<p>Donghong Xu, Weijue Bu, Zhouliang Ye, and Shilong Wu</p>

<p>
  <a href="https://www.python.org/downloads/release/python-3110/"><img src="https://img.shields.io/badge/Python-3.11-3776AB.svg" alt="Python 3.11"></a>
  <a href="CITATION.cff"><img src="https://img.shields.io/badge/Artifact-v1.0.0-555555.svg" alt="Artifact v1.0.0"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-2E8B57.svg" alt="MIT License"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/IEEE%20TCC-Accepted-00629B.svg" alt="Accepted by IEEE TCC"></a>
</p>

<p>
  <a href="#quick-start">Quick Start</a> |
  <a href="#reproduce-the-paper">Reproduce the Paper</a> |
  <a href="#released-artifacts">Released Artifacts</a> |
  <a href="#citation">Citation</a>
</p>

</div>

CALO combines static function-code features with dynamic workload signals to
select memory, CPU architecture, and timeout configurations for serverless
functions. This repository is the reproducibility artifact for the accepted
paper. It contains the experiment code, frozen configurations, calibrated
simulator assets, final three-seed results, and curated OpenWhisk evidence.

The repository supports two distinct workflows:

- **Verify the published results** without retraining, downloading CodeBERT,
  accessing the Azure trace dataset, or connecting to OpenWhisk.
- **Rerun the experiments** from the accepted-paper configuration for one seed
  or the complete three-seed suite.

## Key Results

The released aggregate preserves the paper's statistical convention. CALO
outperforms Online BO in mean reward, median reward, and lower-tail performance.

| Reward metric | CALO | Online BO |
| --- | ---: | ---: |
| Mean | **0.569** | 0.513 |
| Median | **0.644** | 0.579 |
| CVaR10 | **0.351** | 0.163 |

CALO achieves a mean latency of **163.4612 ms** and a mean invocation cost of
**1.480487e-6 USD**. Its mean-reward gains over Online BO are `+0.06124484`,
`+0.04934411`, and `+0.05651807` for seeds 42, 52, and 62.

The real-platform bundle covers **120 OpenWhisk targets**, including **112
deployable configurations**, with median warm and cold prediction errors of
**13.9%** and **29.0%**. It also includes eight focused replay cases. Across
these cases, CALO reduces mean p95 latency by **766.62 ms** relative to Online
BO.

All values above are checked directly by `calo verify` against the released
JSON files and SHA-256 manifest.

## Quick Start

The authoritative v1 environment is Linux x86_64 with Python 3.11. Run all
commands from a source checkout because the configurations, benchmarks, and
artifacts are part of the repository.

```bash
git clone https://github.com/MuQY1818/calo-experiments.git
cd calo-experiments

python3.11 -m venv python-venv
. python-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0,<3"
python -m pip install -e .
```

Verify all released checksums and accepted-paper claims:

```bash
calo verify
```

A successful verification starts with:

```text
CALO artifact verification: PASS
```

Then run the calibrated smoke matrix:

```bash
calo smoke --steps 1
```

This executes five benchmarks across four workload families and exercises both
x64 and ARM64 calibration routes. A successful run ends with:

```text
Smoke passed: 20 cases; mode=calibrated; observation=79; actions=48.
```

Neither command trains PPO, downloads CodeBERT, connects to OpenWhisk, or reads
the external Azure trace dataset.

## Reproduce the Paper

The main protocol is frozen in
[`config/rl_experiments/paper_full48.json`](config/rl_experiments/paper_full48.json).

| Protocol component | Accepted-paper setting |
| --- | --- |
| Benchmarks | `110.dynamic-html`, `120.uploader`, `210.thumbnailer`, `311.compression`, `411.image-recognition` |
| Workload families | `sine`, `spike`, `decay`, `random` |
| State | 79 dimensions |
| Action catalog | 48 actions: 6 memory sizes x 2 architectures x 4 timeouts |
| Calibration | x64 and ARM64 warm, cold, and burst profiles |
| Code features | 32-dimensional CodeBERT embedding |
| PPO training | 32,768 steps per CALO case with checkpoint selection |
| Seeds | 42, 52, 62 |

Run one seed:

```bash
calo run --seed 42 --output-dir outputs/paper_seed_42
```

Run the complete three-seed suite and generate its aggregate:

```bash
calo suite --output-dir outputs/paper_suite
```

Resume an interrupted run by adding `--resume`. Existing run directories or
result JSON files can also be aggregated independently:

```bash
calo aggregate \
  outputs/paper_suite/seed_42 \
  outputs/paper_suite/seed_52 \
  outputs/paper_suite/seed_62 \
  --output-dir outputs/paper_suite/aggregate
```

Full runs use `microsoft/codebert-base` at revision
`3b0952feddeffad0063f274080e3c23d75e7eb39`. CALO uses a matching local cache
when available and otherwise downloads that exact revision. After the first
successful online load, add `--offline-model` to require cached model files.

> [!NOTE]
> Quick verification is suitable for a laptop. A full seed contains 20
> benchmark/workload cases and can require many CPU-hours. The three-seed suite
> requires approximately three times the single-seed work. Runtime and peak
> memory depend on the host.

## Command Reference

| Command | Purpose | Network or platform access |
| --- | --- | --- |
| `calo verify` | Check all released hashes and paper claims | None |
| `calo smoke` | Run the 5 x 4 calibrated simulator matrix | None |
| `calo run` | Run one configured experiment seed | CodeBERT download if not cached |
| `calo suite` | Run multiple seeds and aggregate them | CodeBERT download if not cached |
| `calo aggregate` | Aggregate completed result files | None |

Use `calo <command> --help` for all options. `--disable-calibration` is a
diagnostic mode and does not reproduce the paper protocol. The optional
`calo verify --measure-inference` command measures 10,000 policy-shape forward
passes on the local CPU; it does not compare the local host with paper hardware
or use a trained checkpoint.

## Released Artifacts

```text
artifacts/
|-- calibration/{x64,arm64}/       Calibration profiles and LightGBM surrogates
|-- main_results/                  Three seed results and paper aggregate
|-- supplementary/                Ablation and sensitivity aggregates
|-- openwhisk_fixed_validation/    Fixed-target and ranking summaries
|-- replay/                        Eight replay summaries and workload sequences
|-- claims.json                    Machine-readable paper claims
|-- provenance.{json,md}           Evidence sources and reduction notes
`-- SHA256SUMS                     Integrity manifest for every artifact file
```

The repository releases the compact evidence needed to inspect and verify the
paper. It intentionally excludes the 2.6 GB Azure dataset, raw invocation
records, partial outputs, plots, collection logs, deployment infrastructure,
host identifiers, historical archives, and the paper PDF. A canonical PPO
checkpoint is not included because no confirmed checkpoint was available for
publication; the frozen protocol reconstructs a policy from training.

The simulator experiments can be regenerated from the published configuration
and calibration assets. The OpenWhisk files preserve fixed-configuration
validation and focused online replay evidence, but they are not a deployment or
data-collection pipeline. See
[`artifacts/provenance.md`](artifacts/provenance.md) for the complete evidence
boundary.

## Supplementary Experiments

The supplementary configurations reproduce the recorded ablation and
sensitivity protocols.

| Study | Configuration files |
| --- | --- |
| Load-only state | `supplementary_load_only.json` |
| Code embedding dimension | `supplementary_embedding_16.json`, `supplementary_embedding_32.json`, `supplementary_embedding_64.json` |
| Feature attribution | `supplementary_feature_attribution.json` |
| Reward sensitivity | `supplementary_reward_default.json`, `supplementary_reward_light.json`, `supplementary_reward_heavy.json`, `supplementary_reward_balanced.json` |

All files are under [`config/rl_experiments/`](config/rl_experiments/). These
historical supplementary protocols are x64-only and use the eight-action
`calibrated_x64_8` preset. They are not substitutes for the dual-architecture,
48-action main protocol.

## Tests

Run the research contract, CLI, and artifact tests inside `python-venv`:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Citation

GitHub's **Cite this repository** menu reads the metadata in
[`CITATION.cff`](CITATION.cff). Until the final IEEE bibliographic record is
available, cite the software artifact as:

```bibtex
@software{calo_experiments_2026,
  author  = {Donghong Xu and Weijue Bu and Zhouliang Ye and Shilong Wu},
  title   = {CALO Experiments},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/MuQY1818/calo-experiments}
}
```

## License

Author-owned source code and curated data are released under the
[`MIT License`](LICENSE). The vendored [`benchmarks/`](benchmarks/) functions
originate from SeBS and remain subject to the BSD-3-Clause license. See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for attribution details.
