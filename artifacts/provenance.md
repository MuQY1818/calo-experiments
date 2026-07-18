# CALO v1.0.0 Artifact Provenance

This directory contains the cached outputs and calibration assets used to verify
the accepted CALO paper. `claims.json` maps the paper-facing values to released
files, and `SHA256SUMS` records every released artifact file except the manifest
itself.

## Evidence Boundaries

The main five-benchmark, four-workload, three-seed comparison is a calibrated
simulator experiment. The simulator consumes aggregated OpenWhisk warm, cold,
and burst profiles for x64 and ARM64. The fixed OpenWhisk validation is a direct
x64 sweep of 120 configuration targets. The adaptive OpenWhisk evidence is the
eight focused replay cases in `replay/`.

The package does not include the 2.6 GB Azure trace corpus, raw invocation CSV
files, OpenWhisk deployment and collection infrastructure, logs, plots, paper
PDFs, historical experiments, or an unconfirmed PPO checkpoint. Full training
uses the released configuration to reconstruct a policy; cached final results
support exact verification of the accepted-paper numbers.

## Data Reduction

Calibration profiles are aggregate tables. Their architecture-specific
LightGBM models are the six warm, cold-overhead, and burst estimators for each
architecture. Main seed files are final JSON outputs; partial files, figures,
and logs were excluded. Supplementary experiments retain aggregate JSON only.
Fixed-validation rows omit deployed function names, and replay summaries omit
deployed names and per-invocation records while retaining the workload sequence
needed to inspect each paired case.

No released file contains a credential, private key, host name, local absolute
path, or infrastructure configuration.
