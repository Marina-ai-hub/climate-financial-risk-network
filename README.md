# climate-financial-risk-network

This repository contains the code, configuration-related inputs and synthetic outputs required to reproduce the analysis reported in the manuscript:

"From climate risks to bank vulnerability: a reproducible scenario framework for firm-bank liability networks"

The repository uses synthetic firm-bank and interbank liability networks. It does not contain confidential bank-level data and does not provide loss estimates for any real banking system.

## Repository structure

- `src/`: Python scripts for network generation, clearing, article tables, sensitivity analysis, permutation diagnostics and figures.
- `data/real/climate/`: processed country-level climate indicators and scenario shock inputs.
- `data/synthetic_uniform_sector_distribution/`: synthetic network files generated under the uniform sector-distribution assumption.
- `outputs/summary/`: archived scenario-level simulation summary.
- `outputs/gnn/`: archived p30 bank-level synthetic dataset used for exposure diagnostics.
- `outputs/analysis/`: reported article tables and reference-calibration outputs.
- `outputs/metrics/`: sector-shock permutation diagnostic outputs.

## Requirements

Python 3.10 or later is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt.
