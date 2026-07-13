# climate-financial-risk-network

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21341328.svg)](https://doi.org/10.5281/zenodo.21341328)

Code, processed inputs, synthetic data and reported outputs accompanying the manuscript:

**"From climate risks to bank vulnerability: a reproducible scenario framework for firm-bank liability networks"**

This repository implements a reproducible climate-financial stress-testing framework in synthetic firm-bank and interbank liability networks. It does not contain confidential bank-level data and does not provide loss estimates for any real banking system.

## Repository Structure

- `src/`: Python scripts for synthetic network generation, Eisenberg-Noe clearing, reference-calibration analysis, interbank sensitivity analysis, sector-shock permutation diagnostics and article figure generation.
- `data/real/climate/`: Processed country-level climate inputs used to construct country-sector-scenario shock rates.
- `data/sector_weights.csv`: Author-defined sectoral sensitivity weights for transition and physical risk components.
- `data/synthetic_uniform_sector_distribution.zip`: Synthetic network data generated under the uniform sector-distribution assumption.
- `outputs/summary/`: Scenario-level simulation outputs for the full parameter grid.
- `outputs/gnn/`: Bank-level tabular dataset used for the reference-calibration exposure and permutation diagnostics.
- `outputs/analysis/`: Article tables, confidence intervals and sensitivity summaries.
- `outputs/metrics/`: Sector-shock permutation diagnostic outputs.

## Manuscript Scope

The manuscript reports results for a synthetic stress-testing experiment with:

- 20 synthetic network seeds;
- 100 firms and 20 banks per network;
- 6 countries, 7 sectors and 5 climate scenarios;
- interbank densities 0.05, 0.15, 0.30 and 0.50;
- interbank-liability scales 0.0, 0.5, 1.0, 1.5, 2.0 and 3.0;
- shock multipliers 0.75, 1.00, 1.25 and 1.50.

The main reference calibration uses interbank density 0.30 and interbank-liability scale 1.0. After the baseline-stability filter, the reference sample contains 2160 scenario-level observations from 18 independent synthetic seeds.

## Requirements

Python 3.10 or later is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt


