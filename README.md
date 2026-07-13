# climate-financial-risk-network

This repository contains the code, configuration-related inputs and synthetic outputs required to reproduce the analysis reported in the manuscript:

"From climate risks to bank vulnerability: a reproducible scenario framework for firm-bank liability networks"

The repository uses synthetic firm-bank and interbank liability networks. It does not contain confidential bank-level data and does not provide loss estimates for any real banking system.

## Repository structure

- `src/`: Python scripts for climate-shock construction, synthetic network generation, Eisenberg–Noe clearing, sensitivity analyses, the sector-shock permutation diagnostic, and the production of article tables and figures.
- `data/real/climate/`: Processed country-level climate indicators and inputs used to construct the country-sector-scenario shock table.
- `data/synthetic_uniform_sector_distribution.zip`: Compressed synthetic network data generated under the uniform sector-distribution assumption.
- `outputs/summary/`: Scenario-level simulation outputs for the full parameter grid under the uniform sector-distribution assumption.
- `outputs/gnn/`: Bank-level dataset for the reference calibration with interbank-liability scale 1.0 and density 0.30.
- `outputs/analysis/`: Summary statistics, sensitivity results, and article tables for the reference calibration.
- `outputs/metrics/`: Results of the sector-shock permutation diagnostic.

## Requirements

Python 3.10 or later is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21341328.svg)](https://doi.org/10.5281/zenodo.21341328)
