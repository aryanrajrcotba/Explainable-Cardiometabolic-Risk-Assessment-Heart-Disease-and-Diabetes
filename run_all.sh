#!/usr/bin/env bash
# Reproduce every number, table and plot in results/ (about 10 minutes on a laptop).
set -e
python -m src.data               # downloads raw data on first run, prints a summary
python -m src.models             # tune 5 models, calibrate, choose thresholds, save results/models
python -m src.ers                # experiment E: ERS vs correctness (LR, RF, XGBoost; heart + diabetes)
python -m src.fairness           # subgroup + cross-source checks
python -m pytest -q tests        # sanity checks
