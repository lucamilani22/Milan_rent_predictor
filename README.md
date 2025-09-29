# Milan_rent_predictor
**Gradient-boosted models (XGBoost/LightGBM/CatBoost) with feature engineering and geo-enrichment to predict Milan apartment rents.**

> Dataset: 7,334 Milan rental listings from Immobiliare.it. 4,500 have prices (`y`) for training; 2,834 are held out for prediction. 11 input variables include size, contract type, availability, description, other features, condition, floor, elevator, energy class, condo fees, and zone.

## Project Overview

Predicted monthly rent `y` for held-out listings using:
- **Feature engineering** from raw text (description/other_features), contract metadata, energy class, floor/elevator, etc.
- **Geo features**: cache **zone** coordinates (manual + `Nominatim` fallback) and compute distance to **Duomo** via Haversine.
- **Models**: tuned **XGBoost**, **LightGBM**, and **CatBoost** (with Optuna), stacked/averaged via out-of-fold predictions.
- **Evaluation**: Mean Absolute Error (MAE) with **StratifiedKFold** (by rent bins) on the 4,500 labeled examples; generate `submission.csv` for the 2,834 test rows.

**Inputs summary:** 11 columns listed in the `train.csv` 

## Repository Structure

- `train.csv` : all 11 variables and y with rent price
- `test.csv` : all 11 variables without y
- `zones_coordinates.csv`: geo-spatial coordinates to feature engineer the average price per zone
- `model.ipynb`: notebook with the machine learning model

## Features
- Contract parsing: split contract_type into (contract, term); one-hot term_*.
- Availability flag: binary available from text.
- Text extraction: bedrooms, bathrooms parsed from description.
- Other features: parse pipe-separated flags (garden, terrace, balcony, furnished, concierge).
- Floor normalization: map strings (Ground, Semi-basement, Mezzanine) → numeric.
- Elevator: yes/no → 1/0.
- Energy class: map A…G, unknown → ordinal code.
- Condo fees: median imputation.
- Zone → distance to Duomo: via cached lat/lon + Haversine; replace zone with numeric distance feature.
- Avg rent per room: per-zone mean of y/bedrooms merged into rows.

## Models & Training
- **Base learners:** XGBoost, LightGBM, CatBoost (regressors).
- **Tuning:** Optuna (tree-structured search space, 50 trials per model by default).
- **CV:** 5-fold StratifiedKFold on rent bins for stable MAE.
- **Blending:** mean or weighted mean of base learners; (optionally) meta-learner with ridge on OOF.

## How to run
All steps for data loading, preprocessing, training, evaluation, and generating submissions are documented directly in the Jupyter notebook:
Follow the instructions in each notebook to reproduce the results and create the submission files.

## Credits
Assignment developed as part of coursework in the Machine Learning course at Bocconi University, academic year 2024-2025.

## Contact
For questions: luca.milani2@studbocconi.it



