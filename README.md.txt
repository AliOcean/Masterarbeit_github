# Master Thesis DSS Prototype

This repository contains the Streamlit prototype developed as part of the master thesis.

## Run locally

pip install -r requirements.txt

python -m streamlit run app.py

## Required files

The app requires the trained XGBoost model, feature list, imputer and prepared feature dataset.

## Structure

- app.py: Streamlit application
- model/: trained model and preprocessing artifacts
- data/: prepared feature dataset
- assets/plots/: optional evaluation plots