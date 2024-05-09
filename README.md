# Dividend Policy Predictor x New Horizons Financial

## Project description
[This is still an ongoing project.]

This project aims to predict a given company's dividend policy in the coming year: whether or not the company would cut dividends. We use companies' historical data and machine learning models to discover the relationship among the dividend policy and other indicators of the company (such as dividend per share, current ratio, industry to which the company belongs, etc.).
## Data usage and acknowledgement
This is a collaborative project among Muzammil Elahi, Eric Lu and Rosalie Huang, with support from David Liu.
We retrieved data from https://financialmodelingprep.com/, https://fred.stlouisfed.org/, Wikipedia and Yahoo Finance.
## Installation and execution
1. `git clone https://github.com/Muzammil-Elahi/Dividend-Policy-Predictor`
2. Register and obtain API Keys from https://financialmodelingprep.com/ and https://fred.stlouisfed.org/. Store them in strings in `.env` with names `API_KEY_FMP` and `API_KEY_FRED`.
3. To run the data ETL process with the latest data, run `database_creation.py`. You can skip this step as there is already a file `Stock_data.csv` in the folder `storage_data`. However, this data might be outdated.
4. Run `data_preprocessing.py` to do categorical encoding and oversampling. You can skip this step as there is already output files in the folder `storage_data`. However, these files are derived from the outdated `Stock_data.csv`.
5. Run `feature_engineering.py` to run Random Forest Model and find out the combination of features with maximal ROC.
6. ... (to be updated)
## Data visualization and outcome of the project
... (To be updated)
