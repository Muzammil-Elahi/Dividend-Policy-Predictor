import pandas as pd
import pickle
import requests
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
import os
import csv
from dotenv import load_dotenv
from company_data_extractor import company_data_extractor

X_train = pd.read_csv("storage_files/X_train.csv")
X_test = pd.read_csv("storage_files/X_test.csv")
X_train_oversampled = pd.read_csv("storage_files/X_train_oversampled.csv")
y_train_oversampled = pd.read_csv("storage_files/y_train_oversampled.csv")
with open('storage_files/best_models_rf_36_features.pkl', 'rb') as file:
    best_model_rf = pickle.load(file)

categorical_columns = ["industry","sector","symbol"]
other_columns = [col for col in X_train.columns if col not in categorical_columns]
column_transformer = ColumnTransformer(
    transformers=[
        ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_columns)
    ],
    remainder='passthrough'
)
X_train_transformed = column_transformer.fit_transform(X_train)
X_test_transformed = column_transformer.transform(X_test)
best_model_rf.fit(X_train_oversampled, y_train_oversampled)
pipeline = Pipeline(steps=[('preprocessor', column_transformer),
                           ('classifier', best_model_rf)
                           ])
try:
    tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    ticker_table = tables[0]
    tickers = ticker_table['Symbol'].tolist()

    print('Please enter a company ticker:')
    company = input()
    if company not in tickers:
        print('Company ticker not found.')
        quit()
    start_year = 2022
    end_year = 2023
    load_dotenv('.env')
    API_KEY_FRED = os.environ.get('API_KEY_FRED')
    API_KEY_FMP = os.environ.get('API_KEY_FMP')
    data_extractor = company_data_extractor(API_KEY_FRED, API_KEY_FMP)

    # Macroeconomics - Federal Interest Rate (Annualized)
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&' \
          f'api_key={API_KEY_FRED}&' \
          f'file_type=json&' \
          f'observation_start={str(start_year) + "-01-01"}&observation_end={str(end_year) + "-12-31"}&' \
          f'frequency=a'
    response = requests.get(url)
    fed_interest_rates = pd.DataFrame(response.json()['observations'])['value']

    company_data = data_extractor.get_data(company, start_year, end_year, fed_interest_rates)
    company_data.drop("dps_change_next_year", axis="columns", inplace=True)
    result = pipeline.predict_proba(company_data)[1][0]
    print(f'Predicted probability for non-decreasing dividend: {result}')
except:
    print('Connection error! Please restart and try again.')
