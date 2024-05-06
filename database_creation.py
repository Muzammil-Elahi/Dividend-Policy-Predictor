import requests
import pandas as pd
import numpy as np
import os
import csv
from dotenv import load_dotenv
import warnings
import matplotlib.pyplot as plt
import pickle
from company_data_extractor import company_data_extractor


# Register API for Financial Modeling Prep (Financial Statements and Company Fundamentals)
# https://site.financialmodelingprep.com/developer/
# Register API for Federal Reserve Economic Data (For Macroeconomics Data)
# https://fred.stlouisfed.org/docs/api/fred/
# Yahoo Finance does not need an API

warnings.filterwarnings('ignore')


# -------------------------------- Do not run this part if Dataset is already available --------------------------------

load_dotenv('.env')
API_KEY_FRED = os.environ.get('API_KEY_FRED')
API_KEY_FMP = os.environ.get('API_KEY_FMP')

start_year = 2012
end_year = 2022

# Scrap sp500 tickers using pandas datareader
tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
ticker_table = tables[0]
tickers = ticker_table['Symbol'].tolist()

# tickers = tickers[0:10]

print('Number of companies: ', len(tickers))

# Macroeconomics - Federal Interest Rate (Annualized)
url = f'https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&' \
      f'api_key={API_KEY_FRED}&' \
      f'file_type=json&' \
      f'observation_start={str(start_year) + "-01-01"}&observation_end={str(end_year) + "-12-31"}&' \
      f'frequency=a'

response = requests.get(url)
fed_interest_rates = pd.DataFrame(response.json()['observations'])['value']
print('Macro data retrieved.')

# Obtain our dataset
data_extractor = company_data_extractor(API_KEY_FRED, API_KEY_FMP)
dataset = []
ticker = tickers[0]
print(f"1: Obtaining data for {ticker}")
company_data = data_extractor.get_data(ticker, start_year, end_year, fed_interest_rates)
if type(company_data).__name__ != "int":
    dataset.append(company_data)
    company_data.to_csv("Stock_data.csv",index=False)
company_number = 2
for ticker in tickers[1:]:
    print(f"{company_number}: Obtaining data for {ticker}")
    company_data = data_extractor.get_data(ticker, start_year, end_year, fed_interest_rates)
    if type(company_data).__name__ == "int":
        continue
    # dataset.append(company_data)
    company_data.to_csv("Stock_data.csv",mode='a',index=False,header=False)
    company_number = company_number + 1
# dataset = pd.concat(dataset, ignore_index=True)


# Save data to disk
# dataset.to_csv("Stock_data.csv", index=False)

