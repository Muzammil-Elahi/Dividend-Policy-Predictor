# These are module level imports and codes
import requests
import pandas as pd
import numpy as np
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

# Define our data extractor class
class company_data_extractor:
    # class level global variables
    BASE_URL = 'https://financialmodelingprep.com/api/v3'

    # Constructor
    def __init__(self, API_KEY_FRED, API_KEY_FMP):
        self.API_KEY_FRED = API_KEY_FRED
        self.API_KEY_FMP = API_KEY_FMP

    def get_data(self, company_tick, start_year, end_year, fed_interest_rates):

        # num_of_years = end_year - start_year + 1 + 2
        start_year = start_year - 1
        end_year = end_year + 1

        # Engineer Dividend Per Share (DPS) data (Dividend related predictors and the target variable)
        try:
            response = requests.get(f"{self.BASE_URL}/historical-price-full/stock_dividend/{company_tick}?apikey={self.API_KEY_FMP}")
            if response.status_code == 429:
                print("FMP API limit reached at get_data")
                return 0
        except:  # requests.exceptions.ConnectionError or urllib3.exceptions.ProtocolError
            print("Connection error occurred at get_data")
            return 0
        dividends = pd.DataFrame(response.json()['historical'])

        if dividends.shape == (0, 0):
            dividends = pd.DataFrame({
                "year": list(range(start_year, end_year + 1)),
                "adjDividend": [0.0] * len(list(range(start_year, end_year + 1)))
            })
        else:
            dividends['year'] = pd.to_datetime(dividends['date']).dt.year
            dividends = dividends.groupby("year").agg({"adjDividend": "sum"}).reset_index()
            # Create a new DataFrame with all years from start to end
            all_years = pd.DataFrame({'year': list(range(start_year - 1, end_year + 1))})
            # Merge the two DataFrames on the year column and fill missing values with 0.0
            dividends = all_years.merge(dividends, on='year', how='left').fillna(0.0)

        dividends['next_year_dividend'] = dividends['adjDividend'].shift(-1)
        dividends['last_year_dividend'] = dividends['adjDividend'].shift(1)

        conditions = [
            dividends['adjDividend'] <= dividends['next_year_dividend'],
            dividends['adjDividend'] > dividends['next_year_dividend']
        ]

        choices = ['constant/increased', 'decreased']

        dividends['dps_growth'] = dividends['adjDividend'] - dividends['last_year_dividend']

        # Another predictor that we can create is dividend change as a percentage
        dividends['dps_growth_rate'] = np.where(
            (dividends['last_year_dividend'] == 0) & (dividends['adjDividend'] == 0),
            0,  # If both are 0 then change is 0
            np.where(
                dividends['last_year_dividend'] != 0,
                ((dividends['adjDividend'] / dividends['last_year_dividend']) - 1) * 100,
                999  # If last year dividend is 0 then return 999
            )
        )
        # Create the target column 'dps_change' based on the conditions
        dividends['dps_change_next_year'] = np.select(conditions, choices, default=np.nan)
        # Remove the first and last year since they will be NaN
        dividends = dividends.loc[(dividends['year'] > start_year) & (dividends['year'] <= end_year - 1)]
        dividends = dividends[["year", "adjDividend", "dps_growth", "dps_growth_rate", "dps_change_next_year"]]

        # Engineer Other Predictors
        predictors = pd.DataFrame({"year": list(range(start_year, end_year + 1))})

        # Company's Industry
        predictors["industry"] = yf.Ticker(company_tick).info.get('industry')
        predictors["sector"] = yf.Ticker(company_tick).info.get('sector')

        # Key Financial Ratios
        num_of_years = 2024 - start_year - 1
        try:
            response = requests.get(f"{self.BASE_URL}/ratios/{company_tick}?limit={num_of_years}&apikey={self.API_KEY_FMP}")
            if response.status_code == 429:
                print("FMP API limit reached at all-year-data")
                return 0
        except:  # requests.exceptions.ConnectionError or urllib3.exceptions.ProtocolError
            print("Connection error occurred at all-year-data")
            return 0
        # Check if all data is available
        data_length = len(response.json())
        if data_length != num_of_years:
            print(f"Company {company_tick} data is not available")
            return 0
        financial_ratios = pd.DataFrame(response.json()).iloc[:, :].sort_values("date", ascending=True).reset_index(
            drop=True)
        financial_ratios['calendarYear'] = financial_ratios['calendarYear'].astype('int64')
        predictors = predictors.merge(financial_ratios, left_on='year', right_on='calendarYear', how='left').fillna(0.0)
        predictors.drop(["date", "calendarYear", "period"], axis="columns", inplace=True)

        predictors['interestRate'] = fed_interest_rates.astype("float64")

        predictor_names = list(predictors.columns)
        predictor_names.remove("year")
        predictor_names.remove("industry")
        predictor_names.remove("sector")
        predictor_names.remove("symbol")

        def compute_change(df, predictor_list):
            for predictor in predictor_list:
                # Calculate percentage change
                percentage_change = df[predictor].pct_change() * 100
                # Create new column name
                new_col_name = f"{predictor}_percentage_change"
                # Find the index position of the original predictor column
                original_col_position = df.columns.get_loc(predictor)
                # Insert the new column right after the original predictor column
                df.insert(original_col_position + 1, new_col_name, percentage_change)
            # Replacing inf and NaN values
            df.replace([float('inf'), float('-inf')], 999, inplace=True)
            df.fillna(0, inplace=True)
            return df

        predictors = compute_change(predictors, predictor_names)

        # Combine dividend data with other predictors
        dataset = pd.merge(predictors, dividends, left_on='year', right_on='year', how='left')

        # Drop first and last row as they contain nan
        last_row = len(dataset) - 1
        dataset.drop([0, last_row], axis="rows", inplace=True)
        dataset.reset_index(drop=True, inplace=True)

        return dataset
