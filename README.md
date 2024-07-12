# Dividend Policy Predictor 

## Project description
This project aims to predict a given company's dividend policy in the coming year: whether or not the company would cut dividends. We use companies' historical data and machine learning models to discover the relationship among the dividend policy and other indicators of the company (such as dividend per share, current ratio, industry to which the company belongs, etc.).
## Data usage and acknowledgement
This is a collaborative project among Muzammil Elahi, Eric Lu and Rosalie Huang.
We retrieved data from Financial Modeling Prep, Federal Reserve Economic Data (FRED), Wikipedia and Yahoo Finance.
## Installation and execution
1. `git clone https://github.com/Muzammil-Elahi/Dividend-Policy-Predictor`
2. Register and obtain API Keys from https://financialmodelingprep.com/ and https://fred.stlouisfed.org/. Store them in strings in `.env` with names `API_KEY_FMP` and `API_KEY_FRED`.

For the following files, we recommend to run with the following order one by one. To save time, you can also simply run the last file, but the outcome would be based on outdated data.

3. `database_creation.py`
4. `data_preprocessing.py`
5. `feature_engineering.py`
6. `model_comparison.py`
7. `pipeline.py`: for company tickers, check "Symbol" in the table under https://en.wikipedia.org/wiki/List_of_S%26P_500_companies

## Project architecture
<img width="671" alt="Screen Shot 2024-05-25 at 10 29 26 AM" src="https://github.com/Muzammil-Elahi/Dividend-Policy-Predictor/assets/130340711/cb2c7b30-364c-4239-9dac-e821bf3796a1">

### Data extraction and transformation
In `database_creation.py`, we collected data entries for S&P 500 companies and their financial performances each year. For each company and each year, there are a lot of features describing the financial condition of the company, including categorical variables such as industry and sector, and numerical variables such as sales, net income and dividend per share. Our target variable is the change on dividend per share, which is a binary variable: decreasing or non-decreasing.

To transform the data, in `data_preprocessing.py` we calculated the **correlation matrix** for all features and removed those highly correlated features. We also did **categorical feature encoding**. After this step, there are 92 features left.

We split the data into train, validation and test data. Due to the fact that most S&P 500 companies are well performing and don’t have decreasing dividend per share, we implemented **oversampling** on the train data to tackle with data imbalance.

### Feature and model selection
After running `data_preprocessing.py` you would see a graph showing the importance scores of all the features, sorted in descending order.

In `feature_engineering.py` we used **Best Subset Method** to figure out the optimal combination of features by starting with all features and removing the least important feature one by one. The optimal numbers of features are stored in 'storage_files/max_no_features.txt'. In our execution, the optimal number was 36. This could vary with latest data.

In `Dividend_Policy_Predictor.ipynb` you can see that we ran several ML models (Logistic Regression, Decision Tree, K-Nearest Neighbor, Random Forest and XGBoost) with a total of 92 features and that the Random Forest model yields the best result (taking ROC-AUC score as the metric). This was also true with our experiment with 36 models, 9 models and 8 models. So we chose Random Forest Model as our training model.

The result in `model_comparison.py` also supports that 36-feature model is the best one. Since RF models yields different results at different times, we ran the model 100 times and saved the best one in `model_comparison.py`. 

### Pipeline
The interactive program in `pipeline.py` calculates the probability for non-decreasing dividend using the model saved in last step. A probability >0.5 shows that the company is likely to increase (or maintain) dividend in the next year.
