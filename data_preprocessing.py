import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer

# Register API for Financial Modeling Prep (Financial Statements and Company Fundamentals)
# https://site.financialmodelingprep.com/developer/
# Register API for Federal Reserve Economic Data (For Macroeconomics Data)
# https://fred.stlouisfed.org/docs/api/fred/
# Yahoo Finance does not need an API

warnings.filterwarnings('ignore')

# Load Data
dataset = pd.read_csv("storage_files/Stock_data.csv")

# Null value analysis
print('Import original dataset.')
# dataset.info(verbose=True)

dataset.isna().sum()

enc = OrdinalEncoder()
df = dataset.copy() #create copy df before one hot encoding
df[["industry","sector","symbol"]] = enc.fit_transform(df[["industry","sector","symbol"]])
df.head()

# Multivariate Analysis
# target = dataset["dps_change_next_year"]
df.drop("dps_change_next_year", axis="columns", inplace=True)
# Correlation matrix
correlation_matrix = df.corr()

def rank_columns_by_correlation(df, threshold=0.9):
    # Calculating correlation matrix
    corr_matrix = df.corr()
    # Initializing a list to hold the tuples (col1, col2, correlation)
    correlations = []
    # Iterating over the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):  # avoiding duplicate and self-correlation
            # Including only correlations above the specified threshold
            if abs(corr_matrix.iloc[i, j]) > threshold:
                correlations.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))

    # Sorting the list by absolute correlation in descending order
    sorted_correlations = sorted(correlations, key=lambda x: abs(x[2]), reverse=True)
    correlation_df = pd.DataFrame(sorted_correlations, columns=['Column1', 'Column2', 'Correlation'])
    return correlation_df

top_correlations = rank_columns_by_correlation(df, 0.98)

# Remove highly correlated columns
columns_to_remove = top_correlations["Column2"].unique()
dataset.drop(columns_to_remove, axis="columns", inplace=True)
print('Non-categorical variables:')
print(columns_to_remove)

# Data Preprocessing
# Missing value
# dataset.info(verbose=True)
dataset.dropna(inplace=True)
dataset.head()

# First let's leave out the last year's data as future test data, and 2021's data as validation data
training_data = dataset.loc[(dataset["year"] != 2022) & (dataset["year"] != 2021)]
validation_data = dataset.loc[dataset["year"] == 2021]
testing_data = dataset.loc[dataset["year"] == 2022]
print('Target split finished.')

# Predictor - Target Split
y_train = training_data["dps_change_next_year"]
X_train = training_data.drop("dps_change_next_year", axis="columns")
y_test = testing_data["dps_change_next_year"]
X_test = testing_data.drop("dps_change_next_year", axis="columns")
y_validate = validation_data["dps_change_next_year"]
X_validate = validation_data.drop("dps_change_next_year", axis="columns")

# Label encode categorical features with many categories
categorical_columns = ["industry","sector","symbol"]
other_columns = [col for col in X_train.columns if col not in categorical_columns]

# Column Transformer
column_transformer = ColumnTransformer(
    transformers=[
        ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_columns)
    ],
    remainder='passthrough'
)

X_train_transformed = column_transformer.fit_transform(X_train)
X_validate_transformed = column_transformer.transform(X_validate)
X_test_transformed = column_transformer.transform(X_test)

# Note: after transformation, the output will be a numpy array and column orders will be changed.
X_train_transformed = pd.DataFrame(X_train_transformed, columns=categorical_columns + other_columns)
X_validate_transformed = pd.DataFrame(X_validate_transformed, columns=categorical_columns + other_columns)
X_test_transformed = pd.DataFrame(X_test_transformed, columns=categorical_columns + other_columns)

# Check our data type
# X_train_transformed.info(verbose=True)
print('Categorical columns transformed.')

# Let's change our data types back to their original forms - However, this time, categorical variables have become
# number like strings
cols_to_convert = {'industry': 'str', 'sector': 'str', 'symbol': 'str', 'year': 'int'}
X_train = X_train.astype(cols_to_convert)
X_validate = X_validate.astype(cols_to_convert)
X_test = X_test.astype(cols_to_convert)

# Check data imbalance
# Let's add target back to our dataset for further analysis
training_data_transformed = pd.concat([X_train, y_train], axis=1)
training_data_transformed["dps_change_next_year"].value_counts()

# Let's do some over sampling
# Perform oversampling using SMOTE
categorical_indices = [X_train.columns.get_loc(col) for col in categorical_columns]
smote = SMOTENC(random_state=1, categorical_features=categorical_indices)

X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train_transformed, y_train)
# Check our training data
pd.DataFrame(y_train_oversampled)["dps_change_next_year"].value_counts()
# X_train_oversampled.info()
print('Oversampling finished.')

# Feature selection
print('Feature selection started.')
# Feature importance analysis - Tree Based
randomForestModel = RandomForestClassifier(max_features=None)  # We want all features to be considered for each tree
randomForestModel.fit(X_train_oversampled, y_train_oversampled)
model_importance = randomForestModel.feature_importances_
importance_table = pd.DataFrame(columns=["Feature", "Importance"])  # Create an importance table to plot bar chart
featureNum = 0
for score in model_importance:
    print("feature " + str(featureNum) + "'s importance score: " + str(score) + " (" + X_train_oversampled.columns[featureNum] + ")")
    rowAdded = pd.DataFrame([[X_train_oversampled.columns[featureNum], score]], columns=["Feature", "Importance"])
    importance_table = pd.concat([importance_table, rowAdded])
    featureNum = featureNum + 1
importance_table.sort_values('Importance', inplace=True, ascending=False)

if __name__ == '__main__':
    # Plot a bar chart to visualize feature importance
    plt.figure(figsize=(20, 10))
    sns.barplot(data=importance_table, x="Feature", y="Importance")
    plt.title("Feature Importance")
    plt.subplots_adjust(bottom=0.2, top=0.95)
    plt.xticks(rotation=45, ha='right', rotation_mode="anchor")
    plt.show()

# Export files.
X_train_oversampled.to_csv("storage_files/X_train_oversampled.csv",index=False)
X_validate_transformed.to_csv("storage_files/X_validate_transformed.csv",index=False)
X_test_transformed.to_csv("storage_files/X_test_transformed.csv",index=False)
y_train_oversampled.to_csv("storage_files/y_train_oversampled.csv",index=False)
y_validate.to_csv("storage_files/y_validate.csv",index=False)
y_test.to_csv("storage_files/y_test.csv",index=False)
importance_table.to_csv("storage_files/importance_table.csv",index=False)
