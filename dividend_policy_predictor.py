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
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score, GridSearchCV
import pickle
import optuna
from xgboost import XGBClassifier

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
print('Categorical variables:')
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
# Plot a bar chart to visualize feature importance
plt.figure(figsize=(20, 10))
sns.barplot(data=importance_table, x="Feature", y="Importance")
plt.title("Feature Importance")
plt.subplots_adjust(bottom=0.2, top=0.95)
plt.xticks(rotation=45, ha='right', rotation_mode="anchor")
plt.show()

# ============================================================================


# Now let's remove the features one by one from the least important one
X_train_temp = X_train_oversampled.copy()
X_validate_temp = X_validate_transformed.copy()

# Initialize the result dataframe
result_df = pd.DataFrame(columns=['Features_Removed', 'ROC_Score'])

# First, evaluate performance using all features
randomForestModel = RandomForestClassifier(max_features=None)
randomForestModel.fit(X_train_temp, y_train_oversampled)
# Predict probabilities on test data
y_pred_probs = randomForestModel.predict_proba(X_validate_temp)[:, 1]
# Compute ROC score
roc_score = roc_auc_score(y_test, y_pred_probs)
# Append the result to the result dataframe
roc_dict = {'Features_Removed': 'None', 'ROC_Score': roc_score}
result_df = pd.DataFrame([roc_dict])
print(f"Feature_Removed: None, Number of features used: {len(X_train_temp.columns)}, ROC_AUC_Score: {roc_score}")

# Sort importance_table by Importance in ascending order to start with the least important
importance_table_sorted = importance_table.sort_values('Importance')
# Loop through features, starting from the least important
for index, row in importance_table_sorted.iterrows():
    # Drop the feature from training and test data
    X_train_temp = X_train_temp.drop(columns=[row['Feature']])
    X_validate_temp = X_validate_temp.drop(columns=[row['Feature']])
    # Train a random forest model
    randomForestModel = RandomForestClassifier(max_features=None)
    randomForestModel.fit(X_train_temp, y_train_oversampled)
    # Predict probabilities on test data
    y_pred_probs = randomForestModel.predict_proba(X_validate_temp)[:, 1]
    # Compute ROC score
    roc_score = roc_auc_score(y_test, y_pred_probs)
    # Append the result to the result dataframe
    roc_dict = {'Features_Removed': row['Feature'], 'ROC_Score': roc_score}
    result_df = pd.DataFrame([roc_dict])
    print(
        f"Feature_Removed: {row['Feature']}, Number of features used: {len(X_train_temp.columns)}, ROC_AUC_Score: {roc_score}")
    # If only one feature left, break the loop
    if X_train_temp.shape[1] == 1:
        break

# Save the results
with open('result_df.pkl', 'wb') as file:
    pickle.dump(result_df, file)
with open('importance_table_sorted.pkl', 'wb') as file:
    pickle.dump(importance_table_sorted, file)

# Load the results
with open('result_df.pkl', 'rb') as file:
    result_df = pickle.load(file)

# Model Selection

X_train_oversampled.info()
model_lr = LogisticRegression()
model_lr.fit(X_train_oversampled, y_train_oversampled)

# GridSearch
param_grid = {
    "penalty": ['l1', 'l2'],  # These have to be the same as the estimator's parameters' name
    "C": np.arange(0.1, 10, 0.1).tolist()
}
gridSearch = GridSearchCV(estimator=LogisticRegression(random_state=1), param_grid=param_grid, scoring='roc_auc',
                          cv=5, n_jobs=-1)
gridSearch.fit(X_train_oversampled, y_train_oversampled)

best_params_lr = gridSearch.best_params_
print("Best Parameters: ", best_params_lr)
print("Best ROC-AUC Score: ", gridSearch.best_score_)

# Create and save model
best_model_lr = LogisticRegression(**best_params_lr, solver='liblinear', n_jobs=-1)
with open('best_models_lr.pkl', 'wb') as file:
    pickle.dump(best_model_lr, file)


# Bayesian Optimization with optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress log messages

# Logistic Regression
def objective_function(trial):
    C = trial.suggest_float('C', 0.1, 10, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])

    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver='liblinear',
        n_jobs=-1
    )

    # Using cross_val_score to get the average precision score for each fold
    scores = cross_val_score(model, X_train_oversampled, y_train_oversampled, cv=5, scoring='roc_auc')
    roc_auc = np.mean(scores)
    # Printing intermediate results
    print(f"Trial {trial.number}, C: {C}, penalty: {penalty}, ROC-AUC: {roc_auc}")
    return roc_auc


study_lr = optuna.create_study(direction="maximize")
study_lr.optimize(objective_function, n_trials=100)

best_params_lr = study_lr.best_params
print("Best Parameters: ", best_params_lr)
print("Best ROC-AUC Score: ", study_lr.best_value)

# Create and save model
best_model_lr = LogisticRegression(**best_params_lr, solver='liblinear', n_jobs=-1)
with open('best_models_lr.pkl', 'wb') as file:
    pickle.dump(best_model_lr, file)

# Decision Tree
def objective_function(trial):
    max_depth = trial.suggest_int('max_depth', 1, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion
    )

    # Using cross_val_score to get the average precision score for each fold
    scores = cross_val_score(model, X_train_oversampled, y_train_oversampled, cv=5, scoring='roc_auc')
    roc_auc = np.mean(scores)
    # Printing intermediate results
    print(f"Trial {trial.number}, max_depth: {max_depth}, min_samples_split: {min_samples_split}, "
          f"min_samples_leaf: {min_samples_leaf}, criterion: {criterion}, ROC-AUC: {roc_auc}")
    return roc_auc

study_dt = optuna.create_study(direction="maximize")
study_dt.optimize(objective_function, n_trials=100)

best_params_dt = study_dt.best_params
print("Best Parameters: ", best_params_dt)
print("Best ROC-AUC Score: ", study_dt.best_value)

# Create and save model
best_model_dt = DecisionTreeClassifier(**best_params_dt)
with open('best_models_dt.pkl', 'wb') as file:
    pickle.dump(best_model_dt, file)

# KNN
def objective_function(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p = trial.suggest_int('p', 1, 5)
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
        metric=metric
    )

    # Using cross_val_score to get the average precision score for each fold
    scores = cross_val_score(model, X_train_oversampled, y_train_oversampled, cv=5, scoring='roc_auc')
    roc_auc = np.mean(scores)
    # Printing intermediate results
    print(f"Trial {trial.number}, n_neighbors: {n_neighbors}, weights: {weights}, p: {p}, metric: {metric}, "
          f"ROC-AUC: {roc_auc}")
    return roc_auc


study_knn = optuna.create_study(direction="maximize")
study_knn.optimize(objective_function, n_trials=100)

best_params_knn = study_knn.best_params
print("Best Parameters: ", best_params_knn)
print("Best ROC-AUC Score: ", study_knn.best_value)

# Create and save model
best_model_knn = KNeighborsClassifier(**best_params_knn)
with open('best_models_knn.pkl', 'wb') as file:
    pickle.dump(best_model_knn, file)

# Random Forest
def objective_function(trial):
    n_estimators = trial.suggest_int('n_estimators', 2, 150)
    max_depth = trial.suggest_int('max_depth', 1, 50)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 15)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 15)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=-1
    )

    # Using cross_val_score to get the average ROC-AUC score for each fold
    scores = cross_val_score(model, X_train_oversampled, y_train_oversampled, cv=5, scoring='roc_auc')
    roc_auc = np.mean(scores)
    # Printing intermediate results
    print(f"Trial {trial.number}, n_estimators: {n_estimators}, max_depth: {max_depth}, "
          f"min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, ROC-AUC: {roc_auc}")
    return roc_auc


study_rf = optuna.create_study(direction="maximize")
study_rf.optimize(objective_function, n_trials=100)

best_params_rf = study_rf.best_params
print("Best Parameters: ", best_params_rf)
print("Best ROC-AUC: Score: ", study_rf.best_value)

# Create and save model
best_model_rf = RandomForestClassifier(**best_params_rf, n_jobs=-1)
with open('best_models_rf.pkl', 'wb') as file:
    pickle.dump(best_model_rf, file)

# XgBoost
# It requires the target to be 0 and 1, and all features be numerical
# Encode our target
label_encoder = LabelEncoder()
# Fit the encoder and transform the target variable
y_train_oversampled_encoded = label_encoder.fit_transform(y_train_oversampled)

# Cast categorical types into numbers
cols_to_convert = {'industry': 'float', 'sector':'float', 'symbol': 'float'}
X_train_oversampled = X_train_oversampled.astype(cols_to_convert)
X_test_transformed = X_test_transformed.astype(cols_to_convert)
# This suppresses printing logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_function(trial):
    n_estimators = trial.suggest_int('n_estimators', 2, 150)
    max_depth = trial.suggest_int('max_depth', 1, 50)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.9, log=True)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 10)
    subsample = trial.suggest_float('subsample', 0.5, 1.0)
    colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)
    gamma = trial.suggest_float('gamma', 0, 1.0)
    reg_alpha = trial.suggest_float('reg_alpha', 0, 1)
    reg_lambda = trial.suggest_float('reg_lambda', 0, 1)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        gamma=gamma,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        use_label_encoder=False,
        n_jobs=-1
    )

    # Using cross_val_score to get the average ROC-AUC score for each fold
    scores = cross_val_score(model, X_train_oversampled, y_train_oversampled_encoded, cv=5, scoring='roc_auc')
    roc_auc = np.mean(scores)
    # Printing intermediate results
    print(f"Trial {trial.number}, n_estimators: {n_estimators}, max_depth: {max_depth}, learning_rate: {learning_rate},"
          f"min_child_weight: {min_child_weight}, subsample: {subsample}, colsample_bytree: {colsample_bytree}, "
          f"gamma: {gamma}, reg_alpha: {reg_alpha}, reg_lambda: {reg_lambda}, ROC-AUC: {roc_auc}")
    return roc_auc


study_xgb = optuna.create_study(direction="maximize")
study_xgb.optimize(objective_function, n_trials=100)

best_params_xgb = study_xgb.best_params
print("Best Parameters: ", best_params_xgb)
print("Best ROC-AUC Score: ", study_xgb.best_value)

best_model_xgb = XGBClassifier(**best_params_xgb, use_label_encoder=False, n_jobs=-1)
with open('best_models_xgb.pkl', 'wb') as file:
    pickle.dump(best_model_xgb, file)

# Label encode categorical features with many categories
# column_transformer = ColumnTransformer(
#     transformers=[
#         ('categorical', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_columns)
#     ],
#     remainder='passthrough'
# )
# y = tmp[]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model selection - Compare Performance
with open('best_models_lr.pkl', 'rb') as file:
    best_model_lr = pickle.load(file)
with open('best_models_dt.pkl', 'rb') as file:
    best_model_dt = pickle.load(file)
with open('best_models_knn.pkl', 'rb') as file:
    best_model_knn = pickle.load(file)
with open('best_models_rf.pkl', 'rb') as file:
    best_model_rf = pickle.load(file)
with open('best_models_xgb.pkl', 'rb') as file:
    best_model_xgb = pickle.load(file)

print("Testing Performances...Please wait")
best_model_lr.fit(X_train_oversampled, y_train_oversampled)
predicted_probs = best_model_lr.predict_proba(X_test_transformed)[:, 1]
lr_performance = roc_auc_score(y_test, predicted_probs)

best_model_dt.fit(X_train_oversampled, y_train_oversampled)
predicted_probs = best_model_dt.predict_proba(X_test_transformed)[:, 1]
dt_performance = roc_auc_score(y_test, predicted_probs)

best_model_knn.fit(X_train_oversampled, y_train_oversampled)
predicted_probs = best_model_knn.predict_proba(X_test_transformed)[:, 1]
knn_performance = roc_auc_score(y_test, predicted_probs)

best_model_rf.fit(X_train_oversampled, y_train_oversampled)
predicted_probs = best_model_rf.predict_proba(X_test_transformed)[:, 1]
rf_performance = roc_auc_score(y_test, predicted_probs)

best_model_xgb.fit(X_train_oversampled, y_train_oversampled_encoded)
predicted_probs = best_model_xgb.predict_proba(X_test_transformed)[:, 1]
xgb_performance = roc_auc_score(y_test, predicted_probs)

# Test performance of the models are
print(f"Logistic Regression Test ROCAUC: {lr_performance}")
print(f"Decision Tree Test ROCAUC: {dt_performance}")
print(f"KNN Test ROCAUC: {knn_performance}")
print(f"Random Forest Test ROCAUC: {rf_performance}")
print(f"XGBoost Test ROCAUC: {xgb_performance}")

# Build the final pipeline for production
pipeline = Pipeline(steps=[('preprocessor', column_transformer),
                           ('classifier', best_model_rf)
                          ])

result = pipeline.predict(X_test)
