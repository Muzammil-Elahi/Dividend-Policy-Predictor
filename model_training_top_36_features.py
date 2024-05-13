import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, make_scorer
# import sklearn.metrics as skm
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

warnings.filterwarnings('ignore')

# Import the files from data_preprocessing.
X_train_oversampled = pd.read_csv("storage_files/X_train_oversampled.csv")
X_validate_transformed = pd.read_csv("storage_files/X_validate_transformed.csv")
X_test_transformed = pd.read_csv("storage_files/X_test_transformed.csv")
y_train_oversampled = pd.read_csv("storage_files/y_train_oversampled.csv")
y_validate = pd.read_csv("storage_files/y_validate.csv")
y_test = pd.read_csv("storage_files/y_test.csv")
importance_table = pd.read_csv("storage_files/importance_table.csv")

f = open('storage_files/max_no_features.txt','r')
max_no_features = int(f.readline())
f.close()
max_features = list(importance_table[:max_no_features]['Feature'])
X_train_oversampled = X_train_oversampled[max_features]
X_validate_transformed = X_validate_transformed[max_features]
X_test_transformed = X_test_transformed[max_features]

# Logistic Regression
print('\nLogistic Regression started.')
def objective_function(trial):
    C = trial.suggest_float('C', 0.1, 10, log=True)
    penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])

    model = LogisticRegression(
        C=C,
        penalty=penalty,
        solver='liblinear',
        n_jobs=-1
    )
    metrics = ['roc_auc','accuracy','precison','recall','f1-score']
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
with open('storage_files/best_models_lr_36_features.pkl', 'wb') as file:
    pickle.dump(best_model_lr, file)

# Decision Tree
print('\nDecision Tree started.')
def objective_function(trial):
    max_depth = trial.suggest_int('max_depth', 1, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion
    )
    metrics = ['roc_auc','accuracy','precison','recall','f1-score']
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
with open('storage_files/best_models_dt_36_features.pkl', 'wb') as file:
    pickle.dump(best_model_dt, file)

# KNN
print('\nKNN started')
def objective_function(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 1, 10)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p = trial.suggest_int('p', 1, 5)
    metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski'])

    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p,
        metric=metric
    )
    metrics = ['roc_auc','accuracy','precison','recall','f1-score']
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
with open('storage_files/best_models_knn_36_features.pkl', 'wb') as file:
    pickle.dump(best_model_knn, file)

# Random Forest
print('\nRandom Forest started')
def objective_function(trial):
    n_estimators = trial.suggest_int('n_estimators', 2, 50)
    max_depth = trial.suggest_int('max_depth', 1, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

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
with open('storage_files/best_models_rf_36_features.pkl', 'wb') as file:
    pickle.dump(best_model_rf, file)

label_encoder = LabelEncoder()
# Fit the encoder and transform the target variable
y_train_oversampled_encoded = label_encoder.fit_transform(y_train_oversampled)
# This suppresses printing logs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# XGBoost
print('\nXGBoost started')
def objective_function(trial):
    n_estimators = trial.suggest_int('n_estimators', 2, 50)
    max_depth = trial.suggest_int('max_depth', 1, 10)
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
with open('storage_files/best_models_xgb_36_features.pkl', 'wb') as file:
    pickle.dump(best_model_xgb, file)

# Model selection - Compare Performance
with open('storage_files/best_models_lr_36_features.pkl', 'rb') as file:
    best_model_lr = pickle.load(file)
with open('storage_files/best_models_dt_36_features.pkl', 'rb') as file:
    best_model_dt = pickle.load(file)
with open('storage_files/best_models_knn_36_features.pkl', 'rb') as file:
    best_model_knn = pickle.load(file)
with open('storage_files/best_models_rf_36_features.pkl', 'rb') as file:
    best_model_rf = pickle.load(file)
with open('storage_files/best_models_xgb_36_features.pkl', 'rb') as file:
    best_model_xgb = pickle.load(file)

print("\nTesting Performances...Please wait")
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
