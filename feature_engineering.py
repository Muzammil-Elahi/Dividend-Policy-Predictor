import pandas as pd
import numpy as np
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTENC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, make_scorer
import sklearn.metrics as skm
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

# Now let's remove the features one by one from the least important one
X_train_temp = X_train_oversampled.copy()
X_validate_temp = X_validate_transformed.copy()

# Initialize the result dataframe
result_df = pd.DataFrame(columns=['Features_Removed', 'ROC_Score'])

# First, evaluate performance using all features
randomForestModel = RandomForestClassifier(max_features=None)
print('Random Forest fitting started.')
randomForestModel.fit(X_train_temp, y_train_oversampled)
# Predict probabilities on test data
y_pred_probs = randomForestModel.predict_proba(X_validate_temp)[:, 1]
# Different metrics of classification
roc_score = skm.roc_auc_score(y_test, y_pred_probs)
# f1_score = skm.f1_score(y_test, y_pred_probs, average=None)


# Append the result to the result dataframe
roc_dict = {'Features_Removed': 'None', 'no_features_used': len(X_train_temp.columns), 'ROC_Score': roc_score}
result_df = pd.DataFrame([roc_dict])
total_num = len(X_train_temp.columns)
print(f"Feature_Removed: None, Number of features used: {len(X_train_temp.columns)}, ROC_AUC_Score: {roc_score}")
result_df.to_csv("storage_files/result_df.csv",index=False)

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
    roc_score = skm.roc_auc_score(y_test, y_pred_probs)
    # Append the result to the result dataframe
    roc_dict = {'Features_Removed': row['Feature'], 'no_features_used': len(X_train_temp.columns), 'ROC_Score': roc_score}
    pd.DataFrame([roc_dict]).to_csv("storage_files/result_df.csv",mode='a',index=False,header=False)
    result_df = pd.concat([result_df, pd.DataFrame([roc_dict])])
    print(
        f"Feature_Removed: {row['Feature']}, Number of features used: {len(X_train_temp.columns)}, ROC_AUC_Score: {roc_score}")
    # If only one feature left, break the loop
    if X_train_temp.shape[1] == 1:
        break

if __name__ == '__main__':
    # Plot a bar chart to visualize ROC scores
    plt.figure(figsize=(20, 10))
    sns.barplot(data=result_df, x="no_features_used", y="ROC_Score")
    plt.title("ROC scores")
    plt.subplots_adjust(bottom=0.2, top=0.95)
    plt.xticks(rotation=45, ha='right', rotation_mode="anchor")
    plt.show()

# Find out the one with max roc_score.
max_roc_score = result_df.iloc[0]['ROC_Score']
max_inds = [0]
for index, row in result_df.iterrows():
    if row['ROC_Score'] > max_roc_score:
        max_roc_score = row['ROC_Score']
        max_inds = [index]
    elif row['ROC_Score'] == max_roc_score:
        max_inds.append(index)
max_no_features = [total_num-i for i in max_inds]
print(f'Conclusion: The best model is to use {max_no_features} features.')

# Save the results
# with open('storage_files/result_df.pkl', 'wb') as file:
#     pickle.dump(result_df, file)
# with open('storage_files/importance_table_sorted.pkl', 'wb') as file:
#     pickle.dump(importance_table_sorted, file)

# Load the results
# with open('storage_files/result_df.pkl', 'rb') as file:
#     result_df = pickle.load(file)


