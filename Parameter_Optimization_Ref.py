# Bayesian Optimization

from skopt import BayesSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define the model
model = RandomForestClassifier()

# Define the parameter search space
search_spaces = {
    'n_estimators': (10, 200),
    'max_depth': (10, 50),
    'min_samples_split': (2, 10)
}

# n_estimators=100: The number of trees in the forest.
# criterion='gini': The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity and "entropy" for the information gain.
# max_depth=None: The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
# min_samples_split=2: The minimum number of samples required to split an internal node.
# min_samples_leaf=1: The minimum number of samples required to be at a leaf node.
# max_features='auto': The number of features to consider when looking for the best split.
# bootstrap=True: Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.
# random_state=None: Controls the randomness of the estimator.


# Initialize BayesSearchCV
bayes_search = BayesSearchCV(estimator=model, search_spaces=search_spaces, n_iter=32, cv=5, scoring='accuracy', random_state=42)

# Fit the Bayesian search
bayes_search.fit(X_train, y_train)

# Get the best parameters
best_params = bayes_search.best_params_

print("Best Parameters:", best_params)
