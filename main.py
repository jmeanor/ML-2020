import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine
from pprint import pprint
import pandas as pd
import graphviz 

# Load Data
# data = load_iris()
data = load_wine()
# print(data.DESCR)
# pprint(data.target_names)
# pprint(data.feature_names)
# pprint(data.data)
# pprint(data.target)

# Randomly split the data into train & test sets.
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=42)
print('X_train,', len(X_train), 'X_test,', len(X_test), 'Y_train,', len(y_train), len(y_test) )

# Decision Tree
from sklearn import tree
print('Classifying Decision Trees')

# Hyperparameter
max_depth = None

clf = tree.DecisionTreeClassifier(max_depth = max_depth)
clf = clf.fit(X_train, y_train)

print(X_test, y_test)
print('Testing with: ', X_test[4], y_test[4])
print('prediction: ', clf.predict([X_test[4]]))

#  Plot Tree
# dot_data = tree.export_graphviz(clf, out_file=None, 
#                      feature_names=data.feature_names,  
#                      class_names=data.target_names,  
#                      filled=True, rounded=True,  
#                      special_characters=True)  
# graph = graphviz.Source(dot_data)
# graph 

# Decision Tree - Learning Curve 
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from sklearn.model_selection import ShuffleSplit

train_sizes, train_scores, valid_scores, fit_times, score_times = learning_curve(
    tree.DecisionTreeClassifier(max_depth = max_depth), data.data, data.target, train_sizes=np.linspace(0.1, 1.0, 5), cv= ShuffleSplit(n_splits=100, test_size=0.2, random_state=0), return_times=True)
print('Training scores:\n\n', train_scores)
print('\n', '-' * 70) # separator to make the output easy to read
print('\nValidation scores:\n\n', valid_scores)
print('\n', '-' * 20) # separator

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(valid_scores, axis=1)
test_scores_mean = np.mean(valid_scores, axis=1)
test_scores_std = np.std(valid_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

import matplotlib  
matplotlib.use("macOSX")
from matplotlib import pyplot as plt

# Plot learning curve
_, axes = plt.subplots(1, 3, figsize=(20, 5))

axes[0].grid()
axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1,
                        color="g")
axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")
axes[0].legend(loc="best")

# Plot n_samples vs fit_times
axes[1].grid()
axes[1].plot(train_sizes, fit_times_mean, 'o-')
axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                        fit_times_mean + fit_times_std, alpha=0.1)
axes[1].set_xlabel("Training examples")
axes[1].set_ylabel("fit_times")
axes[1].set_title("Scalability of the model")

# Plot fit_time vs score
axes[2].grid()
axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1)
axes[2].set_xlabel("fit_times")
axes[2].set_ylabel("Score")
axes[2].set_title("Performance of the model")
plt.show()

# Model-Complexity Curve
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
gsc = GridSearchCV(
        estimator=tree.DecisionTreeClassifier(),
        param_grid={
            'max_depth': [None, 1, 5, 10, 25, 50, 100]
        },
        cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

grid_result = gsc.fit(data.data, data.target)
best_params = grid_result.best_params_
print('Best Params: ', best_params)
# best_svr = (kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"],
#                    coef0=0.1, shrinking=True,
#                    tol=0.001, cache_size=200, verbose=False, max_iter=-1)