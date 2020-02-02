# SVM
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
import os
import graph
# Logging
import logging
log = logging.getLogger()

# Plotting
matplotlib.use("macOSX")

HyperParams = {
    'kernel': ('linear', 'rbf'),
    'C': np.logspace(-6, -1, 50)
}


def runSVM(X_train, X_test, y_train, y_test, data, path):
    log.debug('Analyizing SVM')
    log.debug('Length of training set: %i' % len(X_train))
    log.debug(X_train.shape[0])

    CV = ShuffleSplit(n_splits=10, test_size=0.333, random_state=0)

    # HyperParameter Testing
    gsc = GridSearchCV(
        estimator=svm.SVC(),
        param_grid=HyperParams,
        cv=CV, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    gridResults = gsc.fit(X_train, y_train)
    bestParams = gridResults.best_params_
    bestModel = gridResults.best_estimator_
    log.info('SVM - Best Params: %s' % bestParams)

    # Learning Curve
    train_sizes, train_scores, valid_scores, fit_times, score_times = learning_curve(
        bestModel, X_train, y_train, cv=CV, return_times=True,  train_sizes=np.linspace(0.1, 1.0, 8))

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(valid_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Complexity Curve
    from sklearn.model_selection import validation_curve
    complex_train_scores, complex_valid_scores = validation_curve(
        bestModel, X_train, y_train, "C", HyperParams['C'], cv=CV)

    complex_train_scores_mean = np.mean(complex_train_scores, axis=1)
    complex_train_scores_std = np.std(complex_train_scores, axis=1)
    complex_test_scores_mean = np.mean(complex_valid_scores, axis=1)
    complex_test_scores_std = np.std(complex_valid_scores, axis=1)

    # Create subplots 
    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot learning curve
    learn_title = 'SVM - Learning Curve (kernel: %s, C: %.3f) ' % (bestParams['kernel'], bestParams['C'])
    learn_vals = (train_sizes, train_scores_mean, train_scores_std, test_scores_mean, test_scores_std)
    labels = ("Training score", "Cross-validation score", "# of samples", "Score")
    graph.plotLearningCurve(axes[0], learn_title, *learn_vals, *labels)
    
    # Plot Model-Complexity Curve
    complex_title = 'SVM - Complexity Curve (kernel: %s, C: %.3f) ' % (bestParams['kernel'], bestParams['C'])
    X = HyperParams['C']
    complex_vals = (X, complex_train_scores_mean, complex_train_scores_std, complex_test_scores_mean, complex_test_scores_std)
    labels = ("Training score", "Cross-validation score", "C Param", "Score")
    graph.plotComplexityCurve(axes[1], complex_title, *complex_vals, *labels)

    # plt.show()
    log.debug('Saving SVM.png')
    saveDir = os.path.join(path, 'SVM.png')
    plt.savefig(saveDir, bbox_inches='tight')
