# SVM
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
import os
# Logging
import logging
log = logging.getLogger()

# Plotting
import matplotlib  
matplotlib.use("macOSX")
from matplotlib import pyplot as plt

HyperParams = {
    'kernel':('linear', 'rbf'),
    'C': np.logspace(-6, -1, 50)
}

def runSVM(X_train, X_test, y_train, y_test, data, path):
    log.debug('Analyizing SVM')
    log.debug('Length of training set: %i' %len(X_train))
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
        bestModel, X_train, y_train, cv=CV, return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(valid_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Complexity Curve
    from sklearn.model_selection import validation_curve
    complex_train_scores, complex_valid_scores = validation_curve(bestModel, X_train, y_train, "C", HyperParams['C'], cv=CV)

    complex_train_scores_mean   = np.mean(complex_train_scores, axis=1)
    complex_train_scores_std    = np.std(complex_train_scores, axis=1)
    complex_test_scores_mean    = np.mean(complex_valid_scores, axis=1)
    complex_test_scores_std     = np.std(complex_valid_scores, axis=1)

    # Plot learning curve
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].grid()
    axes[0].set_title('SVM - Learning Curve (kernel: %s, C: %.3f) ' %(bestParams['kernel'], bestParams['C']))
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
    axes[0].set_ylim((0, 1.1))
    axes[0].set_xlabel("# of samples")
    axes[0].set_ylabel("Score")
    axes[0].legend(loc="best")

    # Plot Model-Complexity Curve
    cVals = HyperParams['C']    
    axes[1].grid()

    axes[1].set_title('SVM - Complexity Curve (kernel: %s, C: %.3f) ' %(bestParams['kernel'], bestParams['C']))

    axes[1].fill_between(cVals, complex_train_scores_mean - complex_train_scores_std,
                            complex_train_scores_mean + complex_train_scores_std, alpha=0.1,
                            color="b")
    axes[1].fill_between(cVals, complex_test_scores_mean - complex_test_scores_std,
                            complex_test_scores_mean + complex_test_scores_std, alpha=0.1,
                            color="m")
    axes[1].plot(cVals, complex_train_scores_mean, 'o-', color="b",
                    label="Training score")
    axes[1].plot(cVals, complex_test_scores_mean, 'o-', color="m",
                    label="Cross-validation score")
    axes[1].set_ylim((0, 1.1))
    axes[1].set_xlabel("C Param")
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="best")

    
    # plt.show()
    log.debug('Saving SVM.png')
    saveDir = os.path.join( path, 'SVM.png')
    plt.savefig(saveDir, bbox_inches='tight')
