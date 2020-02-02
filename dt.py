
# Decision Tree
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
import os
import math
# Plotting
import matplotlib  
matplotlib.use("macOSX")
from matplotlib import pyplot as plt
from pprint import pprint
# Logging
import logging
log = logging.getLogger()

HyperParams = {
    'max_depth': [None, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 175, 200, 250, 300],
    'ccp_alpha': np.linspace(0.0, 5.0, 25)
    # 'ccp_alpha': [0.0]
}
ComplexityParam = {
    'ccp_alpha': np.linspace(0.0, 5.0, 25)
}
MaxNumSteps = 25

def runDT(X_train, X_test, y_train, y_test, data, path):
    log.debug('Analyizing Decision Trees')

    trainLength = X_train.shape[0]
    log.debug(trainLength)

    if trainLength <= MaxNumSteps: 
        HyperParams['max_depth'] = [None, *np.arange(1, trainLength + 1 )]
    else:
        step = math.ceil(trainLength / MaxNumSteps ) 
        log.debug('Step: %s' %step)
        HyperParams['max_depth'] = [None, *np.arange(1, trainLength, step=step), trainLength]
    log.debug('HyperParams[\'max_depth\'] = %s' %HyperParams)

    CV = ShuffleSplit(n_splits=10, test_size=0.333, random_state=0)

    # HyperParameter Testing
    gsc = GridSearchCV(
            estimator=tree.DecisionTreeClassifier(),
            param_grid=HyperParams,
            cv=CV, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    gridResults = gsc.fit(X_train, y_train)
    bestParams = gridResults.best_params_
    bestModel = gridResults.best_estimator_
    log.info('DT - Best Params: %s' %bestParams)

    # Learning Curve 
    train_sizes, train_scores, valid_scores, fit_times, score_times = learning_curve(
        bestModel, X_train, y_train, cv=CV, return_times=True)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Complexity Curve
    from sklearn.model_selection import validation_curve
    complex_train_scores, complex_valid_scores = validation_curve(bestModel, X_train, y_train, "max_depth", HyperParams['max_depth'], cv=CV)

    complex_train_scores_mean   = np.mean(complex_train_scores, axis=1)
    complex_train_scores_std    = np.std(complex_train_scores, axis=1)
    complex_test_scores_mean    = np.mean(complex_valid_scores, axis=1)
    complex_test_scores_std     = np.std(complex_valid_scores, axis=1)
    # log.info(complex_train_scores_mean.shape, complex_train_scores_std.shape, complex_test_scores_mean.shape, complex_test_scores_std.shape)

    # Plot learning curve
    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].grid()
    axes[0].set_title('DT - Learning Curve (max_depth: %i) ' % bestParams['max_depth'])
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
    axes[1].grid()
    maxDepth = np.array(HyperParams['max_depth'], dtype=float)
    maxDepth[0] = np.inf

    axes[1].set_title('DT - Complexity Curve (max_depth: %s, ccp_alpha: %d) ' %(bestParams['max_depth'], bestParams['ccp_alpha']))

    axes[1].fill_between(maxDepth, complex_train_scores_mean - complex_train_scores_std,
                            complex_train_scores_mean + complex_train_scores_std, alpha=0.1,
                            color="b")
    axes[1].fill_between(maxDepth, complex_test_scores_mean - complex_test_scores_std,
                            complex_test_scores_mean + complex_test_scores_std, alpha=0.1,
                            color="m")
    axes[1].plot(maxDepth, complex_train_scores_mean, 'o-', color="b",
                    label="Training score")
    axes[1].plot(maxDepth, complex_test_scores_mean, 'o-', color="m",
                    label="Cross-validation score")
    axes[1].set_ylim((0, 1.1))
    axes[1].set_xlabel("Maximum Depth Param")
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="best")

    saveDir = os.path.join( path, 'DT - max_depth.png')
    plt.savefig(saveDir, bbox_inches='tight')

    # ============================
    # Second output plot for Alpha
    # ============================
    complex_train_scores, complex_valid_scores = validation_curve(bestModel, X_train, y_train, "ccp_alpha", ComplexityParam['ccp_alpha'], cv=CV)

    complex_train_scores_mean   = np.mean(complex_train_scores, axis=1)
    complex_train_scores_std    = np.std(complex_train_scores, axis=1)
    complex_test_scores_mean    = np.mean(complex_valid_scores, axis=1)
    complex_test_scores_std     = np.std(complex_valid_scores, axis=1)

    # Plot Model-Complexity Curve for Alpha pruning Value 
    axes[1].clear()
    axes[1].grid()
    ccpAlpha = np.array(ComplexityParam['ccp_alpha'], dtype=float)

    axes[1].set_title('DT - Complexity Curve (max_depth: %i, ccp_alpha: %d) ' %(bestParams['max_depth'], bestParams['ccp_alpha']))

    axes[1].fill_between(ccpAlpha, complex_train_scores_mean - complex_train_scores_std,
                            complex_train_scores_mean + complex_train_scores_std, alpha=0.1,
                            color="b")
    axes[1].fill_between(ccpAlpha, complex_test_scores_mean - complex_test_scores_std,
                            complex_test_scores_mean + complex_test_scores_std, alpha=0.1,
                            color="m")
    axes[1].plot(ccpAlpha, complex_train_scores_mean, 'o-', color="b",
                    label="Training score")
    axes[1].plot(ccpAlpha, complex_test_scores_mean, 'o-', color="m",
                    label="Cross-validation score")
    axes[1].set_ylim((0, 1.1))
    axes[1].set_xlabel("Alpha Value")
    axes[1].set_ylabel("Score")
    axes[1].legend(loc="best")

    saveDir = os.path.join(path, 'DT - ccp_alpha.png')
    plt.savefig(saveDir, bbox_inches='tight')
