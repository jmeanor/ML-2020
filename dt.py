
# Decision Tree
import logging
from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import os
import math
# Plotting
import matplotlib
matplotlib.use("macOSX")
# Logging
log = logging.getLogger()
# Assignment Code Files
import graph
from analysis import runAnalysisIteration

HyperParams = {
    # 'max_depth': [None, 1, 2, 3, 4, 5, 10, 20],
    'ccp_alpha': np.linspace(0.0, 1.0, 15)
    # 'ccp_alpha': [0.0]
}
ComplexityParam = {
    'ccp_alpha': np.linspace(0.0, 2.0, 5),
    'max_depth': [None, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100, 500, 800]
}
MaxNumSteps = 10

def setHyperParams(trainLength):
    if trainLength <= MaxNumSteps:
        print('')
        # HyperParams['max_depth'] = [None, *np.arange(1, trainLength + 1)]
        # HyperParams['max_depth'] = [None, *np.logspace(0, trainLength+1, 10)]
    else:
        step = math.ceil(trainLength / MaxNumSteps)
        log.debug('Step: %s' % step)
        # HyperParams['max_depth'] = [
        #     None, 10, 20, 30, 40, *np.arange(50, trainLength, step=step), trainLength]
    log.debug('HyperParams[\'max_depth\'] = %s' % HyperParams)
    return

def runDT(X_train, X_test, y_train, y_test, data, path):
    log.debug('Analyizing Decision Trees')
    trainLength = X_train.shape[0]
    log.debug(trainLength)
    setHyperParams(X_train.shape[0])

    # CV = ShuffleSplit(n_splits=10, test_size=0.333, random_state=0)
    from cv import CV

    dataPack = (X_train, X_test, y_train, y_test, data, path)
    runAnalysisIteration('DT', tree.DecisionTreeClassifier(), HyperParams, ComplexityParam, 'max_depth', CV, data=dataPack)

    # ============================
    # Second output plot for Alpha
    # ============================
    # complex_train_scores, complex_valid_scores = validation_curve(
    #     bestModel, X_train, y_train, "ccp_alpha", ComplexityParam['ccp_alpha'], cv=CV)

    # complex_train_scores_mean = np.mean(complex_train_scores, axis=1)
    # complex_train_scores_std = np.std(complex_train_scores, axis=1)
    # complex_test_scores_mean = np.mean(complex_valid_scores, axis=1)
    # complex_test_scores_std = np.std(complex_valid_scores, axis=1)

    # # Plot Model-Complexity Curve for Alpha pruning Value
    # axes[1].clear()
    # axes[1].grid()
    # ccpAlpha = np.array(ComplexityParam['ccp_alpha'], dtype=float)

    # axes[1].set_title('DT - Complexity Curve (max_depth: %i, ccp_alpha: %d) ' %
    #                   (bestParams['max_depth'], bestParams['ccp_alpha']))

    # axes[1].fill_between(ccpAlpha, complex_train_scores_mean - complex_train_scores_std,
    #                      complex_train_scores_mean + complex_train_scores_std, alpha=0.1,
    #                      color="b")
    # axes[1].fill_between(ccpAlpha, complex_test_scores_mean - complex_test_scores_std,
    #                      complex_test_scores_mean + complex_test_scores_std, alpha=0.1,
    #                      color="m")
    # axes[1].plot(ccpAlpha, complex_train_scores_mean, 'o-', color="b",
    #              label="Training score")
    # axes[1].plot(ccpAlpha, complex_test_scores_mean, 'o-', color="m",
    #              label="Cross-validation score")
    # axes[1].set_ylim((0, 1.1))
    # axes[1].set_xlabel("Alpha Value")
    # axes[1].set_ylabel("Score")
    # axes[1].legend(loc="best")

    # saveDir = os.path.join(path, 'DT - ccp_alpha.png')
    # plt.savefig(saveDir, bbox_inches='tight')

    # # ======================================================
    # #   Final Testing Performance
    # # ======================================================

    # # Learning Curve
    # train_sizes, train_scores, test_scores = learning_curve(
    #     bestModel, X_test, y_test, cv=None, return_times=False, train_sizes=np.linspace(0.1, 1.0, 8))

    # train_scores_mean = np.mean(train_scores, axis=1)
    # train_scores_std = np.std(train_scores, axis=1)
    # test_scores_mean = np.mean(test_scores, axis=1)
    # test_scores_std = np.std(test_scores, axis=1)
    # # fit_times_mean = np.mean(fit_times, axis=1)
    # # fit_times_std = np.std(fit_times, axis=1)

    # # Plot Final Testing learning curve
    # _, axes = plt.subplots(1, 1, figsize=(10, 5))
    # axes.grid()
    # axes.set_title('DT - Final Test Learning Curve (max_depth: %i) ' %
    #                   bestParams['max_depth'])
    # axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
    #                      train_scores_mean + train_scores_std, alpha=0.1,
    #                      color="r")
    # axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
    #                      test_scores_mean + test_scores_std, alpha=0.1,
    #                      color="g")
    # axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
    #              label="Training score")
    # axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
    #              label="Testing score")
    # axes.set_ylim((0, 1.1))
    # axes.set_xlabel("# of samples")
    # axes.set_ylabel("Score")
    # axes.legend(loc="best")

    # saveDir = os.path.join(path, 'DT - final test.png')
    # plt.savefig(saveDir, bbox_inches='tight')
