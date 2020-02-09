# SVM
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.model_selection import learning_curve
import os
import graph
# Logging
import logging
log = logging.getLogger()
# Assignment Code Files
from analysis import runAnalysisIteration

HyperParams = {
    'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
    'C': np.logspace(-6, -1, 5),
    'max_iter': [5000]
}
ComplexParams = {
    'kernel': ('linear', 'rbf'),
    'C': np.logspace(-6, -1, 5),
    'max_iter': [1000,2000,3000,5000]
}

def runSVM(X_train, X_test, y_train, y_test, data, path):
    log.debug('Analyizing SVM')
    log.debug('Length of training set: %i' % len(X_train))
    log.debug(X_train.shape[0])

    # CV = ShuffleSplit(n_splits=10, test_size=0.333, random_state=0) # StratifiedShuffleSplit
    # CV = StratifiedShuffleSplit(n_splits=10, test_size=0.333, random_state=0)
    from cv import CV

    # Debug
    HyperParams = {'C': [0.1], 'kernel': ['poly'], 'max_iter': [10000]}
    
    dataPack = (X_train, X_test, y_train, y_test, data, path)
    runAnalysisIteration('SVM', svm.SVC(), HyperParams, ComplexParams, 'max_iter', CV, data=dataPack)

