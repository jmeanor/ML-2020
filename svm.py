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
    
    dataPack = (X_train, X_test, y_train, y_test, data, path)
    from analysis import runAnalysisIteration
    runAnalysisIteration('SVM', svm.SVC(), HyperParams, 'C', CV, data=dataPack)

