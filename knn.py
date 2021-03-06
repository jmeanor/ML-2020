import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
import os
import graph
# Logging
import logging
log = logging.getLogger()
# Assignment Code Files
from analysis import runAnalysisIteration

# Plotting
matplotlib.use("macOSX")

HyperParams = {
    'n_neighbors': [*np.arange(1, 20)],
    'weights': ['uniform', 'distance'],
    'leaf_size': [1, 10, 20, 30, 40, 50]
}
ComplexParams = {
    'n_neighbors': [*np.arange(1, 20)],
    'leaf_size': [1, 10, 20, 30, 40, 50]
}
def runKNN(X_train, X_test, y_train, y_test, data, path, classification=True):
    log.debug('Analyizing KNN')
    log.debug('Length of training set: %i' % len(X_train))
    log.debug(X_train.shape[0])

    # CV = ShuffleSplit(n_splits=10, test_size=0.333, random_state=0)
    from cv import CV
    if(classification):
        est = KNeighborsClassifier()
    else:
        est = KNeighborsRegressor()

    dataPack = (X_train, X_test, y_train, y_test, data, path)
    runAnalysisIteration('KNN', est, HyperParams, ComplexParams, 'n_neighbors', CV, data=dataPack)
