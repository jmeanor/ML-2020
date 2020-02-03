# SVM
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn import neural_network
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

HyperParams = {
    'learning_rate': ('constant', 'invscaling', 'adaptive'),
    'max_iter': [200, 300, 400]
}

def runANN(X_train, X_test, y_train, y_test, data, path):
    log.debug('Analyizing ANN')
    log.debug('Length of training set: %i' % len(X_train))
    log.debug(X_train.shape[0])

    CV = ShuffleSplit(n_splits=10, test_size=0.333, random_state=0)
    
    dataPack = (X_train, X_test, y_train, y_test, data, path)
    runAnalysisIteration('ANN', neural_network.MLPClassifier(), HyperParams, 'max_iter', CV, data=dataPack)

