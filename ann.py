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
    'hidden_layer_sizes': [(100,) (50,50), (100,100)],
    'learning_rate': ('constant', 'adaptive'),
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],

    # 'hidden_layer_sizes': [(5,), (20,), (100,)],
    # 'solver': ['adam'],
    # 'alpha': [0.0001],
}
ComplexParams = {
    'learning_rate': ('constant', 'invscaling', 'adaptive'),
    'max_iter': [25, 50, 100, 150, 200, 250],
    # 'hidden_layer_sizes': [(50,50), (100,100)],
    'hidden_layer_sizes': [(5,), (20,), (100,)],
    
}
def runANN(X_train, X_test, y_train, y_test, data, path):
    log.debug('Analyizing ANN')
    log.debug('Length of training set: %i' % len(X_train))
    log.debug(X_train.shape[0])

    # CV = ShuffleSplit(n_splits=10, test_size=0.333, random_state=0)
    from cv import CV
    
    dataPack = (X_train, X_test, y_train, y_test, data, path)
    runAnalysisIteration('ANN', neural_network.MLPClassifier(), HyperParams, ComplexParams, 'max_iter', CV, data=dataPack)

