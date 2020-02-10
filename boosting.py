# Boosting
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn import ensemble  as ens
from sklearn import tree 
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
    'n_estimators': [10, 25, 50, 100, 200]
}
ComplexParams = {
    'learning_rate': [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
}
def runBoost(X_train, X_test, y_train, y_test, data, path, classification=True):
    log.debug('Analyizing Boosting')
    log.debug('Length of training set: %i' % len(X_train))
    log.debug(X_train.shape[0])

    # CV = ShuffleSplit(n_splits=10, test_size=0.333, random_state=0)
    from cv import CV
    
    if(classification):
        est = ens.AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=5), n_estimators=25)
    else:
        est = ens.AdaBoostRegressor(tree.DecisionTreeRegressor(max_depth=5), n_estimators=25)

    dataPack = (X_train, X_test, y_train, y_test, data, path)
    runAnalysisIteration('BST', est, HyperParams, ComplexParams, 'learning_rate', CV, data=dataPack)
