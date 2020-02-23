
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
    'max_depth': [None, 1, 2, 3, 4, 5, 10, 20],
    'ccp_alpha': np.linspace(0.0, 1.0, 15),
    # 'min_samples_split': [2, 5, 10, 20]
}
ComplexityParam = {
    'ccp_alpha': np.linspace(0.0, 1.0, 15),
    'max_depth': [None, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100],
    'min_samples_split': [2, 5, 10, 20]
}
MaxNumSteps = 10

def setHyperParams(trainLength):
    if trainLength <= MaxNumSteps:
        HyperParams['max_depth'] = np.arange(1, trainLength)
        # HyperParams['max_depth'] = [None, *np.logspace(0, trainLength+1, 10)]
    else:
        step = math.ceil(trainLength / MaxNumSteps)
        log.debug('Step: %s' % step)
        HyperParams['max_depth'] = [
            1, 10, 20, 30, 40, *np.arange(50, trainLength, step=step)]
    log.debug('HyperParams[\'max_depth\'] = %s' % HyperParams)
    return

def runDT(X_train, X_test, y_train, y_test, data, path, classification=True):
    log.debug('Analyizing Decision Trees')
    trainLength = X_train.shape[0]
    log.debug(trainLength)
    setHyperParams(X_train.shape[0])

    # CV = ShuffleSplit(n_splits=10, test_size=0.333, random_state=0)
    from cv import CV

    if(classification):
        est = tree.DecisionTreeClassifier()
    else:
        est = tree.DecisionTreeRegressor()


    dataPack = (X_train, X_test, y_train, y_test, data, path)
    model, params = runAnalysisIteration('DT', est, HyperParams, ComplexityParam, 'min_samples_split', CV, data=dataPack)

    # saveDir = os.path.join(path, 'tree_model.dot')
    # dotfile = open(saveDir, 'wr')
    names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    
    import graphviz
    dot_graph = tree.export_graphviz(model, out_file = None, feature_names = names)
    g = graphviz.Source(dot_graph)
    g.format = "png"
    g.render("tree-model")

    # tree.export_graphviz(model, out_file = dotfile, feature_names = names)
    # g = graphviz.Source(dotfile)
    # g.render()
    # dotfile.close()
