# Main
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine
from pprint import pprint
import pandas as pd
import graphviz
import errno, os
from datetime import datetime
# Logging
import myLogger
import logging
logger = logging.getLogger()
logger.info('Initializing main.py')
# Plotting
import matplotlib  
matplotlib.use("macOSX")
from matplotlib import pyplot as plt
# Assignment code
from dt import runDT
from svm import runSVM

###
#    source: https://stackoverflow.com/questions/14115254/creating-a-folder-with-timestamp/14115286
###
def filecreation():
    mydir = os.path.join( os.getcwd(), 'output', datetime.now().strftime('%b-%d-%y %I:%M:%S %p') )
    # print('mydir %s' %mydir)
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return mydir

# Load Data
# data = load_iris()
data = load_wine()
# print(data.DESCR)
# pprint(data.target_names)
# pprint(data.feature_names)
# pprint(data.data)
# pprint(data.target)

path = filecreation()

# Randomly split the data into train & test sets.
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Run Analysis with Decision Tree
runDT(X_train, X_test, y_train, y_test, data, path)
runSVM(X_train, X_test, y_train, y_test, data, path)
