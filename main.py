import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine
from pprint import pprint
import pandas as pd
import graphviz 

# Plotting
import matplotlib  
matplotlib.use("macOSX")
from matplotlib import pyplot as plt

from dt import runDT

# Load Data
data = load_iris()
# data = load_wine()
# print(data.DESCR)
# pprint(data.target_names)
# pprint(data.feature_names)
# pprint(data.data)
# pprint(data.target)

# Randomly split the data into train & test sets.
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
print('X_train,', len(X_train), 'X_test,', len(X_test), 'Y_train,', len(y_train), len(y_test) )

# Run Analysis with Decision Tree
runDT(X_train, X_test, y_train, y_test, data)
