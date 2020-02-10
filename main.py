# Main
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_files
from pprint import pprint
import pandas as pd
import graphviz
import errno
import os
from datetime import datetime
# Assignment Code
from svm import runSVM
from dt import runDT
from knn import runKNN
from ann import runANN
from boosting import runBoost
# Logging
import myLogger
import logging
logger = logging.getLogger()
logger.info('Initializing main.py')
log = logging.getLogger()
# Plotting
matplotlib.use("macOSX")

# Assignment code

###
#    source: https://stackoverflow.com/questions/14115254/creating-a-folder-with-timestamp/14115286
###


def createDateFolder(suffix=("")):
    mydir = os.path.join(os.getcwd(), 'output', *suffix)
    # print('mydir %s' %mydir)
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return mydir

def setLog(path, oldHandler = None):
    if oldHandler != None:
        myLogger.logger.removeHandler(oldHandler)
    logPath = os.path.join(path, 'metadata.txt')
    fh = myLogger.logging.FileHandler(logPath)
    fh.setLevel(logging.INFO)
    fmtr = logging.Formatter('%(message)s')
    fh.setFormatter(fmtr)
    myLogger.logger.addHandler(fh)
    return fh

# ==========================================
#   Load Data Set 1
# ==========================================
def loadDataset1():
    # df = pd.read_csv('./input/AB_NYC_2019.csv', delimiter=',', header=0)
    df = pd.read_csv('./input/singapore-listings.csv', delimiter=',', header=0)
    
    # Pre-Processing - Removing attributes with no value
    # df = df.drop(['id', 'name', 'host_name', 'last_review'], axis=1)
    df = df.drop(['id', 'name', 'host_name', 'last_review'], axis=1)
    # df = df[pd.notnull(df['reviews_per_month'])]
    # df['last_review'] = df['last_review'].astype('datetime64[ns]').astype(str)

    # ==========================================
    # Discretize the classifications 
    #  Source: https://dfrieds.com/data-analysis/bin-values-python-pandas
    # ==========================================
    df['price_bins'] = pd.cut(x=df['price'], bins= np.arange(10, 10010, step=10), labels=np.arange(10, 10000, step=10)).astype(int)


    # ==========================================
    # PreProcessing - Encoding Categories
    # Source - https://blog.cambridgespark.com/robust-one-hot-encoding-in-python-3e29bfcec77e
    # ==========================================
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.preprocessing import KBinsDiscretizer
    categories = [
        'neighbourhood_group',
        'neighbourhood',
        'room_type'
    ]
    df_processed = pd.get_dummies(df, prefix_sep="__", columns=categories)
    dummies = [col for col in df_processed
        if "__" in col and col.split("__")[0] in categories]
    processed_columns = list(df_processed.columns[:])

    print(processed_columns)
    # target = np.array(df_processed['price'])
    # data = np.array(df_processed.drop('price', axis=1))
    target = np.array(df_processed['price_bins'])
    data = np.array(df_processed.drop('price_bins', axis=1))
    
    # 2nd Iteration with imputing missing values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'constant', fill_value=0)
    imputer = imputer.fit(data)
    data = imputer.transform(data)
    # =====================================
    # pprint(data)
    
    data1 = {
        'data': data,
        'target': target
    }
    log.debug(data1)
    return data1

# ==========================================
#   Load Data Set 2
# ==========================================
def loadDataset2():
    df = pd.read_csv('./input/Phishing.csv', delimiter=',', header=0)
    
    pprint(df)
    target = np.array(df['Result'])
    data = np.array(df.drop('Result', axis=1))
    data2 = {
        'data': data,
        'target': target
    }
    return data2

def loadDataset3():
    df = pd.read_csv('./input/hirosaki_temp_cherry.csv', delimiter=',', header=0)
    
    pprint(df)
    target = np.array(df['flower_status'])
    data = np.array(df.drop('flower_status', axis=1))
    data2 = {
        'data': data,
        'target': target
    }
    return data2
# ==========================================
#   Load DEV Data Sets
# ==========================================
def loadDevDatasets():
    data1 = load_iris()
    data2 = load_breast_cancer()
    # data2 = load_wine()
    # pprint(data)
    # pprint(type(data))
    return data1, data2

def runAnalysis(data_set, output_path, classification=True):
    # Randomly split the data into train & test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        data_set['data'], data_set['target'], test_size=0.15, random_state=0)

    log.info('Length of training set: %i' % len(X_train))
    log.info('Length of testing  set: %i' % len(X_test))

    # Run Analysis with Decision Tree
    runDT(X_train, X_test, y_train, y_test, data_set, output_path, classification=classification)
    runBoost(X_train, X_test, y_train, y_test, data_set, output_path, classification=classification)
    runKNN(X_train, X_test, y_train, y_test, data_set, output_path, classification=classification)
    runSVM(X_train, X_test, y_train, y_test, data_set, output_path)
    runANN(X_train, X_test, y_train, y_test, data_set, output_path)

# ==========================================

timestamp = datetime.now().strftime('%b-%d-%y %I:%M:%S %p')
# ==========================================
# Analyize Data Set 1
# ==========================================
data1 = loadDataset1() 
path1 = createDateFolder((timestamp, "AirBNB Singapore"))
oldHandler = setLog(path1)
runAnalysis(data_set=data1, output_path=path1, classification=True)

# ==========================================
# Analyize Data Set 2
# ==========================================
data2 = loadDataset3() 
path2 = createDateFolder((timestamp, "Cherry-Blossoms"))
setLog(path2, oldHandler)
runAnalysis(data_set=data2, output_path=path2)
