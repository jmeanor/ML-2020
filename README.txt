Assignment 1
Supervised Learning Algorithms
Spring 2020
John Meanor

To run the project:
1. Using Python 3.7+, simply install all dependencies/imports.
2. Run `python3 main.py`

The main.py module will:
1.  Load both datasets from the relative /input/ directory. 
2.  Create a new /output/ directory, if one does not already exist.
2a. Create a new subdirectory within /output/ named by the datetime of the instance run.
3.  Starting with Dataset1, split each dataset into an 85% training set 15% testing set split
3a. Each dataset has its own subdirectory for each instance run.
4.  Run each analysis:
    - runDT (Decision Tree)
    - runKNN (k-Nearest Neighbor)
    - runSVM (SVM)
    - runANN (Neural Network)

5.  Within each analysis module above, the process is the same as follows:
    - Explore the hyper-parameters defined in each of the specific algorithm files (i.e. dt.py, knn.py, etc.) 
    - Select the most optimal hyper parameter sets based on cross-validation
    - Log the optimal parameters and values to the metadata.txt file in each dataset's subdirectory
    - Pass the learned model, with optimal parameters from those tested, to the validation curve function.
    - Explore and generate the model complexity (validation) curve data
    - Graph the learning curve and model-complexity curve

Notes:
To change the hyper-parameters under test, and the complexity analysis, they're simply changed in each file by either commenting in/out or adding to the dictionary.

Normally, main.py will run every analysis for both datasets. To selectively run the analysis for only one dataset or one algorithm, simply comment it out within main.py.

Project Structure

├── README.txt
├── analysis.py
├── ann.py
├── boosting.py
├── cv.py
├── dt.py
├── graph.py
├── input
│   ├── hirosaki_temp_cherry.csv
│   └── singapore-listings.csv
├── knn.py
├── log.txt
├── main.py
├── myLogger.py
└── svm.py