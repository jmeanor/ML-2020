from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

CV = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0) # StratifiedShuffleSplit
# CV = StratifiedShuffleSplit(n_splits=10, test_size=0.333, random_state=0)