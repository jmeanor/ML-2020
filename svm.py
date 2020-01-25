# SVM
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve


# Plotting
import matplotlib  
matplotlib.use("macOSX")
from matplotlib import pyplot as plt

HyperParams = {
    'kernel':('linear', 'rbf'),
    'C':[1, 10]
}

def runSVM(X_train, X_test, y_train, y_test, data):
    print('Analyizing SVM')
    print('Length of training set: ', len(X_train))
    print(X_train.shape[0])

    CV = ShuffleSplit(n_splits=10, test_size=0.333, random_state=0)

    # HyperParameter Testing
    gsc = GridSearchCV(
            estimator=svm.SVC(),
            param_grid=HyperParams,
            cv=CV, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

    gridResults = gsc.fit(X_train, y_train)
    bestParams = gridResults.best_params_
    bestModel = gridResults.best_estimator_
    print('Best SVM Params: ', bestParams)

    # Decision Tree - Learning Curve 
    train_sizes, train_scores, valid_scores, fit_times, score_times = learning_curve(
        bestModel, X_train, y_train, cv=CV, return_times=True)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(valid_scores, axis=1)
    test_scores_mean = np.mean(valid_scores, axis=1)
    test_scores_std = np.std(valid_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    _, axes = plt.subplots(1, 2, figsize=(20, 5))
    axes[0].grid()
    axes[0].set_title('SVM - Learning Curve (kernel: %s, C: %i) ' %(bestParams['kernel'], bestParams['C']))
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                            train_scores_mean + train_scores_std, alpha=0.1,
                            color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                            test_scores_mean + test_scores_std, alpha=0.1,
                            color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                    label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                    label="Cross-validation score")
    axes[0].legend(loc="best")

    # # Plot n_samples vs fit_times
    # axes[1].grid()
    # axes[1].plot(train_sizes, fit_times_mean, 'o-')
    # axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
    #                         fit_times_mean + fit_times_std, alpha=0.1)
    # axes[1].set_xlabel("Training examples")
    # axes[1].set_ylabel("fit_times")
    # axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    # axes[1].grid()
    # axes[1].plot(fit_times_mean, test_scores_mean, 'o-')
    # axes[1].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
    #                     test_scores_mean + test_scores_std, alpha=0.1)
    # axes[1].set_xlabel("fit_times")
    # axes[1].set_ylabel("Score")
    # axes[1].set_title("Performance of the model")
    plt.show()