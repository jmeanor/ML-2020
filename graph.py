# Plotting
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("macOSX")

# Plot learning curve
def plotLearningCurve(axes, title, train_sizes, train_scores_mean, train_scores_std, test_scores_mean, 
    test_scores_std, label1="", label2="", xlabel="", ylabel=""):
    axes.grid()
    axes.set_title(title)
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label=label1)
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label=label2)
    axes.set_ylim((0, 1.1))
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend(loc="best")

# Plot learning curve
def plotComplexityCurve(axes, title, X, train_scores_mean, train_scores_std, test_scores_mean, 
    test_scores_std, label1="", label2="", xlabel="", ylabel=""):
    axes.grid()
    axes.set_title('Model Complexity')

    axes.fill_between(X, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="b")
    axes.fill_between(X, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="m")
    axes.plot(X, train_scores_mean, 'o-', color="b", label=label1)
    axes.plot(X, test_scores_mean, 'o-', color="m", label=label2)
    axes.set_ylim((0, 1.1))
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.legend(loc="best")