import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics, linear_model, ensemble, model_selection
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def runTree(X,Y):
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20)
    # Max depth will suffice as an optimizer for trees, but other optimizable parameters are:
    full_parameters_list = {'max_depth': np.arange(1, 19)
        , 'max_features': ('sqrt', 'log2')
        , 'min_samples_split': np.arange(1, 10)
                  }

    regressor = DecisionTreeRegressor()
    param_grid = {'max_depth': np.arange(1, 10),
                  'min_samples_split': np.arange(0.01, 1, 0.01)
                  }
    model = model_selection.GridSearchCV(estimator=regressor, param_grid=param_grid,
                                         verbose=10, n_jobs=1, cv=5)
    model.fit(xtrain, ytrain)
    # Re-build the model with best estimated tree
    tree = model.best_estimator_
    print_regression_solutions(xtrain, ytrain, xtest, ytest, tree, 'Decision Tree')

    ypred = tree.predict(xtest)
    plot_regression_outcome(ytest, ypred, 'Decision Tree')

    # Compare decision tree with least squares l2 regularization
    l2 = Ridge(alpha=0.01)
    print_regression_solutions(xtrain, ytrain, xtest, ytest, l2, 'Ridge Regression')
    l2_pred = l2.predict(xtest)

    plot_regression_outcome(ytest, l2_pred, 'Ridge Regression')
    print_regression_residuals(ytest, ypred, 'Decision Tree')
    return


def plot_regression_outcome(ytest, ypred, method):
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.suptitle('Predicted vs Actual for %s' % method, fontweight="bold", fontsize=15)
    error_value = round(metrics.mean_absolute_error(ytest, ypred), 2)
    plt.subplot()
    sns.regplot(x=ytest, y=ypred, fit_reg=True, color='red') # fit_reg prints the area
    sns.regplot(x=ytest, y=ytest, fit_reg=True, color='black')
    plt.legend(["Prediction points", "Prediction best fit", "Area", "Actual point", "Actual line"])
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.title("Mean absolute error: %.2f" % error_value)
    plt.grid(True)
    plt.show()

    # Evaluation metrics
    mse = mean_squared_error(ytest, ypred)
    rmse = mean_squared_error(ytest, ypred) ** (1 / 2.0)
    MAE = mae(ytest, ypred)
    print("%s MSE: " % method, mse)
    print("%s RMSE: " % method, rmse)
    print("%s MAE:" % method, MAE)

    return


def print_regression_solutions(xtrain, ytrain, xtest, ytest, model, method):
    model.fit(xtrain, ytrain)
    print(" ")
    print(f'%s Train Accuracy - : {model.score(xtrain, ytrain):.3f}' % method)
    print(f'%s Test Accuracy - : {model.score(xtest, ytest):.3f}' % method)
    score = model.score(xtrain, ytrain)
    print("%s R-squared:" % method, score)
    print(" ")
    return


def print_regression_residuals(ytest, ypred, method):
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle('Residual Plot for %s' % method, fontweight="bold", fontsize=15)
    residuals = ytest - ypred
    ax = sns.residplot(ax=ax, x=ypred, y=residuals, lowess=True, color='darkcyan',
                       scatter_kws={'alpha': 0.5},
                       line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    plt.legend(["", "Predicted values", "Actual values line"])
    ax.set_xlabel('Sound Pressure Level, (dB)')
    ax.set_ylabel('Decibel Error in predicted values using %s' % method)
    plt.show()
    return


