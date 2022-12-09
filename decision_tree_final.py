import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics, linear_model, ensemble, model_selection

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def runTree(X, Y):
    # Splitting test and train data into test_size %
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20)
    # Finding an optimized decision tree
    # Using max_depth, criterion will suffice for DT Models, rest all will remain constant
    parameters = {'max_depth': np.arange(1, 19)
        , 'max_features': ('sqrt', 'log2')
        , 'min_samples_split': np.arange(1, 10)
                  }

    """ BASELINE TREE """
    tree = DecisionTreeRegressor(random_state=42,max_depth=4,min_samples_split=0.025)
    tree.fit(xtrain, ytrain)
    ypred = tree.predict(xtest)
    # Evaluation metrics
    mse = mean_squared_error(ytest, ypred)
    rmse = mean_squared_error(ytest, ypred) ** (1 / 2.0)
    MAE = mae(ytest, ypred)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("MAE:", MAE)
    print(f'Train Accuracy - : {tree.score(xtrain, ytrain):.3f}')
    print(f'Test Accuracy - : {tree.score(xtest, ytest):.3f}')
    score = tree.score(xtrain, ytrain)
    print("R-squared:", score)


    fig, ax = plt.subplots(figsize=(8, 12))

    fig.suptitle('Predicted vs Actual', fontweight="bold", fontsize=15)

    ### Tree
    ax = sns.regplot(ax=ax, x=ypred, y=ytest, label='predicted',
                     color='darkcyan',
                     scatter_kws={'alpha': 0.5},
                     line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    # Plot 45 degree line
    ax.plot(ytest, ytest, ls='solid', c='grey', label='actual')
    ax.plot()
    # Labels
    ax.set_title('Linear Regression')
    ax.set_xlabel('')
    ax.set_ylabel('True values')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 12))
    error_value = metrics.mean_squared_error(ytest, ypred)
    plt.subplot()
    sns.regplot(x=ytest, y=ypred, fit_reg=True, color='red')
    sns.regplot(x=ytest, y=ytest, fit_reg=True, color='black')
    plt.legend(["Prediction points", "Prediction best fit", "Area", "Actual point", "Actual line"])
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.suptitle("Evaluation of Decision Tree")
    plt.title("Mean squared error: " + str(error_value))
    plt.grid(True)
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 12))
    # Plotting test vs predicted data
    x_ax = range(len(ytest))
    plt.plot(x_ax, ytest, linewidth=1, label="original")
    plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
    plt.title("y-test and y-predicted data")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()