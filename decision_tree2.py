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
    tree = DecisionTreeRegressor(random_state=42)
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

    ParameterGrid = {"min_samples_split": np.arange(0.025, 1, 0.025)}

    min_samples_split_gsearch = GridSearchCV(estimator=tree, param_grid=ParameterGrid, cv=5)

    min_samples_split_gsearch.fit(xtrain, ytrain)

    min_samples_split_gsearch.best_params_
    best_min_samples_split_tree = min_samples_split_gsearch.best_estimator_
    print(best_min_samples_split_tree)

    ypred = min_samples_split_gsearch.predict(xtest)

    ParameterGrid = {"max_depth": np.arange(3, 10, 1)}

    depth_gsearch = GridSearchCV(estimator=tree, param_grid=ParameterGrid, cv=5)

    depth_gsearch.fit(xtrain, ytrain)

    depth_gsearch.best_params_
    depth_tree = depth_gsearch.best_estimator_
    print(depth_tree)


    # Evaluation metrics
    mse = mean_squared_error(ytest, ypred)
    rmse = mean_squared_error(ytest, ypred) ** (1 / 2.0)
    MAE = mae(ytest, ypred)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("MAE:", MAE)
    print(f'Train Accuracy - : {min_samples_split_gsearch.score(xtrain, ytrain):.3f}')
    print(f'Test Accuracy - : {min_samples_split_gsearch.score(xtest, ytest):.3f}')
    score = min_samples_split_gsearch.score(xtrain, ytrain)
    print("R-squared:", score)



    """

    param_grid = {'max_depth': np.arange(1, 19)
        , 'max_features': ('sqrt', 'log2')
        , 'min_samples_split': np.arange(2, 10)
                  }

    model = model_selection.GridSearchCV(estimator=regressor, param_grid=param_grid, verbose=10, n_jobs=1, cv=5)
    model.fit(xtrain, ytrain)
    print(model.best_score_)
    print(model.best_estimator_.get_params())

    # train DT classifier for each ccp_alpha value

    # Re-build the model with best estimated tree
    best_model = model.best_estimator_
    tree = best_model
    tree.fit(xtrain, ytrain)

    print(f'Train Accuracy - : {tree.score(xtrain, ytrain):.3f}')
    print(f'Test Accuracy - : {tree.score(xtest, ytest):.3f}')
    score = tree.score(xtrain, ytrain)
    print("R-squared:", score)

    ypred = tree.predict(xtest)

    # Evaluation metrics
    mse = mean_squared_error(ytest, ypred)
    rmse = mean_squared_error(ytest, ypred) ** (1 / 2.0)
    MAE = mae(ytest, ypred)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("MAE:", MAE)

    l2 = Ridge(alpha=0.01)
    l2.fit(xtrain, ytrain)
    l2_pred = l2.predict(xtrain)
    print("Ridge RMSE for train data :", mean_squared_error(ytrain, l2_pred, squared=False))
    print("Ridge R2 score for train:", r2_score(ytrain, l2_pred))
    print(" ")
    l2_pred = l2.predict(xtest)
    print("Ridge RMSE for test data :", mean_squared_error(ytest, l2_pred, squared=False))
    print("Ridge R2 score for test:", r2_score(ytest, l2_pred))
    """

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