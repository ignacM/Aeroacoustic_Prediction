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

def runTree(X,Y):
    # Splitting test and train data into test_size %
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20)
    # Finding an optimized decision tree
    # Using max_depth, criterion will suffice for DT Models, rest all will remain constant
    parameters = {'max_depth': np.arange(1,19)
        , 'max_features': ('sqrt', 'log2')
        , 'min_samples_split': np.arange(1, 10)
                  }

    regressor = DecisionTreeRegressor()
    param_grid = {'max_depth': np.arange(1,19)
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

    fig, ax = plt.subplots(figsize=(8, 12))

    fig.suptitle('Predicted vs Actual', fontweight="bold", fontsize=15)

    ### Tree
    ax = sns.regplot(ax=ax, x=ypred, y=ytest, label='predicted',
                          color='darkcyan',
                          scatter_kws={'alpha': 0.5},
                          line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})
    x0 = -0.00415
    x1 = 0.00025
    y0 = -0.03
    y1 = 0.03
    # Plot 45 degree line
    ax.plot(ytest, ytest, ls='solid', c='grey', label='actual')
    ax.plot()
    # Labels
    ax.set_title('Linear Regression')
    ax.set_xlabel('')
    ax.set_ylabel('True values')
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 12))
    fig.suptitle('Figures', fontweight="bold", fontsize=15)

    ax= sns.residplot(ax=ax, x=ypred, y=ytest,
                            lowess=True,
                            color='darkcyan',
                            scatter_kws={'alpha': 0.5},
                            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8})

    ax.set_title('')
    ax.set_xlabel('ypred')
    ax.set_ylabel('ytest')
    plt.show()

    """
    # Plotting test vs predicted data
    x_ax = range(len(ytest))
    axes[1] = plt.plot(x_ax, ytest, linewidth=1, label="original")
    axes[1] = plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
    axes[1] = plt.title("y-test and y-predicted data")
    axes[1] = plt.xlabel('X-axis')
    axes[1] = plt.ylabel('Y-axis')
    axes[1] = plt.legend(loc='best', fancybox=True, shadow=True)
    axes[1] = plt.grid(True)
    plt.show()
    """
    """
    ############
    error_value = metrics.mean_squared_error(ytest, ypred)
    plt.subplot()
    sns.regplot(x=ytest, y=ypred, fit_reg=True)
    sns.regplot(x=ytest, y=ytest, fit_reg=True, color="Red")
    plt.xticks(range(0, 10))
    plt.yticks(range(0, 10))
    plt.legend(["Prediction", "Actual data"])
    plt.xlabel("Real value")
    plt.ylabel("Predicted value")
    plt.suptitle("Evaluation of Decision Tree")
    plt.title("Mean squared error: " + str(error_value))
    plt.grid(True)
    plt.show()

    n_folds = 5
    regr = dtr
    cv_error = np.average(cross_val_score(regr, X, Y, scoring='neg_mean_squared_error', cv=n_folds))
    print("Cross Validation: {}".format(cv_error))
    regr.fit(xtrain, ytrain)
    y_pred = regr.predict(xtest)
    y_train_pred = regr.predict(xtrain)
    print("Mean squared error testing data: {}".format(mean_squared_error(ytest, y_pred)))
    print("Mean squared error training data: {}".format(mean_squared_error(ytrain, y_train_pred)))
    plt.scatter(ytest, y_pred, marker='o')
    sns.regplot(x=ytest, y=ypred, fit_reg=True)
    plt.show()"""

