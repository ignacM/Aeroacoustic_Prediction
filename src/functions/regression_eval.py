import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn import metrics, linear_model, ensemble, model_selection, tree
from sklearn.pipeline import Pipeline

from graphviz import Source

from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score


def runTree(X,Y):
    """
    Returns fitted tree and compares its results with l2 ridge regression
    :param X: complete X data
    :param Y: complete Y data
    :return:
    """
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.20)
    # Max depth will suffice as an optimizer for trees, but other optimizable parameters are:
    """full_parameters_list = {
            'max_depth': np.arange(1, 19),
            'max_features': ('sqrt', 'log2'),
            'min_samples_split': np.arange(1, 10)
                                }"""

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
    """
    Plots a graph comparing ytest and ypred. Prints MSE, RMSE, and MAE.
    :param ytest:
    :param ypred:
    :param method: 'str', name of method used. Eg. 'Regressor', 'Decision Tree'
    :return:
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    plt.suptitle('Predicted vs Actual for %s' % method, fontweight="bold", fontsize=15)
    error_value = round(metrics.mean_absolute_error(ytest, ypred), 2)
    plt.subplot()
    sns.regplot(x=ytest, y=ypred, fit_reg=True, color='red')  # fit_reg prints the area
    sns.regplot(x=ytest, y=ytest, fit_reg=True, color='black')
    plt.legend(["Prediction points", "Prediction best fit", "Confidence Interval", "Actual point", "Actual line"],
               fancybox=True, shadow=True)
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


def print_regression_solutions(xtrain, ytrain, xtest, ytest, model):
    """
    Fits given model. Prints train and test accuracy as well as R2.
    :param xtrain:
    :param ytrain:
    :param xtest:
    :param ytest:
    :param model:
    :return:
    """
    model.fit(xtrain, ytrain)
    print(" ")
    print(f'%s Train Accuracy - : {model.score(xtrain, ytrain):.3f}' )
    print(f'%s Test Accuracy - : {model.score(xtest, ytest):.3f}' )
    score = model.score(xtrain, ytrain)
    print("%s R-squared:", score)
    print(" ")
    return


def print_regression_residuals(ytest, ypred, method):
    """
    Plots a graph of residuals of predicted values. Accepts series and arrays.
    :param ytest:
    :param ypred:
    :param method: 'str', name of method used. Eg. 'Regressor', 'Decision Tree'
    :return:

    Confidence Interval: 'ci'
    ci int in [0, 100] or None, optional Size of the confidence interval for the regression estimate. This will be drawn
    using translucent bands around the regression line. The confidence interval is estimated using a bootstrap; for
    large datasets, it may be advisable to avoid that computation by setting this parameter to None.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.suptitle('Residual Plot for %s' % method, fontweight="bold", fontsize=15)
    if isinstance(ytest, pd.Series):
        error_value = round(metrics.mean_absolute_error(np.array(ytest), np.array(ypred)), 2)
        ax = sns.residplot(ax=ax, x=np.array(ytest), y=np.array(ypred), lowess=True, color='black',
                           scatter_kws={'alpha': 1, 'marker': 'x', 's': 75},
                           line_kws={'color': 'red', 'lw': 1, 'alpha': 1})
    else:
        error_value = round(metrics.mean_absolute_error(ytest, ypred), 2)
        ax = sns.residplot(ax=ax, x=ytest, y=ypred, lowess=True, color='black',
                           scatter_kws={'alpha': 1, 'marker': 'x', 's': 75},
                           line_kws={'color': 'red', 'lw': 1, 'alpha': 1})

    plt.legend(["Zero-error line", "Predicted values", "Lowess Smooth line"], fancybox=True, shadow=True)
    ax.set_xlabel('Sound Pressure Level, (dB)')
    ax.set_ylabel('Decibel Error in predicted values')
    plt.title("Mean absolute error: %.2f" % error_value)
    plt.show()
    return


def print_test_vs_real(y_test, y_pred):
    """
    Plots a graph of test vs predicted values. Fits line to points.
    :param y_test:
    :param y_pred:
    :return:
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    error_value = round(metrics.mean_absolute_error(y_test, y_pred), 2)
    # Plotting test vs predicted data
    x_ax = range(len(y_test))
    plt.plot(x_ax, y_test, linewidth=1.5, label="Actual data")
    plt.plot(x_ax, y_pred, linewidth=1.5, label="Predicted data")
    plt.suptitle("Actual sound data vs Predicted sound data", fontweight="bold", fontsize=15)
    plt.title("Mean absolute error: %.2f" % error_value)
    plt.xlabel('Microphone number')
    plt.ylabel('Sound Pressure Level, dB')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()


def print_tree(fitted_dt_model):
    """
    Plots decision tree. Exports tree as .dot through graphviz.
    :param fitted_dt_model: decision tree after fitted with train data.
    :return:
    """
    fig, axe = plt.subplots(figsize=(60, 40))
    tree.plot_tree(fitted_dt_model, ax=axe, fontsize=10)
    plt.show()
    tree.plot_tree(fitted_dt_model)
    tree.export_graphviz(fitted_dt_model,
                         out_file="../models/tree.dot",
                         filled=True, )

    path = '../models/tree.dot'
    s = Source.from_file(path)


def print_predictions(y_pred, ypred_labels):
    """
    Plots a graph of predicted values. Fits line to points.
    :param y_pred:
    :param ypred_labels: array of names of ypred
    :return:
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    # Plotting test vs predicted data
    for item in range(len(y_pred)):
        plt.plot(y_pred[item], linewidth=1.5)
    plt.title("Predicted sound data", fontweight="bold", fontsize=15)
    plt.xlabel('Microphone number')
    plt.ylabel('Sound Pressure Level, dB')
    plt.legend(ypred_labels)
    plt.grid(True)
    plt.show()


def plot_preds_vs_real(predictions, real, prediction_labels, suptitle):

    fig, ax = plt.subplots(figsize=(7, 6))
    for item in range(len(predictions)):
        plt.plot(predictions[item], linewidth=1.25)
        # plt.scatter(np.arange(0, 23, 1), predictions[item])
    plt.plot(real, linewidth=1.25)
    plt.suptitle("Predicted sound data", fontweight="bold", fontsize=16)
    plt.title(suptitle, fontweight="bold", fontsize=14)
    plt.xlabel('Microphone number', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('Sound Pressure Level, dB', fontsize=14)
    plt.yticks(fontsize=14)
    try:
        prediction_labels = np.append(prediction_labels, 'Actual Data')
        plt.legend(prediction_labels, fontsize=13)
    except:
        pass
    plt.grid(False)
    plt.show()


def plot_preds_vs_real_2(predictions, real, prediction_labels, suptitle):

    fig, ax = plt.subplots(figsize=(7, 6))
    for item in range(len(predictions)):
        if item < 2:
         plt.plot(predictions[item], linewidth=1.25)
        elif item ==2:
            x = np.arange(0, 23, 1)
            poly = np.polyfit(x, predictions[item], deg=4)
            ax.plot(np.polyval(poly, x))
        # plt.scatter(np.arange(0, 23, 1), predictions[item])
    plt.plot(real, linewidth=1.25)
    plt.suptitle("Predicted sound data", fontweight="bold", fontsize=16)
    plt.title(suptitle, fontweight="bold", fontsize=14)
    plt.xlabel('Microphone number', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('Sound Pressure Level, dB', fontsize=14)
    plt.yticks(fontsize=14)
    try:
        prediction_labels = np.append(prediction_labels, 'Actual Data')
        plt.legend(prediction_labels, fontsize=13)
    except:
        pass
    plt.grid(False)
    plt.show()

def r2(actual, predicted, model):
    score = r2_score(actual, predicted)
    print("%s R-squared:" %model, score)
    return

