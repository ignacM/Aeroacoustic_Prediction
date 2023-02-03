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
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def runTree(X, Y):
    # Splitting test and train data into test_size %20
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    # Finding an optimized decision tree
    # Using max_depth, criterion will suffice for DT Models, rest all will remain constant
    parameters = {'max_depth': np.arange(1, 19),
                  'max_features': ('sqrt', 'log2'),
                  'min_samples_split': np.arange(1, 10)
                  }

    tree = DecisionTreeRegressor(random_state=42)
    tree.fit(x_train, y_train)
    ypred = tree.predict(x_test)
    ParameterGrid = {"min_samples_split": np.arange(0.01, 1, 0.01)}
    min_samples_split_gsearch = GridSearchCV(estimator=tree, param_grid=ParameterGrid, cv=5)
    min_samples_split_gsearch.fit(x_train, y_train)
    min_samples_split_gsearch.best_params_
    best_min_samples_split_tree = min_samples_split_gsearch.best_estimator_
    print(best_min_samples_split_tree)

    lista = {
        'criterion': 'friedman_mse',
        'max_depth': None,
        'min_samples_split': None
    }
    def mae_scores_min_sample_split(lista, arguments):
        #lista.update({arguments})
        regressor = DecisionTreeRegressor(**arguments)
        scores = -1 * cross_val_score(tree, x_train, y_train, cv=3, scoring='neg_mean_absolute_error')
        return np.mean(scores)

    def parameter_vs_mae(start, end, step, parameter_name):
        results = {}
        for i in range(start, end, step):
            results[i] = mae_scores_min_sample_split(lista, {'max_depth': i})
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(list(results.keys()), list(results.values()))
        plt.title("%s vs MAE" % parameter_name, fontweight="bold", fontsize=15)
        plt.xlabel("%s" % parameter_name)
        plt.ylabel("Mean Absolute Error")
        plt.grid(True)
        plt.show()
        return

    parameter_vs_mae(1, 10, 1, {'Max Depth'})

    return


    #mae_scores_min_sample_split(list, 'max_depth', i, 1, 2)
    #parameter_vs_mae(2, 20, 1, mae_scores_min_sample_split(list, {'min_samples_split'}), 'Minimum Samples Split')



"""def tryfeatures(feature, range):
    tree = DecisionTreeRegressor(**{feature: range})
    return

    tryfeatures('max_depth', 5)"""
