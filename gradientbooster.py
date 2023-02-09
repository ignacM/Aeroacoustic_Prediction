from sklearn.ensemble import GradientBoostingRegressor
from decision_tree import print_actual_vs_real
from decision_tree import print_regression_solutions
import tensorflow as tf
import keras
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import SGDRegressor

def runGBM(X, Y, model=GradientBoostingRegressor):
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    """Set parameters to be optimized:"""
    """parameters = {'n_estimators': [1, 350, 5],
                  'learning_rate': [0.01, 0.5, 0.02]
                  }"""
    parameters = {'n_estimators': [1, 200, 20],
                  'learning_rate': [0.05, 0.5, 0.05]
                  }

    def mae_scores(arguments):

        regressor = model(**arguments)
        scores = -1 * cross_val_score(regressor, x_train, y_train, cv=3,
                                      scoring='neg_mean_absolute_error')
        regressor.fit(x_train, y_train)
        ypred_train = regressor.predict(x_train)
        ypred_test = regressor.predict(x_test)

        train_error = mae(y_train, ypred_train)
        test_error = mae(y_test, ypred_test)
        return np.mean(scores), train_error, test_error


    def parameter_loss_mae(start, end, step, parameter_name):
        results = {}
        train_errors = {}
        test_errors = {}
        if type(start) == float:
            start = int(start*100)
            end = int(end*100)
            step = int(step*100)
            for i in range(start, end, step):
                results[i/100], train_errors[i/100], test_errors[i/100] = mae_scores({parameter_name: i/100})
        else:
            for i in range(start, end, step):
                results[i], train_errors[i], test_errors[i] = mae_scores({parameter_name: i})
        """ Plotting Parameter vs Mean Absolute Error"""
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(list(results.keys()), list(results.values()))
        plt.title("%s vs MAE" % parameter_name, fontweight="bold", fontsize=25)
        plt.xlabel("%s" % parameter_name)
        plt.ylabel("Mean Absolute Error")
        plt.grid(True)
        plt.show()

        """ Plotting Train vs Test Error"""
        fig, ax = plt.subplots(figsize=(8, 6))
        plt.plot(list(train_errors.keys()), list(train_errors.values()))
        plt.plot(list(test_errors.keys()), list(test_errors.values()))
        plt.title("Train vs Test Error for %s" % parameter_name, fontweight="bold", fontsize=25)
        plt.xlabel("%s" % parameter_name)
        plt.ylabel("Error")
        plt.legend(["Training Error", "Test Error"])
        plt.grid(True)
        plt.show()
        return test_errors


    for i in range(0, len(parameters)):
        parameter_name = list(parameters)[i]
        start = list(parameters.values())[i][0]
        end = list(parameters.values())[i][1]
        step = list(parameters.values())[i][2]

        results = parameter_loss_mae(start, end, step, parameter_name)
        sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        """var = tf.Variable(2.5)
        cost = lambda: 2 + var ** 2

        for _ in range(100):
            sgd.minimize()"""

        X_sgd = np.array(list(results.values()))
        Y_sgd = np.array(list(results.keys()))
        X_sgd = X_sgd.reshape(-1, 1)

        #optimizer.fit(X_sgd, Y_sgd.ravel())


        #param = int(input('What is the best %s?' % parameter_name))

        """parameter_loss_mae(0.01, 0.3, 0.01, 'learning_rate')
        lr = int(input('What is the best learning rate?'))"""

    return

