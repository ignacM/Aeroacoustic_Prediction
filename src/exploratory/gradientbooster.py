from sklearn.ensemble import GradientBoostingRegressor
from decision_tree import print_actual_vs_real
from decision_tree import print_regression_solutions
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mlxtend.evaluate import bias_variance_decomp


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn import model_selection, metrics, ensemble
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt import space

def run_model_variance(X, Y, parameters, model=GradientBoostingRegressor):
    """
    Runs a regression task. parameters and model to be specified.
    :param X:
    :param Y:
    :param parameters: parameters to be optimized from the given model. give a list as:
    parameters = {'param1': [start, end, step]
                'param2': ...}
    :param model: specify regressor to be used (DecisionTreeRegressor, GradientBoostingRegressor...)
    :return:
    """
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

    parameters = parameters

    def mae_scores(arguments):
        """
        Takes arguments of a model and trains the model.
        :param arguments: parameters of the regressor
        :return:
        """

        regressor = model(**arguments)
        scores = -1 * cross_val_score(regressor, x_train, y_train, cv=3,
                                      scoring='neg_mean_squared_error')
        regressor.fit(x_train, y_train)
        ypred_train = regressor.predict(x_train)
        ypred_test = regressor.predict(x_test)

        train_error = mae(y_train, ypred_train)
        test_error = mae(y_test, ypred_test)
        """loss, bias, variance = bias_variance_decomp(
            regressor, x_train.values, y_train.values, x_test.values, y_test.values,
            loss='mse', random_seed=123)"""
        return np.mean(scores), train_error, test_error

    def bias_variance(arguments):
        """
        Given model arguments, returns bias and variance of the model
        :param arguments:
        :return: loss, bias and variane of model with specific arguments
        """
        regressor = model(**arguments)
        loss, bias, variance = bias_variance_decomp(
            regressor, x_train.values, y_train.values, x_test.values, y_test.values,
            loss='mse', random_seed=123)
        return loss, bias, variance

    def parameter_loss_mae(start, end, step, parameter_name, *args):
        results = {}
        train_errors = {}
        test_errors = {}
        loss = {}
        bias = {}
        variance = {}

        # Selecting how often to produce variance / bias calculations. Selected to only calculate every 4 steps.
        varray = np.arange(start, end, step*4)
        if type(start) == float:
            start = int(start*100)
            end = int(end*100)
            step = int(step*100)
            for i in range(start, end, step):
                grid = {parameter_name: i / 100}
                if not bool(args):  # Protective statement to add parameters to dictionary if dictionary is not empty
                    grid.update(args)
                results[i/100], train_errors[i/100], test_errors[i/100] = mae_scores(grid)
                if i/100 in varray:
                    grid = {parameter_name: i/100}
                    if not bool(args):
                        grid.update(args)
                    loss[i / 100], bias[i / 100], variance[i / 100] = bias_variance(grid)
        else:
            for i in range(start, end, step):
                grid = {parameter_name: i}
                if not bool(args):
                    grid.update(args)
                results[i], train_errors[i], test_errors[i] = mae_scores(grid)
                if i in varray:
                    grid = {parameter_name: i}
                    if not bool(args):
                        grid.update(args)
                    loss[i], bias[i], variance[i] = bias_variance(grid)


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
        plt.plot(list(variance.keys()), list(variance.values()))
        plt.plot(list(bias.keys()), list(bias.values()))

        plt.title("Train vs Test Error for %s" % parameter_name, fontweight="bold", fontsize=25)
        plt.xlabel("%s" % parameter_name)
        plt.ylabel("Error")
        plt.legend(["Training Error", "Test Error", "Variance", "Bias"])
        plt.grid(True)
        plt.show()
        return test_errors


    final_parameters = {}
    for i in range(0, len(parameters)):
        parameter_name = list(parameters)[i]
        start = list(parameters.values())[i][0]
        end = list(parameters.values())[i][1]
        step = list(parameters.values())[i][2]

        test_errors = parameter_loss_mae(start, end, step, parameter_name, final_parameters)
        parameter_value = float(input('Where is the %s?' % parameter_name))

        final_parameters[parameter_name] = parameter_value



    return

def optimize(x, y, params, param_names):

     # convert params to dictionary
     parameters = dict(zip(param_names, params))

     # initialize model with current parameters
     # model = regressor(**parameters)
     regressor = ensemble.GradientBoostingRegressor(**parameters)
     lst_accu_stratified = []
     """skf = KFold(n_splits=10, shuffle=True, random_state=1)
     lst_accu_stratified = []

     for train_index, test_index in skf.split(x, y):
         x_train_fold, x_test_fold = x[train_index], x[test_index]
         y_train_fold, y_test_fold = y[train_index], y[test_index]
         regressor.fit(x_train_fold, y_train_fold)
         lst_accu_stratified.append(regressor.score(x_test_fold, y_test_fold))"""
     #regressor.fit(x, y)
     #lst_accu_stratified.append(regressor.score(x, y))
     # return negative accuracy
     return -np.mean(cross_val_score(regressor, x, y, cv=5, n_jobs=-1,
                                        scoring="neg_mean_absolute_error"))


