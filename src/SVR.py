import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error as mae

from skopt import gp_minimize
from skopt import space
from skopt.utils import use_named_args

from src.functions.regression_eval import print_test_vs_real, print_regression_residuals, plot_regression_outcome
from train_test_split import dataSplit, deNormalize


if __name__ == '__main__':

    # Split data into train / test. Exclude Angle 60
    x_train, y_train, x_test, y_test, scaling = dataSplit(exclude=135, scaler=MinMaxScaler())
    # SVR seemed to have the best performance when Min-max scaling of x data and log scale on y

    # define a parameter space for SVR:
    """
    C: Length of the margin. Distance between support points and margin.
    epsilon: amount of points past the support points.
    Gamma: radius of group similarity. Big gamma = points need to be very close to each other.
    """

    # Define parameter names and space
    param_names = ['C', 'epsilon', 'gamma', 'tol']

    """param_space = [
        space.Real(2, 10, name='C'),
        space.Real(0.005, 0.05, name='epsilon'),
        space.Categorical(['scale'], name='gamma'),
        space.Real(0.000000000001, 0.00005, name='tol')
    ]"""

    param_space = [
        space.Real(4, 10, name='C'),
        space.Real(0.001, 0.01, name='epsilon'),
        space.Categorical(['scale'], name='gamma'),
        space.Real(0.001, 0.01, name='tol')
    ]

    @use_named_args(param_space)
    def optimize_mae_function(**params):
        """
        Fits an SVM regressor and returns an average of MAE over a number of cross validations
        :param params:
        :return:
        """
        regressor = SVR(**params)
        return -np.mean(cross_val_score(regressor, x_train, y_train, cv=3, n_jobs=-1, scoring="neg_mean_absolute_error"))

    # Use Bayesian Optimization with Gaussian process to find function minimum
    res_gp = gp_minimize(optimize_mae_function, dimensions=param_space, n_calls=30, verbose=10, n_initial_points=20)

    # Save parameters fround in global minimum
    best_parameters = dict(zip(param_names, res_gp.x))
    print(best_parameters)


    def compare_trends(x_train, y_train, x_test, y_test):
        """
        Fits regressor with best parameters and produces a graph of y_test vs ypred
        :param x_train:
        :param y_train:
        :param x_test:
        :param y_test:
        :return: descaled y_train, descaled ypred, and fitted model
        """
        # Fitting regressor with best parameters from optimization
        regressor = SVR(**best_parameters)
        regressor.fit(x_train, y_train)
        # Predicting data
        ypred = regressor.predict(x_test)
        # Descaling the data:
        # If we were going to use Min-Max or Z-score:
        """ypred = deNormalize(ypred, min, max)
        y_test = deNormalize(y_test, min, max)"""
        # Using logarithmic scale:
        ypred = np.exp(ypred)
        y_test = np.exp(y_test)
        print_test_vs_real(y_test, ypred)

        y_train = np.exp(y_train)
        ypred_train = np.exp(regressor.predict(x_train))
        ypred_test = ypred

        train_error = mae(y_train, ypred_train)
        test_error = mae(y_test, ypred_test)
        print('Train loss is:', train_error)
        print('Test loss is:', test_error)
        return y_test, ypred, regressor

    y_test, ypred, final_model = compare_trends(x_train, y_train, x_test, y_test)

    plot_regression_outcome(y_test, ypred, 'SVM Regressor')
    print_regression_residuals(y_test, ypred, 'SVM Regressor')


    # x_train, y_train, scaling = allData()
    #
    # # Define parameter names and space
    # param_names = ['C', 'epsilon', 'gamma', 'tol']
    #
    # param_space = [
    #     space.Real(900, 1000, name='C'),
    #     space.Real(0.000005, 0.5, name='epsilon'),
    #     space.Categorical(['scale'], name='gamma'),
    #     space.Real(0.0000000000001, 0.5, name='tol')
    # ]
    #
    #
    # @use_named_args(param_space)
    # def optimize_mae_function(**params):
    #     """
    #     Fits an SVM regressor and returns an average of MAE over a number of cross validations
    #     :param params:
    #     :return:
    #     """
    #     regressor = SVR(**params)
    #     return -np.mean(
    #         cross_val_score(regressor, x_train, y_train, cv=3, n_jobs=-1, scoring="neg_mean_absolute_error"))
    #
    #
    # # Use Bayesian Optimization with Gaussian process to find function minimum
    # res_gp = gp_minimize(optimize_mae_function, dimensions=param_space, n_calls=30, verbose=10, n_initial_points=20)
    #
    # # Save parameters fround in global minimum
    # best_parameters = dict(zip(param_names, res_gp.x))
    # print(best_parameters)
    #
    # regressor = SVR(**best_parameters)
    # regressor.fit(x_train, y_train)

    joblib.dump(final_model, '../models/SVM_regressor_azi.pkl')
    joblib.dump(scaling, '../models/scaler_azi.pkl')


    # SVR_search = {
    #     'model': space.Categorical([SVR()]),
    #     'model__C': space.Real(40, 60, name='C'),
    #     'model__epsilon': space.Real(0.02,  0.2, name='epsilon'),
    #     'model__gamma': space.Categorical(['auto', 'scale'], name='gamma'),
    #     'model__tol': space.Real(0.00005, 0.001, name='tol')
    # }
    #
    # pipe = Pipeline([
    #     ('model', SVR())
    # ])
    # res_bp = BayesSearchCV(pipe, [(SVR_search, 40)], cv=3)
    # res_bp.fit(x_train, y_train)
    #
    # params = dict(res_bp.best_params_)
    #
    # best_parameters = params['model'].get_params()
    # print(best_parameters)
    # compare_trends(x_train, y_train, x_test, y_test)



"""
Try GD like Neider Mead
Minimizing the cost function
Other methods like Bayesian optimization
"""