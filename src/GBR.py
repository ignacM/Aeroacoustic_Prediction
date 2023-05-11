import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import sklearn.gaussian_process.kernels as kernels
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
    # GPR seemed to have the best performance when Min-max scaling of x data and log scale on y


    # Define parameter names and space

    param_names = ['learning_rate', 'n_estimators', 'alpha']

    """param_space = [
        space.Real(0.14, 0.17, name='learning_rate'),
        space.Integer(30, 40, name='n_estimators'),
        space.Real(0.4, 0.5, name='alpha'),
    ]"""

    param_space = [
        space.Real(0.4, 0.7, name='learning_rate'),
        space.Integer(30, 40, name='n_estimators'),
        space.Real(0.3, 0.6, name='alpha'),

    ]

    """space.Integer(2, 3, name='max_depth'),"""
    """space.Categorical(['squared_error', 'absolute_error', 'huber', 'quantile'], name='loss'),"""
    @use_named_args(param_space)
    def optimize_mae_function(**params):
        """
        Fits an GP regressor and returns an average of MAE over a number of cross validations
        :param params:
        :return:
        """
        regressor = GradientBoostingRegressor(**params, loss='huber', max_depth=3)
        return -np.mean(cross_val_score(regressor, x_train, y_train, cv=3, n_jobs=-1, scoring="neg_mean_absolute_error"))

    # Use Bayesian Optimization with Gaussian process to find function minimum
    res_gp = gp_minimize(optimize_mae_function, dimensions=param_space, n_calls=30, verbose=10, n_initial_points=20)

    # Save parameters fround in global minimum
    best_parameters = dict(zip(param_names, res_gp.x))
    print(res_gp.x)
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
        regressor = GradientBoostingRegressor(**best_parameters)
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

    plot_regression_outcome(y_test, ypred, 'GBR Regressor')
    print_regression_residuals(y_test, ypred, 'GBR Regressor')

    joblib.dump(final_model, '../models/GBR_regressor_azi.pkl')





"""
Try GD like Neider Mead
Minimizing the cost function
Other methods like Bayesian optimization
"""