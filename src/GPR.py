import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
import sklearn.gaussian_process.kernels as kernels
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt

from skopt import gp_minimize
from skopt import space
from skopt.utils import use_named_args

from src.functions.regression_eval import print_test_vs_real, print_regression_residuals, plot_regression_outcome
from train_test_split import dataSplit, deNormalize


if __name__ == '__main__':

    # Split data into train / test. Exclude Angle 180
    x_train, y_train, x_test, y_test, scaling = dataSplit(exclude=135, scaler=MinMaxScaler())
    # GPR seemed to have the best performance when Min-max scaling of x data and log scale on y

    # Define parameter names and space
    param_names = ['alpha']
    Kernel = kernels.ConstantKernel(100, constant_value_bounds=(1e2, 1e3))\
             * kernels.RBF(1, length_scale_bounds=(1e-1, 5e-1))

    # define a parameter space for GaussianProcessR:
    param_space = [
        space.Real(0.0001, 0.01, prior="uniform", name="alpha")
                   ]

    # Kernel = kernels.ConstantKernel(constant_value=1.4)
    # Kernel = kernels.Exponentiation(kernel=ConstantKernel(), exponent=1.1)



    @use_named_args(param_space)
    def optimize_mae_function(**params):
        """
        Fits an GP regressor and returns an average of MAE over a number of cross validations
        :param params:
        :return:
        """
        regressor = GaussianProcessRegressor(**params, kernel=Kernel)
        return -np.mean(cross_val_score(regressor, x_train, y_train, cv=3, n_jobs=-1, scoring="neg_mean_absolute_error"))

    # Use Bayesian Optimization with Gaussian process to find function minimum
    res_gp = gp_minimize(optimize_mae_function, dimensions=param_space, n_calls=5, verbose=10, n_initial_points=5)

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
        regressor = GaussianProcessRegressor(**best_parameters, kernel=Kernel)
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

    plot_regression_outcome(y_test, ypred, 'GPR Regressor')
    print_regression_residuals(y_test, ypred, 'GPR Regressor')

    def gp_distribution_plot():
        X = np.arange(1,24,1)
        y = y_test
        x_train = np.arange(1,24,3)
        y_train = [y_test[0], y_test[2], y_test[5], y_test[8], y_test[11], y_test[14], y_test[17], y_test[20], y_test[21]]

        mean_prediction, std_prediction = final_model.predict(x_test, return_std=True)
        mean_prediction = np.exp(mean_prediction)
        print(mean_prediction)

        plt.plot(X, y, label=r"$f(x) = x \sin(x)$", linestyle="dotted")
        plt.scatter(x_train, y_train, label="Observations")
        plt.plot(X, mean_prediction, label="Mean prediction")
        plt.fill_between(
            np.arange(1,24,1),
            mean_prediction - 1.96 * std_prediction,
            mean_prediction + 1.96 * std_prediction,
            alpha=0.5,
            label=r"95% confidence interval",
        )
        plt.legend()
        plt.xlabel("$x$")
        plt.ylabel("$f(x)$")
        _ = plt.title("Gaussian process regression on noise-free dataset")
        plt.show()

        return


    gp_distribution_plot()

    joblib.dump(final_model, '../models/GPR_regressor.pkl')






"""
Try GD like Neider Mead
Minimizing the cost function
Other methods like Bayesian optimization
"""