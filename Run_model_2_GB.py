import pandas as pd
import numpy as np
from tree_search import runmodel
from decision_tree_final import runTree
from sklearn.tree import DecisionTreeRegressor
from decision_tree import print_actual_vs_real
from gradientbooster import run_model_variance, optimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from skopt import gp_minimize
from skopt import space
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from skopt.utils import use_named_args

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel, Exponentiation, RBF

from sklearn.preprocessing import StandardScaler, MinMaxScaler

if __name__ == '__main__':

    # Dividing data between X and Y
    path = 'Data\combined_data.xlsx'
    df_azi = pd.read_excel(path, sheet_name='azi')
    df_azi['exclude'] = 0


    exclude = 60
    df_azi.loc[(df_azi['degree'] == 60), 'exclude'] = 1

    X_data = df_azi.drop(['db'], axis=1)
    scaler = MinMaxScaler()
    X_data = scaler.fit_transform(X_data)
    Y_data = df_azi.drop(['theta', 'degree'], axis=1)

    x_train = X_data[(X_data[:, -1] == 0), 0:2]
    x_test = X_data[(X_data[:, -1] == 1), 0:2]
    y_train = Y_data.loc[(Y_data['exclude'] == 0), 'db']
    y_test = Y_data.loc[(Y_data['exclude'] == 1), 'db']

    # define a parameter space for GaussianProcessR:
    param_space = [space.Integer(1, 5, name='max_depth'),
          space.Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
          space.Integer(2, 100, name='min_samples_split'),
          space.Integer(1, 100, name='min_samples_leaf'),
          space.Categorical(['huber', 'quantile'], name='loss'),
          space.Real(10**-5, 0.1,  name='alpha')]

    param_names = ['max_depth', 'learning_rate', 'min_samples_split', 'min_samples_leaf', 'loss', 'alpha']
    #Kernel = ConstantKernel(constant_value=1.4)
    #Kernel = Exponentiation(kernel=ConstantKernel(), exponent=1.1)

    @use_named_args(param_space)
    def optimize_2(**params):
        regressor = GradientBoostingRegressor(**params)
        return -np.mean(cross_val_score(regressor, x_train, y_train, cv=5, n_jobs=-1, scoring="neg_mean_absolute_error"))

    # inside the main function
    res_gp = gp_minimize(optimize_2, dimensions=param_space, n_calls=100, verbose=10, n_initial_points=20)

    best_parameters = dict(zip(param_names, res_gp.x))
    print(best_parameters)

    def compare_trends(x_train, y_train, x_test, y_test):
        tre = GradientBoostingRegressor(**best_parameters)
        tre.fit(x_train, y_train)
        ypred = tre.predict(x_test)
        print_actual_vs_real(y_test, ypred)
        return

    compare_trends(x_train, y_train, x_test, y_test)








"""
Try GD like Neider Mead
Minimizing the cost function
Other methods like Bayesian optimization
"""