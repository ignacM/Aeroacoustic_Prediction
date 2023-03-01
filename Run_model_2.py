import pandas as pd
import numpy as np
from tree_search import runmodel
from decision_tree_final import runTree
from sklearn.tree import DecisionTreeRegressor
from decision_tree import print_actual_vs_real
from gradientbooster import run_model_variance, optimize
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from skopt import gp_minimize
from skopt import space
from sklearn.model_selection import cross_val_score
from sklearn import ensemble
from skopt.utils import use_named_args

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, ConstantKernel, Exponentiation

from sklearn.preprocessing import StandardScaler, MinMaxScaler

if __name__ == '__main__':

    # Dividing data between X and Y
    path = 'Data\combined_data.xlsx'
    df_azi = pd.read_excel(path, sheet_name='pol')
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


    # define a parameter space for SVR:
    # https://uk.mathworks.com/help/stats/fitcsvm.html?s_tid=srchtitle#namevaluepairarguments
    # C: manual box constraint
    """param_space = [space.Real(0.25, 10, name='C'),
                   space.Real(0.01, 1, name='epsilon'),
                   space.Categorical(['auto', 'scale'], name='gamma')]"""
    param_space = [space.Real(65, 90, name='C'),
                   space.Real(0.350, 0.500, name='epsilon'),
                   space.Categorical(['auto', 'scale'], name='gamma')
                   ]

    param_names = ['C', 'epsilon', 'gamma']


    def gaussian_kernel(X, Y):
        return np.exp(-abs(abs(X-Y))**2)

    @use_named_args(param_space)
    def optimize_2(**params):
        regressor = SVR(**params)
        return -np.mean(cross_val_score(regressor, x_train, y_train, cv=5, n_jobs=-1, scoring="neg_mean_absolute_error"))

    # inside the main function
    res_gp = gp_minimize(optimize_2, dimensions=param_space, n_calls=50, verbose=10, n_initial_points=20)

    best_parameters = dict(zip(param_names, res_gp.x))
    print(best_parameters)

    def compare_trends(x_train, y_train, x_test, y_test):
        tre = SVR(**best_parameters)
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