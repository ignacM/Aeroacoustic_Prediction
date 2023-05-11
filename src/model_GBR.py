import pandas as pd
import numpy as np
from src.functions.regression_eval import print_actual_vs_real
from sklearn.ensemble import GradientBoostingRegressor
from skopt import gp_minimize, BayesSearchCV
from skopt import space
from sklearn.model_selection import cross_val_score
from skopt.utils import use_named_args
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import MinMaxScaler

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
    param_space = [space.Integer(1, 15, name='max_depth'),
                   space.Integer(100, 1500, name='n_estimators'),
                   space.Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
                   space.Integer(2, 200, name='min_samples_split'),
                   space.Integer(1, 100, name='min_samples_leaf'),
                   space.Categorical(['huber', 'quantile'], name='loss'),
                   space.Real(10**-5, 0.99,  name='alpha')]

    param_names = ['max_depth', 'n_estimators', 'learning_rate', 'min_samples_split', 'min_samples_leaf', 'loss', 'alpha']
    #Kernel = ConstantKernel(constant_value=1.4)
    #Kernel = Exponentiation(kernel=ConstantKernel(), exponent=1.1)

    @use_named_args(param_space)
    def optimize_2(**params):
        regressor = GradientBoostingRegressor(**params)
        return -np.mean(cross_val_score(regressor, x_train, y_train, cv=3, n_jobs=-1, scoring="neg_mean_absolute_error"))

    # inside the main function
    res_gp = gp_minimize(optimize_2, dimensions=param_space, n_calls=40, verbose=10, n_initial_points=40)


    best_parameters = dict(zip(param_names, res_gp.x))
    print(best_parameters)

    def compare_trends(x_train, y_train, x_test, y_test):
        tre = GradientBoostingRegressor(**best_parameters)
        tre.fit(x_train, y_train)
        ypred = tre.predict(x_test)
        print_actual_vs_real(y_test, ypred)
        return

    compare_trends(x_train, y_train, x_test, y_test)
    GBR_search = {
        'model': space.Categorical([GradientBoostingRegressor()]),
        'model__max_depth': space.Integer(1, 15, name='max_depth'),
        'model__n_estimators': space.Integer(100, 1500, name='n_estimators'),
        'model__min_samples_split': space.Integer(2, 100, name='min_samples_split'),
        'model__min_samples_leaf': space.Integer(1, 100, name='min_samples_leaf'),
        'model__loss': space.Categorical(['huber', 'quantile'], name='loss'),
        'model__alpha': space.Real(10 ** -5, 0.99, name='alpha')
    }

    pipe = Pipeline([
        ('model', GradientBoostingRegressor())
    ])
    res_bp = BayesSearchCV(pipe, [(GBR_search, 40)], cv=3)
    res_bp.fit(x_train, y_train)

    """best_parameters = dict(zip(param_names, res_bp.best_params_))
    params = best_parameters[1:]"""
    params = dict(res_bp.best_params_)
    print(params)
    best_parameters = params['model'].get_params()
    print(type(best_parameters))
    compare_trends(x_train, y_train, x_test, y_test)





"""
Try GD like Neider Mead
Minimizing the cost function
Other methods like Bayesian optimization
"""