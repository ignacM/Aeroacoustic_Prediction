import pandas as pd
import numpy as np
from tree_search import runmodel
from decision_tree_final import runTree
from sklearn.tree import DecisionTreeRegressor
from ..functions.regression_eval import print_test_vs_real
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
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline


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



    # pipeline class is used as estimator to enable
    # search over different model types
    pipe = Pipeline([('model', DecisionTreeRegressor())])

    GPR_search = {
        'model': space.Categorical([GaussianProcessRegressor()]),
        'model__alpha': space.Real(0.000001, 0.01, prior="uniform", name="alpha"),
    }

    # explicit dimension classes can be specified like this
    GBR_search = {
        'model': space.Categorical([GradientBoostingRegressor()]),
        'model__max_depth': space.Integer(1, 5, name='max_depth'),
        'model__min_samples_split': space.Integer(2, 100, name='min_samples_split'),
        'model__min_samples_leaf': space.Integer(1, 100, name='min_samples_leaf'),
        'model__loss': space.Categorical(['huber', 'quantile'], name='loss'),
        'model__alpha': space.Real(10 ** -5, 0.1, name='alpha')
    }

    SVR_search = {
        'model': space.Categorical([SVR()]),
        'model__C': space.Real(65, 90, name='C'),
        'model__epsilon': space.Real(0.350, 0.500, name='epsilon'),
        'model__gamma': space.Categorical(['auto', 'scale'], name='gamma')
    }

    opt = BayesSearchCV(
        pipe,
        # (parameter space, # of evaluations)
        [(GBR_search, 10), (GPR_search, 1), (SVR_search, 10)],
        cv=3
    )

    opt.fit(x_train, y_train)

    print("val. score: %s" % opt.best_score_)
    print("test score: %s" % opt.score(x_test, y_test))
    print("best params: %s" % str(opt.best_params_))









"""
Try GD like Neider Mead
Minimizing the cost function
Other methods like Bayesian optimization
"""