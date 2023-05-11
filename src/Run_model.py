import pandas as pd
from src.functions.regression_eval import print_test_vs_real
from gradientbooster import run_model_variance, optimize
from sklearn.ensemble import GradientBoostingRegressor
from skopt import gp_minimize
from skopt import space
from functools import partial


def compare_trends(x_train, y_train, x_test, y_test):
    tre = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05)
    tre.fit(x_train, y_train)
    ypred = tre.predict(x_test)
    print_test_vs_real(y_test, ypred)
    return


if __name__ == '__main__':

    path = '..\Data\dec22\data_input.xlsx'
    df = pd.read_excel(path)
    X_data = df.iloc[:253, 1:]
    Y_data = df.iloc[:253, 0]

    parameter_range = {'n_estimators': [10, 200, 20],
                  'learning_rate': [0.01, 0.5, 0.03]
                  }

    run_model_variance(X_data, Y_data, parameter_range)
    """param_list = {'n_estimators': np.arange(50, 200, 50),
                  'learning_rate': np.arange(0.01, 0.1, 0.02),
                  'loss': ['squared_error', 'absolute_error', 'huber', 'quantile']}"""

    # define a parameter space
    param_space = [
        space.Integer(3, 15, name="max_depth"), space.Integer(50, 300, name="n_estimators"),
        space.Categorical(['squared_error', 'absolute_error', 'huber', 'quantile'], name="loss"),
        space.Real(0.01, 1, prior="uniform", name="learning_rate")
        ]

    # inside the main function
    param_names = [
        "max_depth",
        "n_estimators",
        'loss',
        "learning_rate"
        ]

    optimization_function = partial(optimize, param_names=param_names, x=X_data, y=Y_data)

    result = gp_minimize(optimization_function, dimensions=param_space, n_calls=15,
                         n_random_starts=10, verbose=10)
    # create best params dict and print it
    best_params = dict(zip(param_names, result.x))

    print(best_params)

    #optimize(X_data, Y_data, param_space, param_names)

    # runTree(X_data, Y_data)
    # best Tree parameters for combined data:
    #tre = DecisionTreeRegressor(max_depth=8, min_samples_split=2, ccp_alpha=0.02)
    #gb = GradientBoostingRegressor(n_estimators=70, learning_rate=0.05)
    exclude = 60

    newX = X_data.drop(X_data.index[X_data['degree'] == exclude])
    newY = df.drop(df.index[df['degree'] == exclude]).reset_index(drop=True)
    newY = newY.iloc[:230, 0]
    # 230 since its the half of the dataset
    x_test = X_data.drop(X_data.index[X_data['degree'] != exclude])
    y_test = df.drop(df.index[df['degree'] != exclude]).reset_index(drop=True)
    y_test = y_test.iloc[:23, 0]

    # compare_trends(newX_train, newY_train, x_test, y_test)






"""
Try GD like Neider Mead
Minimizing the cost function
Other methods like Bayesian optimization
"""