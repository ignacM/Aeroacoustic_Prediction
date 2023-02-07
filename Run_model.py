import pandas as pd
from tree_search import runmodel
from decision_tree_final import runTree
from sklearn.tree import DecisionTreeRegressor
from decision_tree import print_actual_vs_real


def compare_trends(x_train, y_train, x_test, y_test):
    tre = DecisionTreeRegressor(max_depth=8, min_samples_split=2, ccp_alpha=0.02)
    tre.fit(x_train, y_train)
    ypred = tre.predict(x_test)
    print_actual_vs_real(y_test, ypred)
    return


if __name__ == '__main__':

    path = 'Data\combined_data.xlsx'
    df = pd.read_excel(path)
    X_data = df.iloc[:253, 1:]
    Y_data = df.iloc[:, 0]
    # runmodel(X_data, Y_data)
    # runTree(X_data, Y_data)
    # best Tree parameters for combined data:
    """tre = DecisionTreeRegressor(max_depth=8, min_samples_split=2, ccp_alpha=0.02)"""

    exclude = 60

    newX = X_data.drop(X_data.index[X_data['degree'] == exclude])
    newY = df.drop(df.index[df['degree'] == exclude]).reset_index(drop=True)
    newY = newY.iloc[:230, 0]
    # 230 since its the half of the dataset
    x_test = X_data.drop(X_data.index[X_data['degree'] != exclude])
    y_test = df.drop(df.index[df['degree'] != exclude]).reset_index(drop=True)
    y_test = y_test.iloc[:23, 0]

    compare_trends(newX, newY, x_test, y_test)






"""
Try GD like Neider Mead
Minimizing the cost function
Other methods like Bayesian optimization
"""