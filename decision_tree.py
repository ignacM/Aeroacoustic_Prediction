import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error as mae

def runTree(X,Y):
    # Splitting test and train data into 10%
    xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.10)
    # Finding an optimized decision tree
    # Using max_depth, criterion will suffice for DT Models, rest all will remain constant
    parameters = {'max_depth': (13,15)
        , 'max_features': ('sqrt', 'log2')
        , 'min_samples_split': (6, 8)
                  }

    dtr_grid = RandomizedSearchCV(DecisionTreeRegressor(), param_distributions=parameters, cv=5, verbose=True)

    dtr_grid.fit(xtrain, ytrain)
    # Check accuracy of the trees
    best_model = dtr_grid.best_estimator_
    dtr_grid.best_estimator_
    # Re-build the model with best estimated tree
    dtr = best_model

    dtr.fit(xtrain, ytrain)

    print(f'Train Accuracy - : {dtr.score(xtrain, ytrain):.3f}')
    print(f'Test Accuracy - : {dtr.score(xtest, ytest):.3f}')
    score = dtr.score(xtrain, ytrain)
    print("R-squared:", score)
    ypred = dtr.predict(xtest)
    # Evaluation metrics
    mse = mean_squared_error(ytest, ypred)
    rmse = mean_squared_error(ytest, ypred) ** (1 / 2.0)
    MAE = mae(ytest, ypred)
    print("MSE: ", mse)
    print("RMSE: ", rmse)
    print("MAE:", MAE)
    # Plotting test vs predicted data
    x_ax = range(len(ytest))
    plt.plot(x_ax, ytest, linewidth=1, label="original")
    plt.plot(x_ax, ypred, linewidth=1.1, label="predicted")
    plt.title("y-test and y-predicted data")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()