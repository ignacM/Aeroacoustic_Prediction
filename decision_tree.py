import pandas as pd

from sklearn.preprocessing import scale
from sklearn.tree import DecisionTreeRegressor

"""Load and Visualize Data"""

path = '\Data\preliminary_data.xlsx'

df = pd.read_excel(path, sep=',')
df.describe()
df.head()

# Scaling data
Xs = scale(X)
Ys = scale(Y)
from sklearn.decomposition import PCA

feature_names = list(X.columns)
pca = PCA(n_components=4)
Xs_pca = pca.fit_transform(Xs)
# Only retain the first two modes of PCA as the new features
PCA_df = pd.DataFrame()
PCA_df['PCA_1'] = Xs_pca[:, 0]
PCA_df['PCA_2'] = Xs_pca[:, 1]


def runTree():
    # Splitting test and train data into 10%
    xtrain, xtest, ytrain, ytest = train_test_split(Xs_pca, Ys, test_size=0.10)
    # Finding an optimized decision tree
    # Using max_depth, criterion will suffice for DT Models, rest all will remain constant
    parameters = {'max_depth': (9, 11)
        , 'max_features': ('sqrt', 'log2')
        , 'min_samples_split': (2, 4)
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
