import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from train_test_split import deNormalize



if __name__ == '__main__':


    # Load model:
    regressor = joblib.load('../models/SVM_regressor.pkl')
    x_val = pd.read_csv('../Data/dec22/x_validation.csv')

    # Scale data
    scaler = MinMaxScaler()
    x_val = scaler.fit_transform(x_val)

    ypred = regressor.predict(x_val)
    ypred = np.exp(ypred)
    print(ypred)








# def dataSplit(exclude=60, scaler=MinMaxScaler()):
#     """
#     Splits data into train and test data.
#     :param exclude: Angle to be excluded from dataset: [0, 15, 30, ... 150, 165]
#     :param scaler: select scaler from sklearn.preprocessing
#     :return: x_train, y_train, x_test, y_test,
#             min, max (min and max returned for descaling)
#     """
#     # Dividing data between X and Y
#     path = '..\Data\dec22\data_input.xlsx'
#     # Sheet name either azi or pol
#     df = pd.read_excel(path, sheet_name='azi')
#     minimum = min(df['db'])
#     maximum = max(df['db'])
#
#     # Selecting which set to use as test data
#     exclude = exclude
#     df['exclude'] = 0
#     df.loc[(df['degree'] == exclude), 'exclude'] = 1
#     exclude_col = df.loc[:, ['exclude']]
#
#
#
#     scaler = scaler
#     df = scaler.fit_transform(df)
#     df = pd.DataFrame(df, columns=['db', 'theta', 'degree', 'exclude'])
#     df['exclude'] = exclude_col
#
#     Y_data = df.drop(['theta', 'degree'], axis=1)
#
#     X_data = df.drop(['db'], axis=1)
#     X_data = X_data.to_numpy()
#
#     x_train = X_data[(X_data[:, -1] == 0), 0:2]
#     x_test = X_data[(X_data[:, -1] == 1), 0:2]
#
#     y_train = Y_data.loc[(Y_data['exclude'] == 0), 'db']
#     y_test = Y_data.loc[(Y_data['exclude'] == 1), 'db']
#
#     return x_train, y_train, x_test, y_test, minimum, maximum

