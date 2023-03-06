import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def MinMax(x):
    min = np.min(x)
    max = np.max(x)
    range = max - min
    return [(a - min) / range for a in x], min, max


def deNormalize(normalizedx, lowerbound, upperbound):
    range = upperbound - lowerbound
    list = (a * range + lowerbound for a in normalizedx)
    return pd.Series(list)


def dataSplit(exclude=60, scaler=MinMaxScaler()):
    """
    Splits data into train and test data.
    :param exclude: Angle to be excluded from dataset: [0, 15, 30, ... 150, 165]
    :param scaler: select scaler from sklearn.preprocessing
    :return: x_train, y_train, x_test, y_test,
            min, max (min and max returned for descaling)
    """
    # Dividing data between X and Y
    path = '..\Data\dec22\data_input.xlsx'
    # Sheet name either azi or pol
    df = pd.read_excel(path, sheet_name='azi')
    minimum = min(df['db'])
    maximum = max(df['db'])

    # Selecting which set to use as test data
    exclude = exclude
    df['exclude'] = 0
    df.loc[(df['degree'] == exclude), 'exclude'] = 1
    exclude_col = df.loc[:, ['exclude']]



    scaler = scaler
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=['db', 'theta', 'degree', 'exclude'])
    df['exclude'] = exclude_col

    Y_data = df.drop(['theta', 'degree'], axis=1)

    X_data = df.drop(['db'], axis=1)
    X_data = X_data.to_numpy()

    x_train = X_data[(X_data[:, -1] == 0), 0:2]
    x_test = X_data[(X_data[:, -1] == 1), 0:2]

    y_train = Y_data.loc[(Y_data['exclude'] == 0), 'db']
    y_test = Y_data.loc[(Y_data['exclude'] == 1), 'db']

    return x_train, y_train, x_test, y_test, minimum, maximum

