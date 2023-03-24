import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    df = pd.read_excel(path, sheet_name='pol')
    # minimum = min(df['db'])
    # maximum = max(df['db'])

    # Selecting which set to use as test data
    exclude = exclude
    df['exclude'] = 0
    df.loc[(df['degree'] == exclude), 'exclude'] = 1
    exclude_col = df.loc[:, ['exclude']]

    Y_data = df.drop(['theta', 'degree'], axis=1)

    Y_data['db'] = np.log(Y_data['db'])

    scaler = scaler
    df = scaler.fit_transform(df)
    df = pd.DataFrame(df, columns=['db', 'theta', 'degree', 'exclude'])
    df['exclude'] = exclude_col

    X_data = df.drop(['db'], axis=1)
    X_data = X_data.to_numpy()

    x_train = X_data[(X_data[:, -1] == 0), 0:2]
    x_test = X_data[(X_data[:, -1] == 1), 0:2]

    y_train = Y_data.loc[(Y_data['exclude'] == 0), 'db']
    y_test = Y_data.loc[(Y_data['exclude'] == 1), 'db']

    return x_train, y_train, x_test, y_test


if __name__ == '__main__':

    path = '..\Data\dec22\data_input.xlsx'
    df = pd.read_excel(path, sheet_name='pol')
    labels = [
        '0 degrees',
        '15 degrees',
        '30 degrees',
        '45 degrees',
        '60 degrees',
        '75 degrees',
        '90 degrees',
        '105 degrees',
        '120 degrees',
        '135 degrees',
        '150 degrees',
        '165 degrees'
    ]

    def print_predictions(y_pred, ypred_labels):
        """
        Plots a graph of predicted values. Fits line to points.
        :param y_pred:
        :param ypred_labels: array of names of ypred
        :return:
        """
        fig, ax = plt.subplots(figsize=(7, 6))
        # Plotting test vs predicted data
        for item in np.arange(0, 180, 15):
            plt.plot(np.arange(1,24,1), df.loc[df['degree'] == item, ['db']], linewidth=1.5)

        plt.title("Simulated  sound data", fontweight="bold", fontsize=15)
        plt.xlabel('Microphone number')
        plt.ylabel('Sound Pressure Level, dB')
        plt.legend(ypred_labels)
        plt.grid(True)
        plt.show()

    print_predictions(df, labels)