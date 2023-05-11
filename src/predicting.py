import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from train_test_split import deNormalize
from functions import regression_eval as reval



if __name__ == '__main__':

    # Load model:
    regressor = joblib.load('../models/SVM_regressor.pkl')
    # regressor = joblib.load('../models/GPR_regressor.pkl')
    # regressor = joblib.load('../models/GBR_regressor.pkl')
    df = pd.read_csv('../Data/dec22/x_validation_nolabels.csv')


    # Scale data
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)

    predictions = []
    dicta = {}
    predictions_name =\
        [
            '7.5 degrees', '22.5 degrees', '37.5 degrees', '52.5 degrees', '67.5 degrees', '82.5 degrees',
            '97.5 degrees', '112.5 degrees', '127.5 degrees', '142.5 degrees', '157.5 degrees', '172.5 degrees'
        ]
    for degree in range(12):
        a = degree*23

        #x_val = np.array(x_val).reshape(-1, 1)
        x_val = df[a:a + 23, [0, 1]]
        print(x_val)

        ypred = regressor.predict(x_val)
        # Models have been trained with log scale dB data, reversing log:
        ypred = np.exp(ypred)
        dicta[predictions_name[degree]] = ypred
        predictions.append(ypred)


# print('These are the preditcions:', predictions)



reval.print_predictions(predictions, predictions_name)

predictions = []
predictions.append(dicta['22.5 degrees'])
# predictions.append(dicta['82.5 degrees'])
reval.print_predictions(predictions, ['22.5 degrees'])
# reval.print_predictions(predictions, ['22.5 degrees', '82.5 degrees'])

