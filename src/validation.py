import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from train_test_split import deNormalize
from functions import regression_eval as reval
from sklearn.metrics import mean_absolute_error as mae



if __name__ == '__main__':

    # Load model:
    svm = joblib.load('../models/SVM_regressor.pkl')
    gpr = joblib.load('../models/GPR_regressor.pkl')
    gbr = joblib.load('../models/GBR_regressor.pkl')
    scaler = joblib.load('../models/scaler.pkl')
    azimuthal_df = pd.read_csv('../Data/dec22/x_validation_polar.csv')


    azi_225_data = azimuthal_df.iloc[:23, 1:3]
    azi_225_data_scaled = scaler.transform(azi_225_data)
    azi_825_data = azimuthal_df.iloc[23:, 1:3]



    azi_225_actual = azimuthal_df.iloc[:23, 3]
    print(azi_225_actual)
    azi_825_actual = azimuthal_df.iloc[23:, 3]

    svm225 = np.exp(svm.predict(azi_225_data_scaled).tolist())
    gpr225 = np.exp(gpr.predict(azi_225_data_scaled).tolist())
    gbr225 = np.exp(gbr.predict(azi_225_data_scaled).tolist())

    """predictions225 = [0, 0, 0]
    predictions225[0] = svm225
    predictions225[1] = gpr225
    predictions225[2] = gbr225"""
    predictions225 = [svm225, gpr225, gbr225]
    predictions225_names = ['SVM', 'GPR', 'GBR']



    reval.plot_preds_vs_real(predictions225, azi_225_actual, predictions225_names)


    svm_val_error = mae(azi_225_actual, svm225)
    print('SVM Validation loss is:', svm_val_error)

    gpr_val_error = mae(azi_225_actual, gpr225)
    print('GPR Validation loss is:', gpr_val_error)

    gbr_val_error = mae(azi_225_actual, gbr225)
    print('GBR Validation loss is:', gbr_val_error)



