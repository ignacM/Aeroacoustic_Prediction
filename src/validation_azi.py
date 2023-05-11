import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from train_test_split import deNormalize
from functions import regression_eval as reval
from sklearn.metrics import mean_absolute_error as mae



if __name__ == '__main__':

    # Load models:
    svm_pol = joblib.load('../models/SVM_regressor_azi.pkl')
    gpr_pol = joblib.load('../models/GPR_regressor_azi.pkl')
    gbr_pol = joblib.load('../models/GBR_regressor_azi.pkl')
    scaler_pol = joblib.load('../models/scaler_azi.pkl')



    polar_df = pd.read_csv('../Data/dec22/x_validation_azimuthal.csv')

    # Selecting Features data and Scaling
    pol_225_data = polar_df.iloc[:23, 1:3]
    pol_225_data_scaled = scaler_pol.transform(pol_225_data)

    pol_825_data = polar_df.iloc[23:, 1:3]
    pol_825_data_scaled = scaler_pol.transform(pol_825_data)

    # Selecting Real data
    pol_225_actual = polar_df.iloc[:23, 3]
    pol_825_actual = polar_df.iloc[23:, 3]
    pol_825_actual = pol_825_actual.reset_index(drop=True)


    # Predicting 22.5 degrees data
    svm225 = np.exp(svm_pol.predict(pol_225_data_scaled).tolist())
    gpr225 = np.exp(gpr_pol.predict(pol_225_data_scaled).tolist())
    gbr225 = np.exp(gbr_pol.predict(pol_225_data_scaled).tolist())

    # Predicting 82.5 degrees data
    svm825 = np.exp(svm_pol.predict(pol_825_data_scaled).tolist())
    gpr825 = np.exp(gpr_pol.predict(pol_825_data_scaled).tolist())
    gbr825 = np.exp(gbr_pol.predict(pol_825_data_scaled).tolist())

    # Plotting 22.5 degrees predictions
    predictions_names = ['SVM', 'GPR', 'GBR']
    predictions225 = [svm225, gpr225, gbr225]
    predictions225_dataframe = pd.DataFrame(predictions225, index=predictions_names)

    reval.plot_preds_vs_real(predictions225, pol_225_actual, predictions_names, '22.5 degrees')
    reval.plot_regression_outcome(pol_225_actual, svm225, 'SVM Regressor')
    reval.print_regression_residuals(pol_225_actual, svm225, 'SVM Regressor')

    print('22.5 Degrees Loss:')

    svm_val_error = mae(pol_225_actual, svm225)
    print('SVM Validation loss is:', svm_val_error)

    gpr_val_error = mae(pol_225_actual, gpr225)
    print('GPR Validation loss is:', gpr_val_error)

    gbr_val_error = mae(pol_225_actual, gbr225)
    print('GBR Validation loss is:', gbr_val_error)

    reval.r2(pol_225_actual, svm225, 'SVM')
    reval.r2(pol_225_actual, gpr225, 'GPR')
    reval.r2(pol_225_actual, gbr225, 'GBR')
    print('........')


    # Plotting 82.5 degrees predictions
    predictions825 = [svm825, gpr825, gbr825]
    predictions825_dataframe = pd.DataFrame(predictions825, index=predictions_names)

    reval.plot_preds_vs_real(predictions825, pol_825_actual, predictions_names, '82.5 degrees')
    print('82.5 Degrees Loss:')

    svm_val_error = mae(pol_825_actual, svm825)
    print('SVM Validation loss is:', svm_val_error)

    gpr_val_error = mae(pol_825_actual, gpr825)
    print('GPR Validation loss is:', gpr_val_error)

    gbr_val_error = mae(pol_825_actual, gbr825)
    print('GBR Validation loss is:', gbr_val_error)
    print('........')

    reval.r2(pol_825_actual, svm825, 'SVM')
    reval.r2(pol_825_actual, gpr825, 'GPR')
    reval.r2(pol_825_actual, gbr825, 'GBR')

    """# Predicting 22.5 degrees data
    svm225 = np.exp(svm_azi.predict(azi_225_data_scaled).tolist())
    gpr225 = np.exp(gpr_azi.predict(azi_225_data_scaled).tolist())
    gbr225 = np.exp(gbr_azi.predict(azi_225_data_scaled).tolist())

    # Predicting 82.5 degrees data
    svm825 = np.exp(svm_azi.predict(azi_825_data_scaled).tolist())
    gpr825 = np.exp(svm_azi.predict(azi_825_data_scaled).tolist())
    gbr825 = np.exp(svm_azi.predict(azi_825_data_scaled).tolist())
    
    # Load models
    svm_azi = joblib.load('../models/SVM_regressor_azi.pkl')
    gpr_azi = joblib.load('../models/GPR_regressor_azi.pkl')
    gbr_azi = joblib.load('../models/GBR_regressor_azi.pkl')
    scaler_azi = joblib.load('../models/scaler_azi.pkl')

    # Read dataset
    azimuthal_df = pd.read_csv('../Data/dec22/x_validation_azimuthal.csv')

    # Selecting Features data and Scaling
    azi_225_data = azimuthal_df.iloc[:23, 1:3]
    azi_225_data_scaled = scaler_azi.transform(azi_225_data)

    azi_825_data = azimuthal_df.iloc[23:, 1:3]
    azi_825_data_scaled = scaler_azi.transform(azi_825_data)

    # Selecting Real data
    azi_225_actual = azimuthal_df.iloc[:23, 3]
    azi_825_actual = azimuthal_df.iloc[23:, 3]
    azi_825_actual = azi_825_actual.reset_index(drop=True)"""

