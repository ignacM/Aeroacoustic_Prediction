import pandas as pd
import matplotlib as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from decision_tree import runTree


path = 'Data\preliminary_data_2.xlsx'
df = pd.read_excel(path)

df_polar = df.iloc[:115,1:]
df_azimuthal = df.drop('db_pol', axis =1)

Y_polar = df_polar.iloc[:,0]
X_polar = df_polar.iloc[:, 1:]

runTree(X_polar, Y_polar)


