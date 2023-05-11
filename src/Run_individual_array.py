import pandas as pd
import matplotlib as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from tree_search import runmodel
from run_final_model import runTree

from sklearn.tree import export_graphviz


path = 'Data\preliminary_data_2.xlsx'
df = pd.read_excel(path)

df_polar = df.iloc[:276, 1:]  # 115 we are omitting contra-rotation data, 276 for preldata2
df_azimuthal = df.drop('db_pol', axis=1)
df_azimuthal = df_azimuthal.iloc[:276, 0:5]

Y_polar = df_polar.iloc[:, 0]
X_polar = df_polar.iloc[:, 1:3]  # Three since we are omitting thrust

Y_azi = df_azimuthal.iloc[:, 0]
X_azi = df_azimuthal.iloc[:, 1:6]

runTree(X_polar, Y_polar)
#runmodel(X_azi, Y_azi)



"""
Try GD like Neider Mead
Minimizing the cost function
Other methods like Bayesian optimization
"""