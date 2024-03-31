# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 16:59:11 2023

@author: truma
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

## code goes here
def load_data(filepath):
    "Loads in data from filepath as Dataframe"
    return pd.read_csv(filepath, index_col = 0, header = 'infer')

def reduce_dimensions(signal_df):
    """Do Principal Component Analysis on input
    and return first 2 PCs in DataFrame"""

    # init std scaler
    # standardize features by recentering at zero and scaling to unit variance
    signals_standardized = StandardScaler().fit_transform(signal_df)

    # Now apply PCA to selected features
    num_components = 2
    signals_PCA = PCA(n_components = num_components).fit_transform(signals_standardized)
    print(signals_PCA.shape)

    # extract index of original dataframe for PCA
    index = signal_df.index
    # init PCA dataframe
    PCA_df = pd.DataFrame(signals_PCA[:, :2], columns = ['PC_001', 'PC_002'], index = index)
    return PCA_df
    pass

def plot_pca(pca_df):
    f, axs = plt.subplots()
    sns.scatterplot(pca_df)
    return axs


signals = load_data('Signal.csv')
print(signals.head())
PCA_df = reduce_dimensions(signals)
print(PCA_df)