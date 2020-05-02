import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import umap


def exploratory_data_analysis( args, X_train, y_train, X_test ):
    # deep copy
    X_train = X_train.copy()
    y_train = y_train.copy()
    X_test = X_test.copy()

    #--------------------------
    # データ構造確認
    #--------------------------
    print( "X_train.shape : \n", X_train.shape )
    print( "y_train.shape : \n", y_train.shape )
    print( "X_test.shape : \n", X_test.shape )

    #--------------------------
    # 基本統計量
    #--------------------------
    pass

    #--------------------------
    # PCA
    #--------------------------
    pass

    #--------------------------
    # UMAP
    #--------------------------
    # 2次元データにする
    X_train = X_train.reshape( X_train.shape[0], -1 )

    # データは標準化されている必要あり
    um = umap.UMAP()
    X_train_umap = um.fit_transform( X_train )

    fig, axis = plt.subplots()
    for i in range(args.n_classes):
        mask = ( y_train == i )
        plt.scatter(X_train_umap[mask, 0], X_train_umap[mask, 1], label=str(i), s=2, alpha=0.5)

    axis.legend( bbox_to_anchor=(1.00, 1), loc='upper left' )
    plt.grid()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "umap.png"), dpi = 300, bbox_inches = 'tight' )

    return
