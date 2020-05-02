import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

import umap
from scipy.sparse.csgraph import connected_components

from preprocessing import preprocessing


def exploratory_data_analysis( args, df_train, df_test ):
    # deep copy
    df_train = df_train.copy()
    df_test = df_test.copy()

    # 目的変数
    target_name = "SalePrice"

    # 前処理済みデータ
    df_train_preprocess, df_test_preprocess = preprocessing( args, df_train, df_test )

    #--------------------------
    # データ構造確認
    #--------------------------
    print( "df_train.head() : \n", df_train.head() )
    print( "df_test.head() : \n", df_test.head() )
    print( "df_train.shape : \n", df_train.shape )
    print( "df_test.shape : \n", df_test.shape )

    for col in df_train.columns:
        print( "df_train[{}].dtypes ] : {}".format(col, df_train[col].dtypes))

    #--------------------------
    # 基本統計量
    #--------------------------
    print( df_train.describe() )
    print( df_test.describe() )

    #--------------------------
    # 目的変数の分布
    #--------------------------
    fig, axis = plt.subplots()
    sns.distplot( df_train[target_name] )
    plt.grid()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "target_dist.png"), dpi = 200, bbox_inches = 'tight' )

    #--------------------------
    # 説明変数の分布
    #--------------------------
    fig, axis = plt.subplots()
    df_train.hist( bins=50, figsize=(80,60) )
    plt.grid()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "train_hist_all_features.png"), dpi = 200, bbox_inches = 'tight' )

    #--------------------------
    # 目的変数と説明変数の関係
    #--------------------------
    # 数値データ
    feats = list( df_train.dtypes[ df_train.dtypes != "object" ].index )
    n_rows, n_cols = 20, 5
    fig, axis = plt.subplots(n_rows, n_cols, figsize=(n_cols*4,n_rows*3) )
    for r in range(0, n_rows):
        for c in range(0, n_cols):  
            i = r * n_cols + c
            if i < len(feats):
                df_data = pd.concat([df_train[target_name], df_train[feats[i]]], axis=1)
                df_data.plot.scatter( x=feats[i], y=target_name, ax = axis[r][c] )
                plt.grid()

    plt.tight_layout()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "target_vs_numuric_features.png"), dpi = 100, bbox_inches = 'tight' )

    # カテゴリーデータ
    feats = list( df_train.dtypes[df_train.dtypes == "object"].index )
    n_rows, n_cols = 30, 2
    fig, axis = plt.subplots(n_rows, n_cols, figsize=(n_cols*6,n_rows*3))
    for r in range(0, n_rows):
        for c in range(0,n_cols):  
            i = r * n_cols+c
            if i < len(feats):
                df_data = pd.concat([df_train[target_name], df_train[feats[i]]], axis=1)
                sns.boxplot( x=feats[i], y=target_name, data=df_data, ax = axis[r][c] )

    plt.tight_layout()    
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "target_vs_categorical_features.png"), dpi = 100, bbox_inches = 'tight' )

    #--------------------------
    # 全特徴量同士の相関をヒートマップで表示
    #--------------------------
    fig, axis = plt.subplots()
    sns.heatmap(df_train.corr(), square=True, vmax=1, vmin=-1, center=0.0 )
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "train_heatmap_all_features.png"), dpi = 300, bbox_inches = 'tight' )

    #--------------------------
    # 目的変数に対する相関をヒートマップで表示
    #--------------------------
    fig, axis = plt.subplots()
    n_heat = 30     # number of variables for heatmap
    cols = df_train.corr().nlargest(n_heat, target_name)[target_name].index
    cm = np.corrcoef(df_train[cols].values.T)
    sns.set( font_scale=0.25 )
    sns.heatmap(cm, cbar=True, annot=True, square=True, vmax=1, vmin=-1, center=0.0, fmt='.2f', annot_kws={'size': 2}, yticklabels=cols.values, xticklabels=cols.values)
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "heatmap_target_vs_features.png"), dpi = 300, bbox_inches = 'tight' )

    #--------------------------
    # PCA
    #--------------------------
    pass

    #--------------------------
    # UMAP（回帰問題では無効）
    #--------------------------
    """
    # データは標準化されている必要あり
    um = umap.UMAP()
    um.fit( df_train_preprocess )
    df_train_umap = um.transform( df_train_preprocess )
    df_test_umap = um.transform( df_test_preprocess )

    fig, axis = plt.subplots()
    plt.scatter( df_train_umap[:,0],df_train_umap[:,1] )
    plt.colorbar()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "umap_train.png"), dpi = 300, bbox_inches = 'tight' )

    fig, axis = plt.subplots()
    plt.scatter( df_test_umap[:,0],df_test_umap[:,1] )
    plt.colorbar()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "umap_test.png"), dpi = 300, bbox_inches = 'tight' )
    """

    return
