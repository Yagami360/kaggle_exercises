import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def preprocessing( args, df_train, df_test ):
    # 全データセット
    df_data = pd.concat([df_train, df_test], sort=False)

    # 時系列データを処理
    df_train['year'] = [t.year for t in pd.DatetimeIndex(df_train.datetime)]
    df_train['year'] = df_train['year'].map( {2011:0, 2012:1} )
    df_train["month"] = [t.month for t in pd.DatetimeIndex(df_train.datetime)]
    df_train["day"] = [t.dayofweek for t in pd.DatetimeIndex(df_train.datetime)]
    df_train["hour"] = [t.hour for t in pd.DatetimeIndex(df_train.datetime)]
    df_train["weekday"] = [t for t in pd.DatetimeIndex(df_train.datetime).weekday]

    df_test['year'] = [t.year for t in pd.DatetimeIndex(df_test.datetime)]
    df_test['year'] = df_test['year'].map( {2011:0, 2012:1} )
    df_test["month"] = [t.month for t in pd.DatetimeIndex(df_test.datetime)]
    df_test["day"] = [t.dayofweek for t in pd.DatetimeIndex(df_test.datetime)]
    df_test["hour"] = [t.hour for t in pd.DatetimeIndex(df_test.datetime)]
    df_test["weekday"] = [t for t in pd.DatetimeIndex(df_test.datetime).weekday]

    # 無用なデータを除外
    df_train.drop(["casual", "registered"], axis=1, inplace=True)
    df_train.drop(["datetime"], axis=1, inplace=True)
    df_test.drop(["datetime"], axis=1, inplace=True)

    # 全特徴量を一括で処理
    for col in df_train.columns:
        if( args.debug ):
            print( "df_train[{}].dtypes ] : {}".format(col, df_train[col].dtypes))

        # 目的変数
        if( col in ["count"] ):
            if( args.target_norm ):
                # 正規分布に従うように対数化
                df_train[col] = pd.Series( np.log(df_train[col].values), name=col )

            continue

        #-----------------------------
        # ラベル情報のエンコード
        #-----------------------------
        if( df_train[col].dtypes == "object" ):
            label_encoder = LabelEncoder()
            label_encoder.fit(list(df_train[col]))
            df_train[col] = label_encoder.transform(list(df_train[col]))

            label_encoder = LabelEncoder()
            label_encoder.fit(list(df_test[col]))
            df_test[col] = label_encoder.transform(list(df_test[col]))

        #-----------------------------
        # 欠損値の埋め合わせ
        #-----------------------------
        """
        # NAN 値の埋め合わせ（平均値）
        pass

        # NAN 値の埋め合わせ（ゼロ値）/ int 型
        if( df_train[col].dtypes in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"] ):
            df_train[col].fillna(0, inplace=True)
            df_test[col].fillna(0, inplace=True)
        # NAN 値の埋め合わせ（ゼロ値）/ float 型
        elif( df_train[col].dtypes in ["float16", "float32", "float64", "float128"] ):
            df_train[col].fillna(0.0, inplace=True)
            df_test[col].fillna(0.0, inplace=True)
        # NAN 値の補完（None値）/ object 型
        else:
            df_train[col] = df_train[col].fillna('NA')
            df_test[col] = df_test[col].fillna('NA')
        """

        #-----------------------------
        # 正規化処理
        #-----------------------------
        if( args.input_norm ):
            #if( df_train[col].dtypes != "object" ):
            if( df_train[col].dtypes in ["float16", "float32", "float64", "float128"] ):
                scaler = StandardScaler()
                scaler.fit( df_train[col].values.reshape(-1,1) )
                df_train[col] = scaler.fit_transform( df_train[col].values.reshape(-1,1) )
                df_test[col] = scaler.fit_transform( df_test[col].values.reshape(-1,1) )

        #-----------------------------
        # 値が単一の特徴量をクレンジング
        #-----------------------------
        if( df_train[col].nunique() == 1 ):
            print( "remove {} : {}".format(col,df_train[col].nunique()) )
            df_train.drop([col], axis=1, inplace=True)
            df_test.drop([col], axis=1, inplace=True)
            
    return df_train, df_test
