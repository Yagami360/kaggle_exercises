import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def preprocessing( args, df_train, df_test ):
    # 全データセット
    df_data = pd.concat([df_train, df_test], sort=False)

    #===========================
    # 特徴量の追加
    #===========================
    df_data['FamilySize'] = df_data['Parch'] + df_data['SibSp'] + 1
    #df_train['FamilySize'] = df_data['FamilySize'][:len(df_train)]
    #df_test['FamilySize'] = df_data['FamilySize'][len(df_train):]
    df_train['FamilySize'] = df_train['Parch'] + df_train['SibSp'] + 1
    df_test['FamilySize'] = df_test['Parch'] + df_test['SibSp'] + 1

    #===========================
    # 無用なデータを除外
    #===========================
    df_train.drop(['Name', 'PassengerId'], axis=1, inplace=True)
    df_test.drop(['Name', 'PassengerId'], axis=1, inplace=True)
    df_train.drop(['SibSp', 'Parch'], axis=1, inplace=True)         # 新たな特徴量 FamilySize に入れ込んだ特徴量を除外
    df_test.drop(['SibSp', 'Parch'], axis=1, inplace=True)          # 
    #df_train.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
    #df_test.drop(['Ticket', 'Cabin'], axis=1, inplace=True)

    #===========================
    # 全特徴量を一括で処理
    #===========================
    for col in df_train.columns:
        if( args.debug ):
            print( "df_train[{}].dtypes ] : {}".format(col, df_train[col].dtypes))

        # 目的変数
        if( col in ["Survived"] ):
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
        # NAN 値の埋め合わせ（平均値）
        if( col in ["Age", 'Fare'] ):
            # データセット全体 df_data での平均値とする
            df_train[col].fillna(np.mean(df_data[col]), inplace=True)
            df_test[col].fillna(np.mean(df_data[col]), inplace=True)
        # NAN 値の埋め合わせ（ゼロ値）/ int 型
        elif( df_train[col].dtypes in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"] ):
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

        #-----------------------------
        # 正規化処理
        #-----------------------------
        if( df_train[col].dtypes in ["float16", "float32", "float64", "float128"] ):
            scaler = StandardScaler()
            scaler.fit( df_train[col].values.reshape(-1,1) )
            df_train[col] = scaler.transform( df_train[col].values.reshape(-1,1) )
            df_test[col] = scaler.transform( df_test[col].values.reshape(-1,1) )

    return df_train, df_test


def preprocessing1( args, df_train, df_test ):
    # 無用なデータを除外
    df_train.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
    df_test.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # データを数量化
    df_train['Sex'].replace(['male','female'], [0, 1], inplace=True)
    df_test['Sex'].replace(['male','female'], [0, 1], inplace=True)

    df_train['Embarked'].fillna(('S'), inplace=True)
    df_train['Embarked'] = df_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    df_test['Embarked'].fillna(('S'), inplace=True)
    df_test['Embarked'] = df_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # NAN 値を補完
    df_train['Fare'].fillna(np.mean(df_train['Fare']), inplace=True)
    df_test['Fare'].fillna(np.mean(df_test['Fare']), inplace=True)

    df_train['Age'].fillna(np.mean(df_train['Age']), inplace=True)
    df_test['Age'].fillna(np.mean(df_test['Age']), inplace=True)
    return df_train, df_test
