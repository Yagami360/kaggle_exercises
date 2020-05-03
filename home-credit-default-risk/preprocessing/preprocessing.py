import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


def rename_columns_levels( df_data, base_name, base_columns_name ):
    # List of column names
    columns = [base_columns_name]

    # Iterate through the variables names
    for var in df_data.columns.levels[0]:
        # Skip the id name
        if var != base_columns_name:            
            # Iterate through the stat names
            for stat in df_data.columns.levels[1][:-1]:
                # Make a new column name for the variable and stat
                columns.append( base_name + '_%s_%s' % (var, stat))

    #print( df_data.columns )
    #print( columns )
    return columns

def preprocessing( 
        args, 
        df_application_train, df_application_test, 
        df_bureau, df_bureau_balance, 
        df_previous_application, df_pos_cash_balance, df_installments_payments, df_credit_card_balance,
):
    # 目的変数
    target_name = 'TARGET'

    #===========================
    # 無用なデータを除外（結合前）
    #===========================
    # application_{train|test}
    df_application_train.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'], axis=1, inplace=True)
    df_application_test.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'], axis=1, inplace=True)

    #===========================
    # サブ構造の結合
    #===========================
    onehot_encode = False

    #---------------------------
    # bureau
    #---------------------------
    if( onehot_encode ):
        df_bureau_balance_categorical = pd.get_dummies( df_bureau_balance.select_dtypes('object') )
        df_bureau_balance_categorical['SK_ID_BUREAU'] = df_bureau_balance['SK_ID_BUREAU']
        df_bureau_balance_categorical = df_bureau_balance_categorical.groupby('SK_ID_BUREAU', as_index = False).agg(['mean']).reset_index()
        df_bureau_balance_categorical.columns = rename_columns_levels( df_bureau_balance_categorical, "bureau_balance", 'SK_ID_BUREAU' )
        df_bureau_balance_categorical = pd.merge(df_bureau_balance, df_bureau_balance_categorical, on='SK_ID_BUREAU', how='left' )
    else:
        # bureau_balance
        for col in df_bureau_balance.columns:
            # ラベル情報のエンコード
            if( df_bureau_balance[col].dtypes == "object" ):
                label_encoder = LabelEncoder()
                label_encoder.fit(list(df_bureau_balance[col]))
                df_bureau_balance[col] = label_encoder.transform(list(df_bureau_balance[col]))

    # 同じ SK_ID_BUREAU を集約
    df_bureau_balance_numric = df_bureau_balance.groupby('SK_ID_BUREAU', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_bureau_balance_numric.columns = rename_columns_levels( df_bureau_balance_numric, "bureau_balance", 'SK_ID_BUREAU' )
    df_bureau_balance_numric = pd.merge(df_bureau_balance, df_bureau_balance_numric, on='SK_ID_BUREAU', how='left' )

    # bureau
    if( onehot_encode ):
        df_bureau_categorical = pd.get_dummies( df_bureau.select_dtypes('object') )
        df_bureau_categorical['SK_ID_CURR'] = df_bureau['SK_ID_CURR']
        df_bureau_categorical = df_bureau_categorical.groupby('SK_ID_CURR', as_index = False).agg(['mean']).reset_index()
        df_bureau_categorical.columns = rename_columns_levels( df_bureau_categorical, "bureau", 'SK_ID_CURR' )
        df_bureau_categorical = pd.merge(df_bureau, df_bureau_categorical, on='SK_ID_CURR', how='left' )
    else:
        for col in df_bureau.columns:
            # ラベル情報のエンコード
            if( df_bureau[col].dtypes == "object" ):
                label_encoder = LabelEncoder()
                label_encoder.fit(list(df_bureau[col]))
                df_bureau[col] = label_encoder.transform(list(df_bureau[col]))

    # 同じ SK_ID_CURR の行を 過去の申込み回数（SK_ID_CURR あたりの SK_ID_BUREAU の個数）,　各々の特徴量の mean, max, min, で集約する。 
    df_bureau_numric = df_bureau.groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_bureau_numric.columns = rename_columns_levels( df_bureau_numric, "bureau", 'SK_ID_CURR' )
    df_bureau = pd.merge(df_bureau, df_bureau_numric, on='SK_ID_CURR', how='left' )

    # サブ構造を結合
    df_bureau = pd.merge(df_bureau, df_bureau_balance_numric, on='SK_ID_BUREAU', how='left' )

    print( df_bureau.shape )    # (25121815, 91)
    print( df_bureau.head() )

    #---------------------------
    # previous_application
    #---------------------------
    # pos_cash_balance
    if( onehot_encode ):
        df_pos_cash_balance_categorical = pd.get_dummies( df_pos_cash_balance.select_dtypes('object') )
        df_pos_cash_balance_categorical['SK_ID_PREV'] = df_pos_cash_balance['SK_ID_PREV']
        df_pos_cash_balance_categorical = df_pos_cash_balance_categorical.groupby('SK_ID_PREV', as_index = False).agg(['mean']).reset_index()
        df_pos_cash_balance_categorical.columns = rename_columns_levels( df_pos_cash_balance_categorical, "pos_cash_balance", 'SK_ID_PREV' )
        df_pos_cash_balance_categorical = pd.merge(df_previous_application, df_pos_cash_balance_categorical, on='SK_ID_PREV', how='left' )
    else:
        for col in df_pos_cash_balance.columns:
            # ラベル情報のエンコード
            if( df_pos_cash_balance[col].dtypes == "object" ):
                label_encoder = LabelEncoder()
                label_encoder.fit(list(df_pos_cash_balance[col]))
                df_pos_cash_balance[col] = label_encoder.transform(list(df_pos_cash_balance[col]))

    df_pos_cash_balance_numric = df_pos_cash_balance.groupby('SK_ID_PREV', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_pos_cash_balance_numric.columns = rename_columns_levels( df_pos_cash_balance_numric, "pos_cash_balance", 'SK_ID_PREV' )
    df_pos_cash_balance_numric = pd.merge(df_previous_application, df_pos_cash_balance_numric, on='SK_ID_PREV', how='left' )

    # installments_payments
    if( onehot_encode ):
        df_installments_payments_categorical = pd.get_dummies( df_installments_payments.select_dtypes('object') )
        df_installments_payments_categorical['SK_ID_PREV'] = df_installments_payments['SK_ID_PREV']
        df_installments_payments_categorical = df_installments_payments_categorical.groupby('SK_ID_PREV', as_index = False).agg(['mean']).reset_index()
        df_installments_payments_categorical.columns = rename_columns_levels( df_installments_payments_categorical, "installments_payments", 'SK_ID_PREV' )
        df_installments_payments_categorical = pd.merge(df_previous_application, df_installments_payments_categorical, on='SK_ID_PREV', how='left' )
    else:
        for col in df_installments_payments.columns:
            # ラベル情報のエンコード
            if( df_installments_payments[col].dtypes == "object" ):
                label_encoder = LabelEncoder()
                label_encoder.fit(list(df_installments_payments[col]))
                df_installments_payments[col] = label_encoder.transform(list(df_installments_payments[col]))

    df_installments_payments_numric = df_installments_payments.groupby('SK_ID_PREV', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_installments_payments_numric.columns = rename_columns_levels( df_installments_payments_numric, "installments_payments", 'SK_ID_PREV' )
    df_installments_payments_numric = pd.merge(df_previous_application, df_installments_payments_numric, on='SK_ID_PREV', how='left' )

    # credit_card_balance
    if( onehot_encode ):
        df_credit_card_balance_categorical = pd.get_dummies( df_credit_card_balance.select_dtypes('object') )
        df_credit_card_balance_categorical['SK_ID_PREV'] = df_credit_card_balance['SK_ID_PREV']
        df_credit_card_balance_categorical = df_credit_card_balance_categorical.groupby('SK_ID_PREV', as_index = False).agg(['mean']).reset_index()
        df_credit_card_balance_categorical.columns = rename_columns_levels( df_credit_card_balance_categorical, "credit_card_balance", 'SK_ID_PREV' )
        df_credit_card_balance_categorical = pd.merge(df_previous_application, df_credit_card_balance_categorical, on='SK_ID_PREV', how='left' )
    else:
        for col in df_credit_card_balance.columns:
            # ラベル情報のエンコード
            if( df_credit_card_balance[col].dtypes == "object" ):
                label_encoder = LabelEncoder()
                label_encoder.fit(list(df_credit_card_balance[col]))
                df_credit_card_balance[col] = label_encoder.transform(list(df_credit_card_balance[col]))

    df_credit_card_balance_numric = df_credit_card_balance.groupby('SK_ID_PREV', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_credit_card_balance_numric.columns = rename_columns_levels( df_credit_card_balance_numric, "credit_card_balance", 'SK_ID_PREV' )
    df_credit_card_balance_numric = pd.merge(df_previous_application, df_credit_card_balance_numric, on='SK_ID_PREV', how='left' )

    # previous_application
    if( onehot_encode ):
        df_previous_application_categorical = pd.get_dummies( df_previous_application.select_dtypes('object') )
        df_previous_application_categorical['SK_ID_CURR'] = df_previous_application['SK_ID_CURR']
        df_previous_application_categorical = df_previous_application_categorical.groupby('SK_ID_CURR', as_index = False).agg(['mean']).reset_index()
        df_previous_application_categorical.columns = rename_columns_levels( df_previous_application_categorical, "revious_application", 'SK_ID_CURR' )
        df_previous_application_categorical = pd.merge(df_previous_application, df_previous_application_categorical, on='SK_ID_CURR', how='left' )
    else:
        for col in df_previous_application.columns:
            # ラベル情報のエンコード
            if( df_previous_application[col].dtypes == "object" ):
                label_encoder = LabelEncoder()
                label_encoder.fit(list(df_previous_application[col]))
                df_previous_application[col] = label_encoder.transform(list(df_previous_application[col]))

    df_previous_application_numric = df_previous_application.groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_previous_application_numric.columns = rename_columns_levels( df_previous_application_numric, "revious_application", 'SK_ID_CURR' )
    df_previous_application = pd.merge(df_previous_application, df_previous_application_numric, on='SK_ID_CURR', how='left' )

    # サブ構造を結合
    df_previous_application = pd.merge(df_previous_application, df_pos_cash_balance_numric, on='SK_ID_CURR', how='left' )
    df_previous_application = pd.merge(df_previous_application, df_installments_payments_numric, on='SK_ID_CURR', how='left' )
    df_previous_application = pd.merge(df_previous_application, df_credit_card_balance_numric, on='SK_ID_CURR', how='left' )

    print( df_previous_application.shape )
    print( df_previous_application.head() )

    #===========================
    # 学習用テスト用データに結合
    #===========================
    # application_{train|test}
    df_train = df_application_train
    df_test = df_application_test

    # bureau とそのサブ構造
    df_train = pd.merge(df_train, df_bureau, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_bureau, on='SK_ID_CURR', how='left' )

    # previous_application とそのサブ構造
    df_train = pd.merge(df_train, df_previous_application, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_previous_application, on='SK_ID_CURR', how='left' )

    print( df_train.shape )
    print( df_train.head() )

    #===========================
    # 特徴量の追加（結合後）
    #===========================
    df_train['DAYS_EMPLOYED_ANOM'] = df_train["DAYS_EMPLOYED"] == 365243    # 異常値のフラグ
    df_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    df_test['DAYS_EMPLOYED_ANOM'] = df_test["DAYS_EMPLOYED"] == 365243      # 異常値のフラグ
    df_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

    #===========================
    # 無用なデータを除外（結合後）
    #===========================
    if 'SK_ID_CURR' in df_train.columns:
        df_train.drop(['SK_ID_CURR'], axis=1, inplace=True)
        df_test.drop(['SK_ID_CURR'], axis=1, inplace=True)
    if 'SK_ID_BUREAU' in df_train.columns:
        df_train.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
        df_test.drop(['SK_ID_BUREAU'], axis=1, inplace=True)
    if 'SK_ID_PREV' in df_train.columns:
        df_train.drop(['SK_ID_PREV'], axis=1, inplace=True)
        df_test.drop(['SK_ID_PREV'], axis=1, inplace=True)
    
    #===========================
    # 全特徴量を一括で処理
    #===========================
    # 全データセット
    df_data = pd.concat([df_train, df_test], sort=False)

    for col in df_train.columns:
        # 目的変数
        if( col in [target_name] ):
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
        # 変換値の再変換・異常値のクレンジング
        #-----------------------------
        if( col in ["DAYS_BIRTH"] ):
            df_train[col] = df_train[col] / 365
            df_test[col] = df_test[col] / 365

        #-----------------------------
        # 欠損値の埋め合わせ
        #-----------------------------
        # NAN 値の埋め合わせ（平均値）
        if( col in ["OWN_CAR_AGE"] ):
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
        """
        if( df_train[col].dtypes in ["float16", "float32", "float64", "float128"] ):
            scaler = StandardScaler()
            scaler.fit( df_train[col].values.reshape(-1,1) )
            df_train[col] = scaler.transform( df_train[col].values.reshape(-1,1) )
            df_test[col] = scaler.transform( df_test[col].values.reshape(-1,1) )
        """

    return df_train, df_test

