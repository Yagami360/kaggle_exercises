import os
import numpy as np
import pandas as pd
import feather
import gc
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures

from utils import read_feature

def agg_dataframe_numric( df_data, agg_column, base_column_name, method = ['count', 'mean', 'max', 'min', 'sum'] ):
    """
    数値型のデータに対して、同じ値を持つ columns を集約したデータフレームを返す
    """
    # Remove id variables other than grouping variable
    for col in df_data:
        if col != agg_column and 'SK_ID' in col:
            df_data = df_data.drop(columns = col)

    df_data_numric = df_data.select_dtypes('number').copy()
    df_data_numric[agg_column] = df_data[agg_column].copy()

    # pd.groupby() で集約
    df_data_numric = df_data_numric.groupby( agg_column, as_index = False ).agg( method ).reset_index()

    # 列名を rename
    new_columns = [agg_column]
    for var in df_data_numric.columns.levels[0]:
        if var != agg_column:            
            for stat in df_data_numric.columns.levels[1][:-1]:
                if( var in base_column_name ):
                    new_columns.append( '%s_%s' % (var, stat))
                else:
                    new_columns.append( base_column_name + '_%s_%s' % (var, stat))

    df_data_numric.columns = new_columns

    # １つの値しか持たない列を除外
    """
    _, idx = np.unique( df_data_numric, axis = 1, return_index=True )
    df_data_numric = df_data_numric.iloc[:, idx]
    """    
    return df_data_numric


def agg_dataframe_categorical( df_data, agg_column, base_column_name, method = ['sum', 'count', 'mean'], one_hot_encode = True ):
    """
    カテゴリ型のデータに対して、同じ値を持つ columns を集約したデータフレームを返す
    """
    df_data_categorical = df_data.select_dtypes('object').copy()
    df_data_categorical[agg_column] = df_data[agg_column].copy()

    if( one_hot_encode ):
        df_data_categorical = pd.get_dummies( df_data_categorical )
    else:
        for col in df_data_categorical.columns:
            # ラベル情報のエンコード
            if( df_data_categorical[col].dtypes == "object" ):
                label_encoder = LabelEncoder()
                label_encoder.fit(list(df_data_categorical[col]))
                df_data_categorical[col] = label_encoder.transform(list(df_data_categorical[col]))

    # pd.groupby() で集約
    df_data_categorical = df_data_categorical.groupby( agg_column, as_index = False ).agg( method ).reset_index()

    # 列名を rename
    new_columns = [agg_column]
    for var in df_data_categorical.columns.levels[0]:
        if var != agg_column:            
            for stat in df_data_categorical.columns.levels[1][:-1]:
                """
                # カテゴリーデータに対しては、sum は count の意味になる
                if( stat == "sum" ):
                    stat = "count"
                # カテゴリーデータに対しては、mean は count_norm の意味になる
                elif( stat == "mean" ):
                    stat = "count_norm"
                """
                if( var in base_column_name ):
                    new_columns.append( '%s_%s' % (var, stat))
                else:
                    new_columns.append( base_column_name + '_%s_%s' % (var, stat))

    df_data_categorical.columns = new_columns

    # １つの値しか持たない列を除外
    """
    _, idx = np.unique( df_data_categorical, axis = 1, return_index=True )
    df_data_categorical = df_data_categorical.iloc[:, idx]
    """

    return df_data_categorical


def preprocessing( args ):
    gc.enable()
    time_bar = tqdm(total = 90, desc = "preprocessing")

    # 目的変数
    target_name = 'TARGET'
    one_hot_encode = args.onehot_encode

    #===========================
    # 無用なデータを除外（結合前）
    #===========================
    # application_{train|test}
    if( args.feature_format ):
        df_application_train = read_feature( os.path.join(args.dataset_dir, "application_train.feature") )
        df_application_test = read_feature( os.path.join(args.dataset_dir, "application_test.feature" ) )
    else:
        df_application_train = pd.read_csv( os.path.join(args.dataset_dir, "application_train.csv" ) )
        df_application_test = pd.read_csv( os.path.join(args.dataset_dir, "application_test.csv" ) )

    #df_application_train.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'], axis=1, inplace=True)
    #df_application_test.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'], axis=1, inplace=True)
    time_bar.update(10)

    #===========================
    # サブ構造の結合
    #===========================
    # 元データ
    df_train = df_application_train
    df_test = df_application_test

    #---------------------------
    # bureau
    #---------------------------
    if( args.feature_format ):
        df_bureau = read_feature( os.path.join(args.dataset_dir, "bureau.feature") )
    else:
        df_bureau = pd.read_csv( os.path.join(args.dataset_dir, "bureau.csv" ) )

    df_bureau_agg_numric = agg_dataframe_numric( df_bureau, agg_column = 'SK_ID_CURR', base_column_name = "bureau" )
    df_bureau_agg_categorical = agg_dataframe_categorical( df_bureau, agg_column = 'SK_ID_CURR', base_column_name = "bureau", one_hot_encode = one_hot_encode )

    # 元のデータに統合
    df_train = pd.merge(df_train, df_bureau_agg_numric, on='SK_ID_CURR', how='left' )
    df_train = pd.merge(df_train, df_bureau_agg_categorical, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_bureau_agg_numric, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_bureau_agg_categorical, on='SK_ID_CURR', how='left' )

    # 不要になったメモリを解放
    del df_bureau_agg_numric, df_bureau_agg_categorical
    gc.collect()
    time_bar.update(10)

    #---------------------------
    # bureau_balance
    #---------------------------
    if( args.feature_format ):
        df_bureau_balance = read_feature( os.path.join(args.dataset_dir, "bureau_balance.feature") )
    else:
        df_bureau_balance = pd.read_csv( os.path.join(args.dataset_dir, "bureau_balance.csv" ) )

    # 同じ SK_ID_BUREAU を集約
    df_bureau_balance_agg_numric = agg_dataframe_numric( df_bureau_balance, agg_column = 'SK_ID_BUREAU', base_column_name = "bureau_balance" )
    df_bureau_balance_agg_categorical = agg_dataframe_categorical( df_bureau_balance, agg_column = 'SK_ID_BUREAU', base_column_name = "bureau_balance", one_hot_encode = one_hot_encode )

    # 親データ （df_bureau） の 'SK_ID_CURR' に、対応する 'SK_ID_BUREAU' を紐付け
    df_bureau_balance_agg_numric = df_bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(df_bureau_balance_agg_numric, on = 'SK_ID_BUREAU', how = 'left')
    df_bureau_balance_agg_categorical = df_bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(df_bureau_balance_agg_categorical, on = 'SK_ID_BUREAU', how = 'left')

    # １つの `SK_ID_CURR` に対して、複数の `SK_ID_BUREAU` が存在することになるので、`SK_ID_CURR` を集約
    df_bureau_balance_agg_numric = agg_dataframe_numric( df_bureau_balance_agg_numric.drop(columns = ['SK_ID_BUREAU']), agg_column = 'SK_ID_CURR', base_column_name = "bureau_balance" )
    df_bureau_balance_agg_categorical = agg_dataframe_numric( df_bureau_balance_agg_categorical.drop(columns = ['SK_ID_BUREAU']), agg_column = 'SK_ID_CURR', base_column_name = "bureau_balance" )

    # 元のデータに統合
    df_train = pd.merge(df_train, df_bureau_balance_agg_numric, on='SK_ID_CURR', how='left' )
    df_train = pd.merge(df_train, df_bureau_balance_agg_categorical, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_bureau_balance_agg_numric, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_bureau_balance_agg_categorical, on='SK_ID_CURR', how='left' )

    # 不要になったメモリを解放
    del df_bureau, df_bureau_balance, df_bureau_balance_agg_numric, df_bureau_balance_agg_categorical
    gc.collect()
    time_bar.update(10)

    #---------------------------
    # previous_application
    #--------------------------
    if( args.feature_format ):
        df_previous_application = read_feature( os.path.join(args.dataset_dir, "previous_application.feature") )
    else:
        df_previous_application = pd.read_csv( os.path.join(args.dataset_dir, "previous_application.csv" ) )

    df_previous_application_agg_numric = agg_dataframe_numric( df_previous_application, agg_column = 'SK_ID_CURR', base_column_name = "previous_application" )
    df_previous_application_agg_categorical = agg_dataframe_categorical( df_previous_application, agg_column = 'SK_ID_CURR', base_column_name = "previous_application", one_hot_encode = one_hot_encode )

    # 元のデータに統合
    df_train = pd.merge(df_train, df_previous_application_agg_numric, on='SK_ID_CURR', how='left' )
    df_train = pd.merge(df_train, df_previous_application_agg_categorical, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_previous_application_agg_numric, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_previous_application_agg_categorical, on='SK_ID_CURR', how='left' )

    # 不要になったメモリを解放
    del df_previous_application_agg_numric, df_previous_application_agg_categorical
    gc.collect()
    time_bar.update(10)

    #---------------------------
    # pos_cash_balance
    #---------------------------
    if( args.feature_format ):
        df_pos_cash_balance = read_feature( os.path.join(args.dataset_dir, "POS_CASH_balance.feature") )
    else:
        df_pos_cash_balance = pd.read_csv( os.path.join(args.dataset_dir, "POS_CASH_balance.csv" ) )

    # 同じ SK_ID_PREV を集約
    df_pos_cash_balance_agg_numric = agg_dataframe_numric( df_pos_cash_balance, agg_column = 'SK_ID_PREV', base_column_name = "pos_cash_balance" )
    df_pos_cash_balance_agg_categorical = agg_dataframe_categorical( df_pos_cash_balance, agg_column = 'SK_ID_PREV', base_column_name = "pos_cash_balance", one_hot_encode = one_hot_encode )

    # 親データ の 'SK_ID_CURR' に、対応する 'SK_ID_PREV' を紐付け
    df_pos_cash_balance_agg_numric = df_previous_application[['SK_ID_PREV', 'SK_ID_CURR']].merge(df_pos_cash_balance_agg_numric, on = 'SK_ID_PREV', how = 'left')
    df_pos_cash_balance_agg_categorical = df_previous_application[['SK_ID_PREV', 'SK_ID_CURR']].merge(df_pos_cash_balance_agg_categorical, on = 'SK_ID_PREV', how = 'left')

    # １つの `SK_ID_CURR` に対して、複数の `SK_ID_BUREAU` が存在することになるので、`SK_ID_CURR` を集約
    df_pos_cash_balance_agg_numric = agg_dataframe_numric( df_pos_cash_balance_agg_numric.drop(columns = ['SK_ID_PREV']), agg_column = 'SK_ID_CURR', base_column_name = "pos_cash_balance" )
    df_pos_cash_balance_agg_categorical = agg_dataframe_numric( df_pos_cash_balance_agg_categorical.drop(columns = ['SK_ID_PREV']), agg_column = 'SK_ID_CURR', base_column_name = "pos_cash_balance" )

    # 元のデータに統合
    df_train = pd.merge(df_train, df_pos_cash_balance_agg_numric, on='SK_ID_CURR', how='left' )
    df_train = pd.merge(df_train, df_pos_cash_balance_agg_categorical, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_pos_cash_balance_agg_numric, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_pos_cash_balance_agg_categorical, on='SK_ID_CURR', how='left' )

    # 不要になったメモリを解放
    del df_pos_cash_balance, df_pos_cash_balance_agg_numric, df_pos_cash_balance_agg_categorical
    gc.collect()
    time_bar.update(10)

    #---------------------------
    # installments_payments
    #---------------------------
    if( args.feature_format ):
        df_installments_payments = read_feature( os.path.join(args.dataset_dir, "installments_payments.feature") )
    else:
        df_installments_payments = pd.read_csv( os.path.join(args.dataset_dir, "installments_payments.csv" ) )

    # カテゴリーデータは存在しない
    # 同じ SK_ID_PREV を集約
    df_installments_payments_agg_numric = agg_dataframe_numric( df_installments_payments, agg_column = 'SK_ID_PREV', base_column_name = "installments_payments" )

    # 親データ の 'SK_ID_CURR' に、対応する 'SK_ID_PREV' を紐付け
    df_installments_payments_agg_numric = df_previous_application[['SK_ID_PREV', 'SK_ID_CURR']].merge(df_installments_payments_agg_numric, on = 'SK_ID_PREV', how = 'left')

    # １つの `SK_ID_CURR` に対して、複数の `SK_ID_BUREAU` が存在することになるので、`SK_ID_CURR` を集約
    df_installments_payments_agg_numric = agg_dataframe_numric( df_installments_payments_agg_numric.drop(columns = ['SK_ID_PREV']), agg_column = 'SK_ID_CURR', base_column_name = "installments_payments" )

    # 元のデータに統合
    df_train = pd.merge(df_train, df_installments_payments_agg_numric, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_installments_payments_agg_numric, on='SK_ID_CURR', how='left' )

    # 不要になったメモリを解放
    del df_installments_payments, df_installments_payments_agg_numric
    gc.collect()
    time_bar.update(10)

    #---------------------------
    # credit_card_balance
    #---------------------------
    if( args.feature_format ):
        df_credit_card_balance = read_feature( os.path.join(args.dataset_dir, "credit_card_balance.feature") )
    else:
        df_credit_card_balance = pd.read_csv( os.path.join(args.dataset_dir, "credit_card_balance.csv" ) )

    # 同じ SK_ID_PREV を集約
    df_credit_card_balance_agg_numric = agg_dataframe_numric( df_credit_card_balance, agg_column = 'SK_ID_PREV', base_column_name = "credit_card_balance" )
    df_credit_card_balance_agg_categorical = agg_dataframe_categorical( df_credit_card_balance, agg_column = 'SK_ID_PREV', base_column_name = "credit_card_balance", one_hot_encode = one_hot_encode )

    # 親データ の 'SK_ID_CURR' に、対応する 'SK_ID_PREV' を紐付け
    df_credit_card_balance_agg_numric = df_previous_application[['SK_ID_PREV', 'SK_ID_CURR']].merge(df_credit_card_balance_agg_numric, on = 'SK_ID_PREV', how = 'left')
    df_credit_card_balance_agg_categorical = df_previous_application[['SK_ID_PREV', 'SK_ID_CURR']].merge(df_credit_card_balance_agg_categorical, on = 'SK_ID_PREV', how = 'left')

    # １つの `SK_ID_CURR` に対して、複数の `SK_ID_BUREAU` が存在することになるので、`SK_ID_CURR` を集約
    df_credit_card_balance_agg_numric = agg_dataframe_numric( df_credit_card_balance_agg_numric.drop(columns = ['SK_ID_PREV']), agg_column = 'SK_ID_CURR', base_column_name = "credit_card_balance" )
    df_credit_card_balance_agg_categorical = agg_dataframe_numric( df_credit_card_balance_agg_categorical.drop(columns = ['SK_ID_PREV']), agg_column = 'SK_ID_CURR', base_column_name = "credit_card_balance" )

    # 元のデータに統合
    df_train = pd.merge(df_train, df_credit_card_balance_agg_numric, on='SK_ID_CURR', how='left' )
    df_train = pd.merge(df_train, df_credit_card_balance_agg_categorical, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_credit_card_balance_agg_numric, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_credit_card_balance_agg_categorical, on='SK_ID_CURR', how='left' )

    # 不要になったメモリを解放
    del df_credit_card_balance, df_credit_card_balance_agg_numric, df_credit_card_balance_agg_categorical
    gc.collect()
    time_bar.update(10)

    #===========================
    # 特徴量の追加（結合後）
    #===========================
    # 異常値を含む特徴量
    if( args.invalid_features ):
        df_train['DAYS_EMPLOYED_ANOM'] = df_train["DAYS_EMPLOYED"] == 365243    # 異常値のフラグ
        df_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
        df_test['DAYS_EMPLOYED_ANOM'] = df_test["DAYS_EMPLOYED"] == 365243      # 異常値のフラグ
        df_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

    # 時系列データ
    if( args.time_features ):
        df_train['DAYS_BIRTH'] = -1 * df_train['DAYS_BIRTH']
        df_test['DAYS_BIRTH'] = -1 * df_test['DAYS_BIRTH']
        df_train['YEARS_BIRTH'] = df_train['DAYS_BIRTH'] / 365
        df_test['YEARS_BIRTH'] = df_test['DAYS_BIRTH'] / 365
        #df_train['YEARS_BINNED'] = pd.cut(df_train['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
        #df_test['YEARS_BINNED'] = pd.cut(df_test['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

    #----------------------------
    # 目的変数と強い相関をもつ特徴量での多項式特徴量（PolynomialFeatures）
    #----------------------------
    if( args.polynomial_features ):
        df_train_poly_features = df_train[ ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET'] ]
        df_test_poly_features = df_test[ ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'] ]
        df_train_poly_features_target = df_train_poly_features[target_name]
        df_train_poly_features = df_train_poly_features.drop(columns = [target_name])

        # Need to impute missing values
        imputer = SimpleImputer(strategy = 'median')
        df_train_poly_features = imputer.fit_transform(df_train_poly_features)
        df_test_poly_features = imputer.transform(df_test_poly_features)

        # Train the polynomial features and Transform the features
        poly_transformer = PolynomialFeatures(degree = 3)
        poly_transformer.fit(df_train_poly_features)
        df_train_poly_features = poly_transformer.transform(df_train_poly_features)
        df_test_poly_features = poly_transformer.transform(df_test_poly_features)

        # Create a dataframe of the features 
        df_train_poly_features = pd.DataFrame(
            df_train_poly_features, 
            columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])
        )
        df_train_poly_features[target_name] = df_train_poly_features_target
        print( df_train_poly_features.head() )

        # Put test features into dataframe
        df_test_poly_features = pd.DataFrame(
            df_test_poly_features, 
            columns = poly_transformer.get_feature_names(['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH'])
        )

        # Merge polynomial features into training dataframe
        df_train_poly_features['SK_ID_CURR'] = df_train['SK_ID_CURR']
        df_train = pd.merge( df_train, df_train_poly_features, on = 'SK_ID_CURR', how = 'left')

        # Merge polnomial features into testing dataframe
        df_test_poly_features['SK_ID_CURR'] = df_test['SK_ID_CURR']
        df_test = pd.merge( df_test, df_test_poly_features, on = 'SK_ID_CURR', how = 'left')

        # Align the dataframes
        df_train, df_test = df_train.align(df_test, join = 'inner', axis = 1)

    #----------------------------
    # ドメイン知識に基づく特徴量
    #----------------------------
    if( args.domain_features ):
        # CREDIT_INCOME_PERCENT: クライアントの収入に対する信用額の割合。
        df_train['CREDIT_INCOME_PERCENT'] = df_train['AMT_CREDIT'] / df_train['AMT_INCOME_TOTAL']
        df_test['CREDIT_INCOME_PERCENT'] = df_test['AMT_CREDIT'] / df_test['AMT_INCOME_TOTAL']

        # ANNUITY_INCOME_PERCENT: クライアントの収入に対するローン年金の割合。
        df_train['ANNUITY_INCOME_PERCENT'] = df_train['AMT_ANNUITY'] / df_train['AMT_INCOME_TOTAL']
        df_test['ANNUITY_INCOME_PERCENT'] = df_test['AMT_ANNUITY'] / df_test['AMT_INCOME_TOTAL']

        # CREDIT_TERM: お支払い期間を月単位で指定します。
        df_train['CREDIT_TERM'] = df_train['AMT_ANNUITY'] / df_train['AMT_CREDIT']
        df_test['CREDIT_TERM'] = df_test['AMT_ANNUITY'] / df_test['AMT_CREDIT']

        # DAYS_EMPLOYED_PERCENT: クライアントの年齢に対する在職日数の割合。
        df_train['DAYS_EMPLOYED_PERCENT'] = df_train['DAYS_EMPLOYED'] / df_train['DAYS_BIRTH']
        df_test['DAYS_EMPLOYED_PERCENT'] = df_test['DAYS_EMPLOYED'] / df_test['DAYS_BIRTH']

    time_bar.update(10)

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
            label_encoder.fit(list(df_data[col]))
            df_train[col] = label_encoder.transform(list(df_train[col]))

            label_encoder = LabelEncoder()
            label_encoder.fit(list(df_data[col]))
            df_test[col] = label_encoder.transform(list(df_test[col]))

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

        #-----------------------------
        # 値が単一の特徴量をクレンジング
        #-----------------------------
        if( df_train[col].nunique() == 1 ):
            print( "remove {} : {}".format(col,df_train[col].nunique()) )
            df_train.drop([col], axis=1, inplace=True)
            df_test.drop([col], axis=1, inplace=True)

    time_bar.update(10)
    gc.disable()
    return df_train, df_test

