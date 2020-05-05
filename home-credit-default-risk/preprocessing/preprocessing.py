import os
import numpy as np
import pandas as pd
import gc

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures

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


def rename_columns_levels( df_data, base_name, base_columns_name ):
    new_columns = [base_columns_name]
    for var in df_data.columns.levels[0]:
        if var != base_columns_name:            
            for stat in df_data.columns.levels[1][:-1]:
                if( var in base_columns_name ):
                    new_columns.append( '%s_%s' % (var, stat))
                else:
                    new_columns.append( base_name + '_%s_%s' % (var, stat))

    return new_columns


def preprocessing( args ):
    gc.enable()

    # 目的変数
    target_name = 'TARGET'
    one_hot_encode = args.onehot_encode

    #===========================
    # 無用なデータを除外（結合前）
    #===========================
    # application_{train|test}
    df_application_train = pd.read_csv( os.path.join(args.dataset_dir, "application_train.csv" ) )
    df_application_test = pd.read_csv( os.path.join(args.dataset_dir, "application_test.csv" ) )
    #df_application_train.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'], axis=1, inplace=True)
    #df_application_test.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'], axis=1, inplace=True)

    #===========================
    # サブ構造の結合
    #===========================
    # 元データ
    df_train = df_application_train
    df_test = df_application_test

    #---------------------------
    # bureau
    #---------------------------
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

    #---------------------------
    # bureau_balance
    #---------------------------
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

    #---------------------------
    # previous_application
    #--------------------------
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

    #---------------------------
    # pos_cash_balance
    #---------------------------
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

    #---------------------------
    # installments_payments
    #---------------------------
    df_installments_payments = pd.read_csv( os.path.join(args.dataset_dir, "installments_payments.csv" ) )

    # 同じ SK_ID_PREV を集約
    df_installments_payments_agg_numric = agg_dataframe_numric( df_installments_payments, agg_column = 'SK_ID_PREV', base_column_name = "installments_payments" )
    df_installments_payments_agg_categorical = agg_dataframe_categorical( df_installments_payments, agg_column = 'SK_ID_PREV', base_column_name = "installments_payments", one_hot_encode = one_hot_encode )

    # 親データ の 'SK_ID_CURR' に、対応する 'SK_ID_PREV' を紐付け
    df_installments_payments_agg_numric = df_previous_application[['SK_ID_PREV', 'SK_ID_CURR']].merge(df_installments_payments_agg_numric, on = 'SK_ID_PREV', how = 'left')
    df_installments_payments_agg_categorical = df_previous_application[['SK_ID_PREV', 'SK_ID_CURR']].merge(df_installments_payments_agg_categorical, on = 'SK_ID_PREV', how = 'left')

    # １つの `SK_ID_CURR` に対して、複数の `SK_ID_BUREAU` が存在することになるので、`SK_ID_CURR` を集約
    df_installments_payments_agg_numric = agg_dataframe_numric( df_installments_payments_agg_numric.drop(columns = ['SK_ID_PREV']), agg_column = 'SK_ID_CURR', base_column_name = "installments_payments" )
    df_installments_payments_agg_categorical = agg_dataframe_numric( df_installments_payments_agg_categorical.drop(columns = ['SK_ID_PREV']), agg_column = 'SK_ID_CURR', base_column_name = "installments_payments" )
    print( df_installments_payments_agg_categorical.head() )

    # 元のデータに統合
    df_train = pd.merge(df_train, df_installments_payments_agg_numric, on='SK_ID_CURR', how='left' )
    df_train = pd.merge(df_train, df_installments_payments_agg_categorical, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_installments_payments_agg_numric, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_installments_payments_agg_categorical, on='SK_ID_CURR', how='left' )

    # 不要になったメモリを解放
    del df_installments_payments, df_installments_payments_agg_numric, df_installments_payments_agg_categorical
    gc.collect()

    #---------------------------
    # credit_card_balance
    #---------------------------
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

    #===========================
    # 特徴量の追加（結合後）
    #===========================
    df_train['DAYS_EMPLOYED_ANOM'] = df_train["DAYS_EMPLOYED"] == 365243    # 異常値のフラグ
    df_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    df_test['DAYS_EMPLOYED_ANOM'] = df_test["DAYS_EMPLOYED"] == 365243      # 異常値のフラグ
    df_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

    # 時系列データ
    df_train['DAYS_BIRTH'] = -1 * df_train['DAYS_BIRTH']
    df_test['DAYS_BIRTH'] = -1 * df_test['DAYS_BIRTH']
    df_train['YEARS_BIRTH'] = df_train['DAYS_BIRTH'] / 365
    df_test['YEARS_BIRTH'] = df_test['DAYS_BIRTH'] / 365
    #df_train['YEARS_BINNED'] = pd.cut(df_train['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
    #df_test['YEARS_BINNED'] = pd.cut(df_test['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

    #----------------------------
    # 目的変数と強い相関をもつ特徴量での多項式特徴量（PolynomialFeatures）
    #----------------------------
    df_train_poly_features = df_train[ ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET'] ]
    df_test_poly_features = df_test[ ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH', 'TARGET'] ]
    df_train_poly_features_target = df_train_poly_features[target_name]
    df_train_poly_features = df_train_poly_features.drop(columns = [target_name])

    # Need to impute missing values
    imputer = Imputer(strategy = 'median')
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
    df_train_poly_features['TARGET'] = df_train_poly_features_target

    # To Do : add train, test
    pass

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

    gc.disable()
    return df_train, df_test



def preprocessing2( args ):
    gc.enable()

    # 目的変数
    target_name = 'TARGET'

    #===========================
    # 無用なデータを除外（結合前）
    #===========================
    # application_{train|test}
    df_application_train = pd.read_csv( os.path.join(args.dataset_dir, "application_train.csv" ) )
    df_application_test = pd.read_csv( os.path.join(args.dataset_dir, "application_test.csv" ) )
    #df_application_train.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'], axis=1, inplace=True)
    #df_application_test.drop(['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'], axis=1, inplace=True)

    #===========================
    # サブ構造の結合
    #===========================
    # 元データ
    df_train = df_application_train
    df_test = df_application_test

    #---------------------------
    # bureau
    #---------------------------
    df_bureau = pd.read_csv( os.path.join(args.dataset_dir, "bureau.csv" ) )
    for col in df_bureau.columns:
        # ラベル情報のエンコード
        if( df_bureau[col].dtypes == "object" ):
            label_encoder = LabelEncoder()
            label_encoder.fit(list(df_bureau[col]))
            df_bureau[col] = label_encoder.transform(list(df_bureau[col]))

    # 同じ SK_ID_CURR の行を 過去の申込み回数（SK_ID_CURR あたりの SK_ID_BUREAU の個数）,　各々の特徴量の mean, max, min, で集約する。 
    df_bureau_agg = df_bureau.groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_bureau_agg.columns = rename_columns_levels( df_bureau_agg, "bureau", 'SK_ID_CURR' )
    #print( df_bureau_agg.shape )
    #print( df_bureau_agg.head() )

    # 元のデータに統合
    df_train = pd.merge(df_train, df_bureau_agg, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_bureau_agg, on='SK_ID_CURR', how='left' )

    # 不要になったメモリを解放
    del df_bureau_agg
    gc.collect()

    #---------------------------
    # bureau_balance
    #---------------------------
    df_bureau_balance = pd.read_csv( os.path.join(args.dataset_dir, "bureau_balance.csv" ) )
    for col in df_bureau_balance.columns:
        # ラベル情報のエンコード
        if( df_bureau_balance[col].dtypes == "object" ):
            label_encoder = LabelEncoder()
            label_encoder.fit(list(df_bureau_balance[col]))
            df_bureau_balance[col] = label_encoder.transform(list(df_bureau_balance[col]))

    # 同じ SK_ID_BUREAU を集約
    df_bureau_balance_agg = df_bureau_balance.groupby('SK_ID_BUREAU', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_bureau_balance_agg.columns = rename_columns_levels( df_bureau_balance_agg, "bureau_balance", 'SK_ID_BUREAU' )

    # 親データ （df_bureau） の 'SK_ID_CURR' に、対応する 'SK_ID_BUREAU' を紐付け
    df_bureau_balance_agg = df_bureau[['SK_ID_BUREAU', 'SK_ID_CURR']].merge(df_bureau_balance_agg, on = 'SK_ID_BUREAU', how = 'left')

    # １つの `SK_ID_CURR` に対して、複数の `SK_ID_BUREAU` が存在することになるので、`SK_ID_CURR` を集約
    df_bureau_balance_agg = df_bureau_balance_agg.drop(columns = ['SK_ID_BUREAU']).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_bureau_balance_agg.columns = rename_columns_levels( df_bureau_balance_agg, "bureau_balance", 'SK_ID_CURR' )
    #print( df_bureau_balance_agg.shape )
    #print( df_bureau_balance_agg.head() )

    # 元のデータに統合
    df_train = pd.merge(df_train, df_bureau_balance_agg, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_bureau_balance_agg, on='SK_ID_CURR', how='left' )
    #print( df_train.shape )
    #print( df_train.head() )

    # 不要になったメモリを解放
    del df_bureau, df_bureau_balance, df_bureau_balance_agg
    gc.collect()

    #---------------------------
    # previous_application
    #--------------------------
    df_previous_application = pd.read_csv( os.path.join(args.dataset_dir, "previous_application.csv" ) )    
    for col in df_previous_application.columns:
        # ラベル情報のエンコード
        if( df_previous_application[col].dtypes == "object" ):
            label_encoder = LabelEncoder()
            label_encoder.fit(list(df_previous_application[col]))
            df_previous_application[col] = label_encoder.transform(list(df_previous_application[col]))

    df_previous_application_agg = df_previous_application.groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_previous_application_agg.columns = rename_columns_levels( df_previous_application_agg, "revious_application", 'SK_ID_CURR' )

    # 元データに統合
    df_train = pd.merge(df_train, df_previous_application_agg, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_previous_application_agg, on='SK_ID_CURR', how='left' )

    # 不要になったメモリを解放
    del df_previous_application_agg
    gc.collect()

    #---------------------------
    # pos_cash_balance
    #---------------------------
    df_pos_cash_balance = pd.read_csv( os.path.join(args.dataset_dir, "POS_CASH_balance.csv" ) )
    for col in df_pos_cash_balance.columns:
        # ラベル情報のエンコード
        if( df_pos_cash_balance[col].dtypes == "object" ):
            label_encoder = LabelEncoder()
            label_encoder.fit(list(df_pos_cash_balance[col]))
            df_pos_cash_balance[col] = label_encoder.transform(list(df_pos_cash_balance[col]))

    df_pos_cash_balance_agg = df_pos_cash_balance.groupby('SK_ID_PREV', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    #print( df_pos_cash_balance_agg.head() )
    df_pos_cash_balance_agg.columns = rename_columns_levels( df_pos_cash_balance_agg, "pos_cash_balance", 'SK_ID_PREV' )

    # 親データの 'SK_ID_CURR' に、対応する 'SK_ID_PREV' を紐付け
    df_pos_cash_balance_agg = df_previous_application[['SK_ID_PREV', 'SK_ID_CURR']].merge(df_pos_cash_balance_agg, on = 'SK_ID_PREV', how = 'left')
    #print( df_pos_cash_balance_agg.head() )

    # １つの `SK_ID_CURR` に対して、複数の `SK_ID_PREV` が存在することになるので、`SK_ID_CURR` を集約
    df_pos_cash_balance_agg = df_pos_cash_balance_agg.drop(columns = ['SK_ID_PREV']).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_pos_cash_balance_agg.columns = rename_columns_levels( df_pos_cash_balance_agg, "bureau_balance", 'SK_ID_CURR' )
    #print( df_pos_cash_balance_agg.head() )

    # 元データに統合
    df_train = pd.merge(df_train, df_pos_cash_balance_agg, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_pos_cash_balance_agg, on='SK_ID_CURR', how='left' )

    # 不要になったメモリを解放
    del df_pos_cash_balance, df_pos_cash_balance_agg
    gc.collect()

    #---------------------------
    # installments_payments
    #---------------------------
    df_installments_payments = pd.read_csv( os.path.join(args.dataset_dir, "installments_payments.csv" ) )
    for col in df_installments_payments.columns:
        # ラベル情報のエンコード
        if( df_installments_payments[col].dtypes == "object" ):
            label_encoder = LabelEncoder()
            label_encoder.fit(list(df_installments_payments[col]))
            df_installments_payments[col] = label_encoder.transform(list(df_installments_payments[col]))

    df_installments_payments_agg = df_installments_payments.groupby('SK_ID_PREV', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_installments_payments_agg.columns = rename_columns_levels( df_installments_payments_agg, "installments_payments", 'SK_ID_PREV' )

    # 親データの 'SK_ID_CURR' に、対応する 'SK_ID_PREV' を紐付け
    df_installments_payments_agg = df_previous_application[['SK_ID_PREV', 'SK_ID_CURR']].merge(df_installments_payments_agg, on = 'SK_ID_PREV', how = 'left')

    # １つの `SK_ID_CURR` に対して、複数の `SK_ID_PREV` が存在することになるので、`SK_ID_CURR` を集約
    df_installments_payments_agg = df_installments_payments_agg.drop(columns = ['SK_ID_PREV']).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_installments_payments_agg.columns = rename_columns_levels( df_installments_payments_agg, "installments_payments", 'SK_ID_CURR' )

    # 元データに統合
    df_train = pd.merge(df_train, df_installments_payments_agg, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_installments_payments_agg, on='SK_ID_CURR', how='left' )

    # 不要になったメモリを解放
    del df_installments_payments, df_installments_payments_agg
    gc.collect()

    #---------------------------
    # credit_card_balance
    #---------------------------
    df_credit_card_balance = pd.read_csv( os.path.join(args.dataset_dir, "credit_card_balance.csv" ) )
    for col in df_credit_card_balance.columns:
        # ラベル情報のエンコード
        if( df_credit_card_balance[col].dtypes == "object" ):
            label_encoder = LabelEncoder()
            label_encoder.fit(list(df_credit_card_balance[col]))
            df_credit_card_balance[col] = label_encoder.transform(list(df_credit_card_balance[col]))

    df_credit_card_balance_agg = df_credit_card_balance.groupby('SK_ID_PREV', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_credit_card_balance_agg.columns = rename_columns_levels( df_credit_card_balance_agg, "credit_card_balance", 'SK_ID_PREV' )

    # 親データの 'SK_ID_CURR' に、対応する 'SK_ID_PREV' を紐付け
    df_credit_card_balance_agg = df_previous_application[['SK_ID_PREV', 'SK_ID_CURR']].merge(df_credit_card_balance_agg, on = 'SK_ID_PREV', how = 'left')

    # １つの `SK_ID_CURR` に対して、複数の `SK_ID_PREV` が存在することになるので、`SK_ID_CURR` を集約
    df_credit_card_balance_agg = df_credit_card_balance_agg.drop(columns = ['SK_ID_PREV']).groupby('SK_ID_CURR', as_index = False).agg(['count', 'mean', 'max', 'min']).reset_index()
    df_credit_card_balance_agg.columns = rename_columns_levels( df_credit_card_balance_agg, "installments_payments", 'SK_ID_CURR' )

    # 元データに統合
    df_train = pd.merge(df_train, df_credit_card_balance_agg, on='SK_ID_CURR', how='left' )
    df_test = pd.merge(df_test, df_credit_card_balance_agg, on='SK_ID_CURR', how='left' )

    # 不要になったメモリを解放
    del df_credit_card_balance, df_credit_card_balance_agg
    gc.collect()

    #===========================
    # 特徴量の追加（結合後）
    #===========================
    df_train['DAYS_EMPLOYED_ANOM'] = df_train["DAYS_EMPLOYED"] == 365243    # 異常値のフラグ
    df_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
    df_test['DAYS_EMPLOYED_ANOM'] = df_test["DAYS_EMPLOYED"] == 365243      # 異常値のフラグ
    df_test['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)

    # 時系列データ
    df_train['DAYS_BIRTH'] = -1 * df_train['DAYS_BIRTH']
    df_test['DAYS_BIRTH'] = -1 * df_test['DAYS_BIRTH']
    df_train['YEARS_BIRTH'] = df_train['DAYS_BIRTH'] / 365
    df_test['YEARS_BIRTH'] = df_test['DAYS_BIRTH'] / 365
    #df_train['YEARS_BINNED'] = pd.cut(df_train['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))
    #df_test['YEARS_BINNED'] = pd.cut(df_test['YEARS_BIRTH'], bins = np.linspace(20, 70, num = 11))

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
        """
        if( df_train[col].nunique() == 1 ):
            print( "remove {} : {}".format(col,df_train[col].nunique()) )
            df_train.drop([col], axis=1, inplace=True)
            df_test.drop([col], axis=1, inplace=True)
        """

    gc.disable()
    return df_train, df_test