import os
import argparse
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import random
from matplotlib import pyplot as plt
import seaborn as sns
from kaggle.api.kaggle_api_extended import KaggleApi

# scikit-learn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="logistic", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="titanic")
    parser.add_argument("--val_rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name))

    # seed 値の固定
    np.random.seed(args.seed)
    random.seed(args.seed)

    #================================
    # データセットの読み込み
    #================================
    ds_train = pd.read_csv( os.path.join(args.dataset_dir, "train.csv" ) )
    ds_test = pd.read_csv( os.path.join(args.dataset_dir, "test.csv" ) )
    ds_gender_submission = pd.read_csv( os.path.join(args.dataset_dir, "gender_submission.csv" ) )
    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )
        print( "ds_gender_submission.head() : \n", ds_gender_submission.head() )

    #================================
    # 前処理
    #================================
    # 無用なデータを除外
    ds_train.drop(['Name', 'PassengerId'], axis=1, inplace=True)
    ds_test.drop(['Name', 'PassengerId'], axis=1, inplace=True)
    ds_train.drop(['SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
    ds_test.drop(['SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # 全特徴量を一括で処理
    for col in ds_train.columns:
        if( args.debug ):
            print( "ds_train[{}].dtypes ] : {}".format(col, ds_train[col].dtypes))

        # 目的変数
        if( col in ["Survived"] ):
            continue

        #-----------------------------
        # 欠損値の埋め合わせ
        #-----------------------------
        # NAN 値の埋め合わせ（平均値）
        if( col in ["Age", 'Fare'] ):
            ds_train[col].fillna(np.mean(ds_train[col]), inplace=True)
            ds_test[col].fillna(np.mean(ds_train[col]), inplace=True)
        # NAN 値の埋め合わせ（ゼロ値）/ int 型
        elif( ds_train[col].dtypes in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"] ):
            ds_train[col].fillna(0, inplace=True)
            ds_test[col].fillna(0, inplace=True)
        # NAN 値の埋め合わせ（ゼロ値）/ float 型
        elif( ds_train[col].dtypes in ["float16", "float32", "float64", "float128"] ):
            ds_train[col].fillna(0.0, inplace=True)
            ds_test[col].fillna(0.0, inplace=True)
        # NAN 値の補完（None値）/ object 型
        else:
            ds_train[col] = ds_train[col].fillna('NA')
            ds_test[col] = ds_test[col].fillna('NA')

        #-----------------------------
        # 正規化処理
        #-----------------------------
        if( ds_train[col].dtypes != "object" ):
            scaler = StandardScaler()
            scaler.fit( ds_train[col].values.reshape(-1,1) )
            ds_train[col] = scaler.fit_transform( ds_train[col].values.reshape(-1,1) )
            ds_test[col] = scaler.fit_transform( ds_test[col].values.reshape(-1,1) )

        #-----------------------------
        # ラベル情報のエンコード
        #-----------------------------
        if( ds_train[col].dtypes == "object" ):
            label_encoder = LabelEncoder()
            label_encoder.fit(list(ds_train[col]))
            ds_train[col] = label_encoder.transform(list(ds_train[col]))

            label_encoder = LabelEncoder()
            label_encoder.fit(list(ds_test[col]))
            ds_test[col] = label_encoder.transform(list(ds_test[col]))

    # 前処理後のデータセットを外部ファイルに保存
    ds_train.to_csv( os.path.join(args.results_dir, args.exper_name, "train_preprocessed.csv"), index=True)
    ds_test.to_csv( os.path.join(args.results_dir, args.exper_name, "test_preprocessed.csv"), index=True)
    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )

    #================================
    # データセットの分割
    #================================
    # 学習用データセットとテスト用データセットの設定
    X_train = ds_train.drop('Survived', axis = 1)
    y_train = ds_train['Survived']
    X_test = ds_test
    if( args.debug ):
        print( "X_train.head() : \n", X_train.head() )
        print( "X_test.head() : \n", X_test.head() )
        print( "y_train.head() : \n", y_train.head() )
        print( "len(X_train) : ", len(X_train) )
        print( "len(X_test) : ", len(X_test) )
        print( "len(y_train) : ", len(y_train) )

    # stratify 引数で y_train を指定することで、割合を保ったままデータセットを2つに分割
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args.val_rate, random_state=args.seed, stratify=y_train)
    if( args.debug ):
        print( "X_valid.head() : \n", X_valid.head() )
        print( "y_valid.head() : \n", y_valid.head() )
        print( "len(X_valid) : ", len(X_valid) )
        print( "len(y_valid) : ", len(y_valid) )

    #================================
    # モデルの定義
    #================================
    # ロジスティクス回帰
    model = LogisticRegression( penalty='l2', solver="sag", random_state=args.seed )

    #================================
    # モデルの学習処理
    #================================
    model.fit(X_train, y_train)

    #================================
    # モデルの推論処理
    #================================
    y_pred = model.predict(X_test)
    y_pred_val = model.predict(X_valid)
    print( "y_pred : ", y_pred[:100] ) 

    # 正解率の計算
    print( "accuracy [val] : {:0.5f}".format( (y_valid == y_pred_val).sum()/len(y_pred_val) ) )

    #================================
    # 可視化処理
    #================================
    pass

    #================================
    # Kaggle API での submit
    #================================
    # 提出用データに値を設定
    sub = ds_gender_submission
    sub['Survived'] = list(map(int, y_pred))
    sub.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)
    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
