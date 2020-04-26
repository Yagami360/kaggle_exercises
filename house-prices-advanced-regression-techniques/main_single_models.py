import os
import argparse
import numpy as np
import pandas as pd
import random
import warnings
import json
import yaml
from matplotlib import pyplot as plt
import seaborn as sns
from kaggle.api.kaggle_api_extended import KaggleApi

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 機械学習モデル
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
import lightgbm as lgb
import catboost

# 自作モジュール
from preprocessing import preprocessing
from models import SklearnRegressor, XGBoostRegressor, LightGBMRegressor, CatBoostRegressor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="single_model", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="house-prices-advanced-regression-techniques")
    parser.add_argument("--classifier", choices=["logistic", "knn", "svm", "random_forest", "bagging", "adaboost", "xgboost", "lightgbm", "catboost", "mlp"], default="catboost", help="チューニングするモデル")
    parser.add_argument("--params_file", type=str, default="")
    parser.add_argument('--train_type', choices=['train', 'fit'], default="fit", help="GDBTの学習タイプ")
    parser.add_argument("--n_splits", type=int, default=4, help="CV での学習用データセットの分割数")
    parser.add_argument('--target_norm', action='store_true')
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # 実験名を自動的に変更
    if( args.exper_name == "single_model" ):
        args.exper_name = args.exper_name + "_" + args.classifier

    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    if not os.path.isdir( os.path.join(args.results_dir, args.exper_name) ):
        os.mkdir(os.path.join(args.results_dir, args.exper_name))

    # 警告非表示
    warnings.simplefilter('ignore', DeprecationWarning)

    # seed 値の固定
    np.random.seed(args.seed)
    random.seed(args.seed)

    #================================
    # データセットの読み込み
    #================================
    df_train = pd.read_csv( os.path.join(args.dataset_dir, "train.csv" ) )
    df_test = pd.read_csv( os.path.join(args.dataset_dir, "test.csv" ) )
    df_submission = pd.read_csv( os.path.join(args.dataset_dir, "sample_submission.csv" ) )
    if( args.debug ):
        print( "df_train.head() : \n", df_train.head() )
        print( "df_test.head() : \n", df_test.head() )
        print( "df_submission.head() : \n", df_submission.head() )

    #================================
    # 前処理
    #================================
    df_train, df_test = preprocessing( args, df_train, df_test )

    # 前処理後のデータセットを外部ファイルに保存
    df_train.to_csv( os.path.join(args.results_dir, args.exper_name, "train_preprocessed.csv"), index=True)
    df_test.to_csv( os.path.join(args.results_dir, args.exper_name, "test_preprocessed.csv"), index=True)
    if( args.debug ):
        print( "df_train.head() : \n", df_train.head() )
        print( "df_test.head() : \n", df_test.head() )

    #==============================
    # 学習用データセットの分割
    #==============================
    X_train = df_train.drop('SalePrice', axis = 1)
    X_test = df_test
    y_train = df_train['SalePrice']
    y_preds_train = np.zeros((len(y_train),))
    if( args.debug ):
        print( "len(X_train) : ", len(X_train) )
        print( "len(y_train) : ", len(y_train) )
        print( "len(y_preds_train) : ", len(y_preds_train) )

    #================================
    # 最良モデルでの学習 & 推論
    #================================    
    # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
    # StratifiedKFold は連続値では無効なので、通常の k-fold を使用
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    y_preds_test = []
    k = 0
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train)):
        #--------------------
        # データセットの分割
        #--------------------
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        #--------------------
        # モデルの定義
        #--------------------
        if( args.classifier == "logistic" ):
            model = SklearnRegressor( model = LogisticRegression( penalty='l2', solver="sag", random_state=args.seed ), debug = args.debug )
        elif( args.classifier == "knn" ):
            model = SklearnRegressor( model = KNeighborsRegressor( n_neighbors = 3, p = 2, metric = 'minkowski' ), debug = args.debug )
        elif( args.classifier == "svm" ):
            model = SklearnRegressor( model = SVR( kernel = 'rbf', gamma = 0.1, C = 10.0 ), debug = args.debug )
        elif( args.classifier == "random_forest" ):
            model = SklearnRegressor( model = RandomForestRegressor( criterion = "mse", bootstrap = True, oob_score = True, n_estimators = 1000, n_jobs = -1, random_state = args.seed ), debug = args.debug )
        elif( args.classifier == "bagging" ):
            model = SklearnRegressor( model = BaggingRegressor( DecisionTreeRegressor(criterion = 'mse', max_depth = None, random_state = args.seed ) ), debug = args.debug )
        elif( args.classifier == "adaboost" ):
            model = SklearnRegressor( model = AdaBoostRegressor( DecisionTreeRegressor(criterion = 'mse', max_depth = None, random_state = args.seed ) ), debug = args.debug )
        elif( args.classifier == "xgboost" ):
            model = XGBoostRegressor( model = xgb.XGBRegressor( booster='gbtree', objective='reg:linear', eval_metric='rmse', learning_rate = 0.01 ), train_type = args.train_type, use_valid = True, debug = args.debug )
            model.load_params( "parames/xgboost_regressor_default.yml" )
        elif( args.classifier == "lightgbm" ):
            model = LightGBMRegressor( model = lgb.LGBMRegressor( objective='regression', metric='rmse' ), train_type = args.train_type, use_valid = True, debug = args.debug )
        elif( args.classifier == "catboost" ):
            model = CatBoostRegressor( model = catboost.CatBoostRegressor(), use_valid = True, debug = args.debug )

        # モデルのパラメータ設定
        if not( args.params_file == "" ):
            model.set_params( args.params_file )

        #--------------------
        # モデルの学習処理
        #--------------------
        model.fit(X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )

        #--------------------
        # モデルの推論処理
        #--------------------
        y_preds_train[valid_index] = model.predict(X_valid_fold)
        y_pred_test = model.predict(X_test)
        y_preds_test.append(y_pred_test)

        #--------------------
        # 可視化処理
        #--------------------
        # 損失関数
        model.plot_loss( os.path.join(args.results_dir, args.exper_name, "losees_k{}.png".format(k) ) )
        k += 1

    # k-fold CV で平均化
    y_preds_test = sum(y_preds_test) / len(y_preds_test)

    # 負の値は０にする
    print( y_preds_test.shape )
    for i, y_pred in enumerate(y_preds_test):
        if( y_pred < 0 ):
            y_preds_test[i] = 0

    # 正解データとの平均2乗平方根誤差で評価
    if( args.target_norm ):
        rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_preds_train) ) ) 
    else:
        rmse = np.sqrt( mean_squared_error(y_train, y_preds_train) )

    print( "RMSE [k-fold CV train-valid] : {:0.5f}".format(rmse) )

    #================================
    # 可視化処理
    #================================
    # 重要特徴量
    if( args.classifier in ["random_forest", "adaboost", "xgboost", "lightgbm", "catboost"] ):
        model.plot_importance( os.path.join(args.results_dir, args.exper_name, "feature_importances.png") )

    # 回帰対象
    _, axis = plt.subplots()
    sns.distplot(df_train['SalePrice'], label = "correct", ax=axis )
    sns.distplot(y_preds_train, label = "predict", ax=axis )
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "SalePrice.png"), dpi = 300, bbox_inches = 'tight' )

    #================================
    # Kaggle API での submit
    #================================
    # 提出用データに値を設定
    if( args.target_norm ):
        df_submission['SalePrice'] = list(map(float, np.exp(y_preds_test)))
    else:
        df_submission['SalePrice'] = list(map(int, y_preds_test))

    df_submission.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)

    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
