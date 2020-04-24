# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import yaml
import random
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from kaggle.api.kaggle_api_extended import KaggleApi

# sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

# 機械学習モデル
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# 自作モジュール
from preprocessing import preprocessing
from models import SklearnRegressor, XGBoostRegressor
from models import StackingEnsembleRegressor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="ensemble_stacking", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="house-prices-advanced-regression-techniques")
    parser.add_argument("--params_file", type=str, default="parames/xgboost_regressor_default.yml")
    parser.add_argument("--n_splits", type=int, default=4, help="CV での学習用データセットの分割数")
    parser.add_argument("--n_splits_gs", type=int, default=1, help="ハイパーパラメーターチューニング時の CV での学習用データセットの分割数")
    parser.add_argument("--n_trials", type=int, default=50, help="Optuna での試行回数")
    parser.add_argument('--target_norm', action='store_true')
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

    #===========================================
    # 学習用データセットとテスト用データセットの設定
    #===========================================
    # 学習用データセットとテスト用データセットの設定
    X_train = df_train.drop('SalePrice', axis = 1)
    X_test = df_test
    y_train = df_train['SalePrice']
    y_preds_train = np.zeros((len(y_train),))
    if( args.debug ):
        print( "len(X_train) : ", len(X_train) )
        print( "len(y_train) : ", len(y_train) )
        print( "len(y_preds_train) : ", len(y_preds_train) )

    #===========================================
    # モデルの学習 & 推論処理
    #===========================================
    #--------------------
    # モデル定義
    #--------------------
    logistic1 = SklearnRegressor( LogisticRegression( penalty='l2', solver="sag", random_state=args.seed ) )
    logistic2 = SklearnRegressor( LogisticRegression( penalty='l2', solver="sag", random_state=args.seed ) )
    logistic3 = SklearnRegressor( LogisticRegression( penalty='l2', solver="sag", random_state=args.seed ) )
    knn1 = SklearnRegressor( KNeighborsRegressor( n_neighbors = 3, p = 2, metric = 'minkowski', n_jobs = -1 ) )
    svc1 = SklearnRegressor( SVR( kernel = 'rbf', C = 0.1 ) )
    forest1 = SklearnRegressor( RandomForestRegressor( criterion = "mse", bootstrap = True, n_estimators = 1001, n_jobs = -1, random_state = args.seed, oob_score = True ) )
    xgboost1 = XGBoostRegressor( use_valid = True, debug = args.debug )
    xgboost1.load_params( "parames/xgboost_regressor_default.yml" )
    xgboost2 = XGBoostRegressor( use_valid = True, debug = args.debug )
    xgboost2.load_params( "parames/xgboost_regressor_default.yml" )

    # アンサンブルモデル（２段）
    """
    model = StackingEnsembleRegressor(
        regressors  = [ knn1, logistic1, svc1, forest1, xgboost1, dnn1 ],
        final_regressors = logistic2,
        n_splits = args.n_splits,
        seed = args.seed,
    )
    """

    # アンサンブルモデル（３段）
    model = StackingEnsembleRegressor(
        regressors  = [ knn1, logistic1, svc1, forest1, xgboost1 ],
        second_regressors  = [ logistic2, xgboost2 ],
        final_regressors = logistic3,
        n_splits = args.n_splits,
        seed = args.seed,
    )

    #--------------------
    # モデルの学習処理
    #--------------------
    model.fit(X_train, y_train, X_test)

    #--------------------
    # モデルの推論処理
    #--------------------
    y_preds_train = model.y_preds_train
    y_preds_test = model.y_preds_test

    # 正解データとの平均2乗平方根誤差で評価
    if( args.target_norm ):
        rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_preds_train) ) ) 
    else:
        rmse = np.sqrt( mean_squared_error(y_train, y_preds_train) )

    print( "RMSE [k-fold CV train-valid] : {:0.5f}".format(rmse) )

    #================================
    # 可視化処理
    #================================
    # 回帰対象
    sns.distplot(df_train['SalePrice'] )
    sns.distplot(y_preds_train)
    if( args.target_norm ):
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "SalePrice_w_norm.png"), dpi = 300, bbox_inches = 'tight' )
    else:
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "SalePrice_wo_norm.png"), dpi = 300, bbox_inches = 'tight' )
    
    #------------------
    # 重要特徴量
    #------------------
    _, ax = plt.subplots(figsize=(8, 16))
    xgb.plot_importance(
        xgboost1.model,
        ax = ax,
        importance_type = 'gain',
        show_values = False
    )
    plt.tight_layout()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "feature_importances.png"), dpi = 300, bbox_inches = 'tight' )

    #================================
    # Kaggle API での submit
    #================================
    # 提出用データに値を設定
    y_sub = list(map(int, y_preds_test))
    df_submission['Survived'] = y_sub
    df_submission.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)
    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
