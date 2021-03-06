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

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# 自作モジュール
from preprocessing import preprocessing
from models import predict_from_submit_files, XGBoostRegressor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="ensemble_submitfiles", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="house-prices-advanced-regression-techniques")
    parser.add_argument("--params_file", type=str, default="parames/xgboost_regressor_default.yml")
    parser.add_argument("--n_splits", type=int, default=4, help="CV での学習用データセットの分割数")
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
    ds_submission = pd.read_csv( os.path.join(args.dataset_dir, "sample_submission.csv" ) )
    if( args.debug ):
        print( "df_train.head() : \n", df_train.head() )
        print( "df_test.head() : \n", df_test.head() )
        print( "ds_submission.head() : \n", ds_submission.head() )

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
    # データセットの設定
    #===========================================
    # 学習用データセットとテスト用データセットの設定
    X_train = df_train.drop('SalePrice', axis = 1)
    X_test = df_test
    y_train = df_train['SalePrice']
    if( args.debug ):
        print( "len(X_train) : ", len(X_train) )
        print( "len(y_train) : ", len(y_train) )
        #print( "X_train.head() : \n", X_train.head() )
        #print( "X_test.head() : \n", X_test.head() )
        #print( "y_train.head() : \n", y_train.head() )

    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    #================================
    # knn での学習 & 推論
    #================================            
    y_preds_train = np.zeros((len(y_train),))
    y_preds_test = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train)):
        # データセットの分割
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        # 回帰モデル定義
        model = KNeighborsRegressor( n_neighbors = 3, p = 2, metric = 'minkowski', n_jobs = -1 )

        # モデルの学習処理
        model.fit(X_train, y_train)

        # モデルの推論処理
        y_pred_test = model.predict(X_test)
        y_preds_test.append(y_pred_test)
        y_preds_train[valid_index] = model.predict(X_valid_fold)
    
    # 正解データとの平均2乗平方根誤差で評価
    if( args.target_norm ):
        rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_preds_train) ) ) 
    else:
        rmse = np.sqrt( mean_squared_error(y_train, y_preds_train) )

    print( "knn | RMSE [val] : {:0.5f}".format(rmse) )

    # 可視化処理
    sns.distplot(df_train['SalePrice'] )
    sns.distplot(y_preds_train)
    if( args.target_norm ):
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "knn_SalePrice_w_norm.png"), dpi = 300, bbox_inches = 'tight' )
    else:
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "knn_SalePrice_wo_norm.png"), dpi = 300, bbox_inches = 'tight' )

    # submit file に値を設定
    y_preds_test = sum(y_preds_test) / len(y_preds_test)
    #sub = ds_submission.copy()
    if( args.target_norm ):
        ds_submission['SalePrice'] = list(map(float, np.exp(y_preds_test)))
    else:
        ds_submission['SalePrice'] = list(map(float, y_preds_test))

    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, "knn_" + args.submit_file), index=False)

    #================================
    # SVR での学習 & 推論
    #================================
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    y_preds_train = np.zeros((len(y_train),))
    y_preds_test = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train)):
        # データセットの分割
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        # 回帰モデル定義
        model = SVR( kernel = 'rbf', C = 10.0 )

        # モデルの学習処理
        model.fit(X_train, y_train)

        # モデルの推論処理
        y_pred_test = model.predict(X_test)
        y_preds_test.append(y_pred_test)
        y_preds_train[valid_index] = model.predict(X_valid_fold)
    
    # 正解データとの平均2乗平方根誤差で評価
    if( args.target_norm ):
        rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_preds_train) ) ) 
    else:
        rmse = np.sqrt( mean_squared_error(y_train, y_preds_train) )

    print( "svr | RMSE [val] : {:0.5f}".format(rmse) )

    # 可視化処理
    sns.distplot(df_train['SalePrice'] )
    sns.distplot(y_preds_train)
    if( args.target_norm ):
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "svr_SalePrice_w_norm.png"), dpi = 300, bbox_inches = 'tight' )
    else:
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "svr_SalePrice_wo_norm.png"), dpi = 300, bbox_inches = 'tight' )

    # submit file に値を設定
    y_preds_test = sum(y_preds_test) / len(y_preds_test)
    #sub = ds_submission.copy()
    if( args.target_norm ):
        ds_submission['SalePrice'] = list(map(float, np.exp(y_preds_test)))
    else:
        ds_submission['SalePrice'] = list(map(float, y_preds_test))

    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, "svr_" + args.submit_file), index=False)

    #================================
    # random forest での学習 & 推論
    #================================
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    y_preds_train = np.zeros((len(y_train),))
    y_preds_test = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train)):
        # データセットの分割
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        # 回帰モデル定義
        model = RandomForestRegressor( criterion = "mse", bootstrap = True, n_estimators = 1001, oob_score = True, n_jobs = -1, random_state = args.seed )

        # モデルの学習処理
        model.fit(X_train, y_train)

        # モデルの推論処理
        y_pred_test = model.predict(X_test)
        y_preds_test.append(y_pred_test)
        y_preds_train[valid_index] = model.predict(X_valid_fold)
    
    # 正解データとの平均2乗平方根誤差で評価
    if( args.target_norm ):
        rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_preds_train) ) ) 
    else:
        rmse = np.sqrt( mean_squared_error(y_train, y_preds_train) )

    print( "random_forest | RMSE [val] : {:0.5f}".format(rmse) )

    # 可視化処理
    sns.distplot(df_train['SalePrice'] )
    sns.distplot(y_preds_train)
    if( args.target_norm ):
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "random_forest_SalePrice_w_norm.png"), dpi = 300, bbox_inches = 'tight' )
    else:
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "random_forest_SalePrice_wo_norm.png"), dpi = 300, bbox_inches = 'tight' )

    # submit file に値を設定
    y_preds_test = sum(y_preds_test) / len(y_preds_test)
    if( args.target_norm ):
        ds_submission['SalePrice'] = list(map(float, np.exp(y_preds_test)))
    else:
        ds_submission['SalePrice'] = list(map(float, y_preds_test))

    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, "random_forest_" + args.submit_file), index=False)

    #================================
    # XGBoost での学習 & 推論
    #================================
    # モデルのパラメータの読み込み
    with open( args.params_file ) as f:
        params = yaml.safe_load(f)
        model_params = params["model"]["model_params"]
        model_train_params = params["model"]["train_params"]
        if( args.debug ):
            print( "params :\n", params )

    # k-fold CV での学習 & 推論
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    y_preds_train = np.zeros((len(y_train),))
    y_preds_test = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train)):
        # データセットの分割
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        # 回帰モデル定義
        model = xgb.XGBRegressor(
            booster = model_params['booster'],
            objective = model_params['objective'],
            learning_rate = model_params['learning_rate'],
            n_estimators = model_params['n_estimators'],
            max_depth = model_params['max_depth'],
            min_child_weight = model_params['min_child_weight'],
            subsample = model_params['subsample'],
            colsample_bytree = model_params['colsample_bytree'],
            gamma = model_params['gamma'],
            alpha = model_params['alpha'],
            reg_lambda = model_params['reg_lambda'],
            random_state = model_params['random_state']                    
        )

        # モデルの学習処理
        model.fit(X_train, y_train)

        # モデルの推論処理
        y_pred_test = model.predict(X_test)
        y_preds_test.append(y_pred_test)
        y_preds_train[valid_index] = model.predict(X_valid_fold)
    
    # 正解データとの平均2乗平方根誤差で評価
    if( args.target_norm ):
        rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_preds_train) ) ) 
    else:
        rmse = np.sqrt( mean_squared_error(y_train, y_preds_train) )

    print( "xgboost | RMSE [val] : {:0.5f}".format(rmse) )

    # 可視化処理
    sns.distplot(df_train['SalePrice'] )
    sns.distplot(y_preds_train)
    if( args.target_norm ):
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "xgboost_SalePrice_w_norm.png"), dpi = 300, bbox_inches = 'tight' )
    else:
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "xgboost_SalePrice_wo_norm.png"), dpi = 300, bbox_inches = 'tight' )

    # submit file に値を設定
    y_preds_test = sum(y_preds_test) / len(y_preds_test)
    if( args.target_norm ):
        ds_submission['SalePrice'] = list(map(float, np.exp(y_preds_test)))
    else:
        ds_submission['SalePrice'] = list(map(float, y_preds_test))

    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, "xgboost_" + args.submit_file), index=False)

    #================================
    # アンサンブル
    #================================
    y_preds_test = predict_from_submit_files(
        key = "SalePrice",
        weights = [0.01, 0.01, 0.4, 1.00 ],
        submit_files = [ 
            os.path.join(args.results_dir, args.exper_name, "knn_" + args.submit_file),
            os.path.join(args.results_dir, args.exper_name, "svr_" + args.submit_file),
            os.path.join(args.results_dir, args.exper_name, "random_forest_" + args.submit_file),
            os.path.join(args.results_dir, args.exper_name, "xgboost_" + args.submit_file),
         ],
    )

    # 提出用データに値を設定
    if( args.target_norm ):
        ds_submission['SalePrice'] = list(map(float, np.exp(y_preds_test)))
    else:
        ds_submission['SalePrice'] = list(map(float, y_preds_test))

    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)

    #================================
    # Kaggle API での submit
    #================================
    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
