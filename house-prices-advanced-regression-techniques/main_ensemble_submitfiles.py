import os
import argparse
import numpy as np
import pandas as pd
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

from models import EnsembleRegressor, RegressorXGBoost

# XGBoost のデフォルトハイパーパラメーター
params_xgboost = {
    'booster': 'gbtree',
    'objective': 'reg:linear',          # 線形回帰
    "learning_rate" : 0.01,             # ハイパーパラメーターのチューニング時は 0.1 で固定  
    "n_estimators" : 1050,              # 
    'max_depth': 6,                     # 3 ~ 9 : 一様分布に従う。1刻み
    'min_child_weight': 1,              # 0.1 ~ 10.0 : 対数が一様分布に従う
    'subsample': 0.8,                   # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
    'colsample_bytree': 0.8,            # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
    'gamma': 0.0,                       # 1e-8 ~ 1.0 : 対数が一様分布に従う
    'alpha': 0.0,                       # デフォルト値としておく。余裕があれば変更
    'reg_lambda': 1.0,                  # デフォルト値としておく。余裕があれば変更
    'eval_metric': 'rmse',              # 平均2乗平方根誤差
    "num_boost_round": 1000,            # 試行回数
    "early_stopping_rounds": 100,       # early stopping を行う繰り返し回数
    'random_state': 71,
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="ensemble_submitfiles", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="house-prices-advanced-regression-techniques")
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
    ds_train = pd.read_csv( os.path.join(args.dataset_dir, "train.csv" ) )
    ds_test = pd.read_csv( os.path.join(args.dataset_dir, "test.csv" ) )
    ds_submission = pd.read_csv( os.path.join(args.dataset_dir, "sample_submission.csv" ) )
    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )
        print( "ds_submission.head() : \n", ds_submission.head() )

    #================================
    # 前処理
    #================================    
    # 無用なデータを除外
    ds_train.drop(["Id"], axis=1, inplace=True)
    ds_test.drop(["Id"], axis=1, inplace=True)

    # 全特徴量を一括で処理
    for col in ds_train.columns:
        if( args.debug ):
            print( "ds_train[{}].dtypes ] : {}".format(col, ds_train[col].dtypes))

        # 目的変数
        if( col in ["SalePrice"] ):
            if( args.target_norm ):
                # 正規分布に従うように対数化
                ds_train[col] = pd.Series( np.log(ds_train[col].values), name=col )
                #ds_train[col] = pd.DataFrame( pd.Series( np.log(ds_train[col].values) ) )
                #ds_train[col] = list(map(float, np.log(ds_train[col].values)))

            continue

        #-----------------------------
        # 欠損値の埋め合わせ
        #-----------------------------
        # NAN 値の埋め合わせ（平均値）
        if( col in ["LotFrontage"] ):
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
        #if( ds_train[col].dtypes != "object" ):
        if( ds_train[col].dtypes in ["float16", "float32", "float64", "float128"] ):
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

    #===========================================
    # データセットの設定
    #===========================================
    # 学習用データセットとテスト用データセットの設定
    X_train = ds_train.drop('SalePrice', axis = 1)
    X_test = ds_test
    y_train = ds_train['SalePrice']
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
    y_pred_val = np.zeros((len(y_train),))
    y_preds = []
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
        y_preds.append(y_pred_test)
        y_pred_val[valid_index] = model.predict(X_valid_fold)
    
    # 正解データとの平均2乗平方根誤差で評価
    if( args.target_norm ):
        rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_pred_val) ) ) 
    else:
        rmse = np.sqrt( mean_squared_error(y_train, y_pred_val) )

    print( "knn | RMSE [val] : {:0.5f}".format(rmse) )

    # 可視化処理
    sns.distplot(ds_train['SalePrice'] )
    sns.distplot(y_pred_val)
    if( args.target_norm ):
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "knn_SalePrice_w_norm.png"), dpi = 300, bbox_inches = 'tight' )
    else:
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "knn_SalePrice_wo_norm.png"), dpi = 300, bbox_inches = 'tight' )

    # submit file に値を設定
    y_preds = sum(y_preds) / len(y_preds)
    #sub = ds_submission.copy()
    if( args.target_norm ):
        ds_submission['SalePrice'] = list(map(float, np.exp(y_preds)))
    else:
        ds_submission['SalePrice'] = list(map(float, y_preds))

    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, "knn_" + args.submit_file), index=False)

    #================================
    # SVR での学習 & 推論
    #================================
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    y_pred_val = np.zeros((len(y_train),))
    y_preds = []
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
        y_preds.append(y_pred_test)
        y_pred_val[valid_index] = model.predict(X_valid_fold)
    
    # 正解データとの平均2乗平方根誤差で評価
    if( args.target_norm ):
        rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_pred_val) ) ) 
    else:
        rmse = np.sqrt( mean_squared_error(y_train, y_pred_val) )

    print( "svr | RMSE [val] : {:0.5f}".format(rmse) )

    # 可視化処理
    sns.distplot(ds_train['SalePrice'] )
    sns.distplot(y_pred_val)
    if( args.target_norm ):
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "svr_SalePrice_w_norm.png"), dpi = 300, bbox_inches = 'tight' )
    else:
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "svr_SalePrice_wo_norm.png"), dpi = 300, bbox_inches = 'tight' )

    # submit file に値を設定
    y_preds = sum(y_preds) / len(y_preds)
    #sub = ds_submission.copy()
    if( args.target_norm ):
        ds_submission['SalePrice'] = list(map(float, np.exp(y_preds)))
    else:
        ds_submission['SalePrice'] = list(map(float, y_preds))

    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, "svr_" + args.submit_file), index=False)

    #================================
    # random forest での学習 & 推論
    #================================
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    y_pred_val = np.zeros((len(y_train),))
    y_preds = []
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
        y_preds.append(y_pred_test)
        y_pred_val[valid_index] = model.predict(X_valid_fold)
    
    # 正解データとの平均2乗平方根誤差で評価
    if( args.target_norm ):
        rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_pred_val) ) ) 
    else:
        rmse = np.sqrt( mean_squared_error(y_train, y_pred_val) )

    print( "random_forest | RMSE [val] : {:0.5f}".format(rmse) )

    # 可視化処理
    sns.distplot(ds_train['SalePrice'] )
    sns.distplot(y_pred_val)
    if( args.target_norm ):
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "random_forest_SalePrice_w_norm.png"), dpi = 300, bbox_inches = 'tight' )
    else:
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "random_forest_SalePrice_wo_norm.png"), dpi = 300, bbox_inches = 'tight' )

    # submit file に値を設定
    y_preds = sum(y_preds) / len(y_preds)
    #sub = ds_submission.copy()
    if( args.target_norm ):
        ds_submission['SalePrice'] = list(map(float, np.exp(y_preds)))
    else:
        ds_submission['SalePrice'] = list(map(float, y_preds))

    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, "random_forest_" + args.submit_file), index=False)

    #================================
    # XGBoost での学習 & 推論
    #================================
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    y_pred_val = np.zeros((len(y_train),))
    y_preds = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train)):
        # データセットの分割
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        # 回帰モデル定義
        model = xgb.XGBRegressor(
            booster = params_xgboost['booster'],
            objective = params_xgboost['objective'],
            learning_rate = params_xgboost['learning_rate'],
            n_estimators = params_xgboost['n_estimators'],
            max_depth = params_xgboost['max_depth'],
            min_child_weight = params_xgboost['min_child_weight'],
            subsample = params_xgboost['subsample'],
            colsample_bytree = params_xgboost['colsample_bytree'],
            gamma = params_xgboost['gamma'],
            alpha = params_xgboost['alpha'],
            reg_lambda = params_xgboost['reg_lambda'],
            random_state = params_xgboost['random_state']                    
        )

        # モデルの学習処理
        model.fit(X_train, y_train)

        # モデルの推論処理
        y_pred_test = model.predict(X_test)
        y_preds.append(y_pred_test)
        y_pred_val[valid_index] = model.predict(X_valid_fold)
    
    # 正解データとの平均2乗平方根誤差で評価
    if( args.target_norm ):
        rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_pred_val) ) ) 
    else:
        rmse = np.sqrt( mean_squared_error(y_train, y_pred_val) )

    print( "xgboost | RMSE [val] : {:0.5f}".format(rmse) )

    # 可視化処理
    sns.distplot(ds_train['SalePrice'] )
    sns.distplot(y_pred_val)
    if( args.target_norm ):
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "xgboost_SalePrice_w_norm.png"), dpi = 300, bbox_inches = 'tight' )
    else:
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "xgboost_SalePrice_wo_norm.png"), dpi = 300, bbox_inches = 'tight' )

    # submit file に値を設定
    y_preds = sum(y_preds) / len(y_preds)
    #sub = ds_submission.copy()
    if( args.target_norm ):
        ds_submission['SalePrice'] = list(map(float, np.exp(y_preds)))
    else:
        ds_submission['SalePrice'] = list(map(float, y_preds))

    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, "xgboost_" + args.submit_file), index=False)

    #================================
    # アンサンブル
    #================================
    ensemble = EnsembleRegressor(
        weights = [0.01, 0.01, 0.4, 1.00 ],
    )

    y_preds = ensemble.predict_from_submit_files(
        key = "SalePrice",
        submit_files = [ 
            os.path.join(args.results_dir, args.exper_name, "knn_" + args.submit_file),
            os.path.join(args.results_dir, args.exper_name, "svr_" + args.submit_file),
            os.path.join(args.results_dir, args.exper_name, "random_forest_" + args.submit_file),
            os.path.join(args.results_dir, args.exper_name, "xgboost_" + args.submit_file),
         ],
    )

    # 提出用データに値を設定
    if( args.target_norm ):
        ds_submission['SalePrice'] = list(map(float, np.exp(y_preds)))
    else:
        ds_submission['SalePrice'] = list(map(float, y_preds))

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
