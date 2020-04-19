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
import optuna

from models import EnsembleRegressor, RegressorXGBoost


def objective_wrapper(args, X_train, y_train):
    """
    objective に trial 以外の引数を指定可能にするためのラッパーメソッド
    """
    def objective(trial):
        #--------------------------------------------
        # ベイズ最適化でのチューニングパイパーパラメーター
        #--------------------------------------------
        params = {
            'weights1': trial.suggest_discrete_uniform('weights1', 0.00, 0.50, 0.10),
            'weights2': trial.suggest_discrete_uniform('weights2', 0.00, 0.50, 0.10),
            'weights3': trial.suggest_discrete_uniform('weights3', 0.00, 1.00, 0.10),
            'weights4': trial.suggest_discrete_uniform('weights4', 0.50, 1.00, 0.10),            
        }

        # モデルのパラメータの読み込み
        with open( args.params_file ) as f:
            xgboost_params = yaml.safe_load(f)
            xgboost_model_params = xgboost_params["model"]["model_params"]
            xgboost_train_params = xgboost_params["model"]["train_params"]
            if( args.debug ):
                print( "xgboost_params :\n", xgboost_params )

        #--------------------------------------------
        # k-fold CV での評価
        #--------------------------------------------
        # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
        # StratifiedKFold は連続値では無効なので、通常の k-fold を使用
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train)):
            #--------------------
            # データセットの分割
            #--------------------
            X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

            #--------------------
            # モデルの定義
            #--------------------
            knn = KNeighborsRegressor( n_neighbors = 3, p = 2, metric = 'minkowski', n_jobs = -1 )
            svr = SVR( kernel = 'rbf', C = 0.1 )
            random_forest = RandomForestRegressor( criterion = "mse", bootstrap = True, n_estimators = 1001, oob_score = True, n_jobs = -1, random_state = args.seed )

            xgboost = xgb.XGBRegressor(
                booster = xgboost_model_params['booster'],
                objective = xgboost_model_params['objective'],
                learning_rate = xgboost_model_params['learning_rate'],
                n_estimators = xgboost_model_params['n_estimators'],
                max_depth = xgboost_model_params['max_depth'],
                min_child_weight = xgboost_model_params['min_child_weight'],
                subsample = xgboost_model_params['subsample'],
                colsample_bytree = xgboost_model_params['colsample_bytree'],
                gamma = xgboost_model_params['gamma'],
                alpha = xgboost_model_params['alpha'],
                reg_lambda = xgboost_model_params['reg_lambda'],
                random_state = xgboost_model_params['random_state']                    
            )

            ensemble = EnsembleRegressor(
                regressors  = [ knn, svr, random_forest, xgboost ],
                weights = [ params["weights1"], params["weights2"], params["weights3"], params["weights4"] ],
                fitting = [ True, True, True, True ],
            )

            #--------------------
            # モデルの学習処理
            #--------------------
            ensemble.fit(X_train_fold, y_train_fold)

            #--------------------
            # モデルの推論処理
            #--------------------
            y_pred_val[valid_index] = ensemble.predict(X_valid_fold)

        if( args.target_norm ):
            rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_pred_val) ) ) 
        else:
            rmse = np.sqrt( mean_squared_error(y_train, y_pred_val) )

        return rmse

    return objective

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="ensemble_gridsearch", help="実験名")
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
    ds_train = pd.read_csv( os.path.join(args.dataset_dir, "train.csv" ) )
    ds_test = pd.read_csv( os.path.join(args.dataset_dir, "test.csv" ) )
    ds_sample_submission = pd.read_csv( os.path.join(args.dataset_dir, "sample_submission.csv" ) )
    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )
        print( "ds_sample_submission.head() : \n", ds_sample_submission.head() )

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
    # k-fold CV による処理
    #===========================================
    # 学習用データセットとテスト用データセットの設定
    X_train = ds_train.drop('SalePrice', axis = 1)
    X_test = ds_test
    y_train = ds_train['SalePrice']
    y_pred_val = np.zeros((len(y_train),))
    if( args.debug ):
        print( "len(X_train) : ", len(X_train) )
        print( "len(y_train) : ", len(y_train) )
        print( "len(y_pred_val) : ", len(y_pred_val) )
        #print( "X_train.head() : \n", X_train.head() )
        #print( "X_test.head() : \n", X_test.head() )
        #print( "y_train.head() : \n", y_train.head() )

    #==============================
    # Optuna によるハイパーパラメーターのチューニング
    #==============================
    study = optuna.create_study(direction="minimize")
    study.optimize(objective_wrapper(args, X_train,y_train), n_trials=args.n_trials)
    print('best params : ', study.best_params)
    #print('best best_trial : ', study.best_trial)

    #================================
    # 最良モデルでの学習 & 推論
    #================================
    # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
    # StratifiedKFold は連続値では無効なので、通常の k-fold を使用
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    y_preds = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train)):
        #--------------------
        # データセットの分割
        #--------------------
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        #--------------------
        # 回帰モデル定義
        #--------------------
        knn = KNeighborsRegressor( n_neighbors = 3, p = 2, metric = 'minkowski', n_jobs = -1 )
        svr = SVR( kernel = 'rbf', C = 0.1 )
        random_forest = RandomForestRegressor( criterion = "mse", bootstrap = True, n_estimators = 1001, oob_score = True, n_jobs = -1, random_state = args.seed )

        xgboost = xgb.XGBRegressor(
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

        ensemble = EnsembleRegressor(
            regressors  = [ knn, svr, random_forest, xgboost ],
            weights = [study.best_params["weights1"], study.best_params["weights2"], study.best_params["weights3"], study.best_params["weights4"] ],
            fitting = [ True, True, True, True ],
        )

        #--------------------
        # モデルの学習処理
        #--------------------
        ensemble.fit(X_train, y_train)

        #--------------------
        # モデルの推論処理
        #--------------------
        y_pred_test = ensemble.predict(X_test)
        y_preds.append(y_pred_test)

        y_pred_val[valid_index] = ensemble.predict(X_valid_fold)
        #print( "[{}] len(y_pred_fold) : {}".format(fold_id, len(y_pred_val)) )
    
    # 正解データとの平均2乗平方根誤差で評価
    if( args.target_norm ):
        rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_pred_val) ) ) 
    else:
        rmse = np.sqrt( mean_squared_error(y_train, y_pred_val) )

    print( "RMSE [val] : {:0.5f}".format(rmse) )

    #================================
    # 可視化処理
    #================================
    # 回帰対象
    sns.distplot(ds_train['SalePrice'] )
    sns.distplot(y_pred_val)
    if( args.target_norm ):
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "SalePrice_w_norm.png"), dpi = 300, bbox_inches = 'tight' )
    else:
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "SalePrice_wo_norm.png"), dpi = 300, bbox_inches = 'tight' )

    #================================
    # Kaggle API での submit
    #================================
    # 提出用データに値を設定
    y_sub = sum(y_preds) / len(y_preds)
    sub = ds_sample_submission
    if( args.target_norm ):
        sub['SalePrice'] = list(map(float, np.exp(y_sub)))
    else:
        sub['SalePrice'] = list(map(float, y_sub))

    sub.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)

    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )