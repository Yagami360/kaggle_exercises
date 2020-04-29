import os
import argparse
import numpy as np
import pandas as pd
import random
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
import json
import yaml
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

# Optuna
import optuna

# 自作モジュール
from preprocessing import preprocessing
from models import SklearnRegressor, XGBoostRegressor, LightGBMRegressor, CatBoostRegressor


def objective_wrapper(args, X_train, y_train):
    """
    objective に trial 以外の引数を指定可能にするためのラッパーメソッド
    """
    def objective(trial):
        #--------------------------------------------
        # ベイズ最適化でのチューニングパイパーパラメーター
        #--------------------------------------------
        if( args.regressor == "logistic" ):
            params = {
                'penalty': trial.suggest_categorical('penalty', ['l2']),
                "solver" : trial.suggest_categorical("solver", ['sag']), 
                'C': trial.suggest_discrete_uniform('C', 0.01, 100.0, 0.1),          # 一様分布に従う。
                'random_state': trial.suggest_int("random_state", 71, 71),
                'n_jobs': trial.suggest_int("n_jobs", -1, -1),
            }
        elif( args.regressor == "knn" ):
            params = {
                "metric" : trial.suggest_categorical("metric", ['minkowski']), 
                "p" : trial.suggest_int("p", 1, 2),
                'n_neighbors': trial.suggest_int("n_neighbors", 1, 50),
                'n_jobs': trial.suggest_int("n_jobs", -1, -1),
            }
        elif( args.regressor == "svm" ):
            params = {
                "kernel" : trial.suggest_categorical("kernel", ['rbf']), 
                'C': trial.suggest_loguniform('C', 0.1, 1000.0),
                'gamma': trial.suggest_loguniform('gamma', 1e-8, 10.0),
            }
        elif( args.regressor == "random_forest" ):
            params = {
                "oob_score" : trial.suggest_int("oob_score", 0, 1),                         # Whether to use out-of-bag samples to estimate the generalization accuracy.(default=False)
                "n_estimators" : trial.suggest_int("n_estimators", 1000, 1000),             # チューニングは固定
                "criterion" : trial.suggest_categorical("criterion", ["mse"]),              # 不純度関数 [purity]
                'max_features': trial.suggest_categorical('max_features', ['auto', 0.2, 0.4, 0.6, 0.8,]),                            
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),         # min_samples_split must be an integer greater than 1 or a float in (0.0, 1.0]; got the integer 1                   
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10), 
                "bootstrap" : trial.suggest_int("bootstrap", True, True),                   # 決定木の構築に、ブートストラップサンプルを使用するか否か（default:True）
                "oob_score" : trial.suggest_int("oob_score", False, True),                  # Whether to use out-of-bag samples to estimate the generalization accuracy.(default=False)
                'random_state': trial.suggest_int("random_state", 71, 71),
                'n_jobs': trial.suggest_int("n_jobs", -1, -1),
            }
        elif( args.regressor == "bagging" ):
            params = {
                "n_estimators" : trial.suggest_int("n_estimators", 1000, 1000),             # チューニングは固定
                'max_samples': trial.suggest_float('max_samples', 0.0, 1.0),                # base_estimator に設定した弱識別器の内, 使用するサンプルの割合
                'max_features': trial.suggest_float('max_features', 0.0, 1.0),              # The number of features to draw from X to train each base estimator.
                "bootstrap" : trial.suggest_int("bootstrap", True, True),                   # 決定木の構築に、ブートストラップサンプルを使用するか否か（default:True）
                "bootstrap_features" : trial.suggest_int("bootstrap", False, True),
                'random_state': trial.suggest_int("random_state", 71, 71),
                'n_jobs': -1,
                # 弱識別器のパラメータ（先頭に "base_estimator__" をつけることでアクセス可能 ）                
                "base_estimator__criterion" : trial.suggest_categorical("base_estimator__criterion", ["mse"]),
                'base_estimator__max_depth': trial.suggest_int("base_estimator__random_state", 1, 8),
                'base_estimator__max_features': trial.suggest_float('base_estimator__max_features', 0.0, 1.0),
                'base_estimator__min_samples_leaf': trial.suggest_int('base_estimator__min_samples_leaf', 1, 10),
                'base_estimator__min_samples_split': trial.suggest_int('base_estimator__min_samples_split', 2, 10),
                'base_estimator__random_state': trial.suggest_int("base_estimator__random_state", 71, 71),
            }
        elif( args.regressor == "adaboost" ):
            params = {
                "n_estimators" : trial.suggest_int("n_estimators", 1000, 1000),             # チューニングは固定
                "learning_rate" : trial.suggest_loguniform("learning_rate", 0.01, 0.01),    # ハイパーパラメーターのチューニング時は固定  
                'random_state': 71,
                # 弱識別器のパラメータ（先頭に "base_estimator__" をつけることでアクセス可能 ）                
                "base_estimator__criterion" : trial.suggest_categorical("base_estimator__criterion", ["mse"]),
                'base_estimator__max_depth': trial.suggest_int("base_estimator__random_state", 1, 10),
                'base_estimator__max_features': trial.suggest_float('base_estimator__max_features', 0.0, 1.0),
                'base_estimator__min_samples_leaf': trial.suggest_int('base_estimator__min_samples_leaf', 1, 10),
                'base_estimator__min_samples_split': trial.suggest_int('base_estimator__min_samples_split', 2, 10),
                'base_estimator__random_state': trial.suggest_int("base_estimator__random_state", 71, 71),
            }
        elif( args.regressor == "xgboost" ):
            params = {
                'booster': trial.suggest_categorical('booster', ['gbtree']),
                'objective': trial.suggest_categorical('objective', ['reg:linear']),
                "learning_rate" : trial.suggest_loguniform("learning_rate", 0.01, 0.01),                      # ハイパーパラメーターのチューニング時は固定  
                "n_estimators" : trial.suggest_int("n_estimators", 1000, 1000),                               # ハイパーパラメーターのチューニング時は固定
                'max_depth': trial.suggest_int("max_depth", 3, 9),                                            # 3 ~ 9 : 一様分布に従う。1刻み
                'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.1, 10.0),                  # 0.1 ~ 10.0 : 対数が一様分布に従う
                'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 0.95, 0.05),                    # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
                'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 0.95, 0.05),      # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
                'gamma': trial.suggest_loguniform("gamma", 1e-8, 1.0),                                        # 1e-8 ~ 1.0 : 対数が一様分布に従う
                'alpha': trial.suggest_float("alpha", 0.0, 0.0),                                              # デフォルト値としておく。余裕があれば変更
                'reg_lambda': trial.suggest_float("reg_lambda", 1.0, 1.0),                                    # デフォルト値としておく。余裕があれば変更
                'random_state': trial.suggest_int("random_state", 71, 71),
            }
        elif( args.regressor == "lightgbm" ):
            params = {
                'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
                'objective': 'regression',
                'metric': 'rmse',
                'num_class': 1,
                'num_leaves': trial.suggest_int("num_leaves", 10, 500),
                'learning_rate': trial.suggest_loguniform("learning_rate", 0.01, 0.01),
                'max_depth': trial.suggest_int("max_depth", 1, 5),
                'reg_alpha': trial.suggest_uniform("reg_alpha", 0, 100),
                'reg_lambda': trial.suggest_uniform("reg_lambda", 1, 5),
                'num_leaves': trial.suggest_int("num_leaves", 10, 500),
                'verbose' : 0,
            }
        elif( args.regressor == "catboost" ):
            params = {
#                'loss_function': trial.suggest_categorical('loss_function', ['RMSE']),
#                'eval_metric': trial.suggest_categorical('eval_metric', ['Logloss']),
                'iterations' : trial.suggest_int('iterations', 1000, 1000),                             # まず大きな数を設定しておく
                'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.01),               
                'depth' : trial.suggest_int('depth', 4, 10),                                       
                'random_strength' :trial.suggest_int('random_strength', 0, 100),                       
                'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00), 
                'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter']),
                'od_wait' :trial.suggest_int('od_wait', 100, 100),                                      # 最適な指標値に達した後、iterationを続ける数。
                'random_state': trial.suggest_int("random_state", 71, 71),
            }

        #--------------------------------------------
        # stratified k-fold CV での評価
        #--------------------------------------------
        y_preds_train = np.zeros((len(y_train),))

        # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
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
            if( args.regressor == "logistic" ):
                model = SklearnRegressor( model = LogisticRegression( penalty='l2', solver="sag", random_state=args.seed ), debug = args.debug )
            elif( args.regressor == "knn" ):
                model = SklearnRegressor( model = KNeighborsRegressor( n_neighbors = 3, p = 2, metric = 'minkowski' ), debug = args.debug )
            elif( args.regressor == "svm" ):
                model = SklearnRegressor( model = SVR( kernel = 'rbf', gamma = 0.1, C = 10.0 ), debug = args.debug )
            elif( args.regressor == "random_forest" ):
                model = SklearnRegressor( model = RandomForestRegressor( criterion = "mse", bootstrap = True, oob_score = True, n_estimators = 1000, n_jobs = -1, random_state = args.seed ), debug = args.debug )
            elif( args.regressor == "bagging" ):
                model = SklearnRegressor( model = BaggingRegressor( DecisionTreeRegressor(criterion = 'mse', max_depth = None, random_state = args.seed ) ), debug = args.debug )
            elif( args.regressor == "adaboost" ):
                model = SklearnRegressor( model = AdaBoostRegressor( DecisionTreeRegressor(criterion = 'mse', max_depth = None, random_state = args.seed ) ), debug = args.debug )
            elif( args.regressor == "xgboost" ):
                model = XGBoostRegressor( model = xgb.XGBRegressor( booster='gbtree', objective='reg:linear', eval_metric='rmse', learning_rate = 0.01 ), train_type = args.train_type, use_valid = True, debug = args.debug )
                model.load_params( "parames/xgboost_regressor_default.yml" )
            elif( args.regressor == "lightgbm" ):
                model = LightGBMRegressor( model = lgb.LGBMRegressor( objective='regression', metric='rmse' ), train_type = args.train_type, use_valid = True, debug = args.debug )
            elif( args.regressor == "catboost" ):
                model = CatBoostRegressor( model = catboost.CatBoostRegressor(), use_valid = True, debug = args.debug )

            # モデルのチューニングパラメータ設定
            model.set_params( **params )

            #--------------------
            # モデルの学習処理
            #--------------------
            model.fit(X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )

            #--------------------
            # モデルの推論処理
            #--------------------
            y_preds_train[valid_index] = model.predict(X_valid_fold)
        
        if( args.target_norm ):
            rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_preds_train) ) ) 
        else:
            rmse = np.sqrt( mean_squared_error(y_train, y_preds_train) )

        return rmse

    return objective

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="tuning_hyper_params", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="bike-sharing-demand")
    parser.add_argument("--regressor", choices=["logistic", "knn", "svm", "random_forest", "bagging", "adaboost", "xgboost", "lightgbm", "catboost"], default="catboost", help="チューニングするモデル")
    parser.add_argument('--train_type', choices=['train', 'fit'], default="fit", help="GDBTの学習タイプ")
    parser.add_argument("--n_splits", type=int, default=4, help="CV での学習用データセットの分割数")
    parser.add_argument("--n_splits_gs", type=int, default=2, help="ハイパーパラメーターチューニング時の CV での学習用データセットの分割数")
    parser.add_argument("--n_trials", type=int, default=10, help="Optuna での試行回数")
    parser.add_argument('--input_norm', action='store_true')
    parser.add_argument('--target_norm', action='store_false')
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # 実験名を自動的に変更
    if( args.exper_name == "tuning_hyper_params" ):
        args.exper_name = args.exper_name + "_" + args.regressor

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
    df_submission = pd.read_csv( os.path.join(args.dataset_dir, "sampleSubmission.csv" ) )
    if( args.debug ):
        print( "df_train.head() : \n", df_train.head() )
        print( "df_test.head() : \n", df_test.head() )
        print( "df_submission.head() : \n", df_submission.head() )
    
    #================================
    # 前処理
    #================================
    df_train, df_test = preprocessing( args, df_train, df_test )
    if( args.debug ):
        print( "df_train.head() : \n", df_train.head() )
        print( "df_test.head() : \n", df_test.head() )

    #==============================
    # 学習用データセットの分割
    #==============================
    # 学習用データセットとテスト用データセットの設定
    X_train = df_train.drop('count', axis = 1)
    X_test = df_test
    y_train = df_train['count']
    y_preds_train = np.zeros((len(y_train),))
    if( args.debug ):
        print( "len(X_train) : ", len(X_train) )
        print( "len(y_train) : ", len(y_train) )
        print( "len(y_preds_train) : ", len(y_preds_train) )

    #==============================
    # Optuna によるハイパーパラメーターのチューニング
    #==============================
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_wrapper(args, X_train,y_train), n_trials=args.n_trials)

    for key, value in study.best_params.items():
        print(key + ' : ' + str(value))

    # チューニングパラメータの外部ファイルへの保存
    df_hist = study.trials_dataframe()
    df_hist.to_csv( os.path.join( args.results_dir, args.exper_name, "tuning.csv") )

    with open( os.path.join( args.results_dir, args.exper_name, args.exper_name + ".yml"), 'w') as f:
        param = { 
            "model": {
                "name" : args.regressor,
                "model_params": study.best_params,
            },
        }
        #yaml.dump(param, f)
        json.dump(param, f, indent=4)

    #================================
    # 最良モデルでの学習 & 推論
    #================================    
    # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
    kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    y_preds_test = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train)):
        #--------------------
        # データセットの分割
        #--------------------
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        #--------------------
        # モデルの定義
        #--------------------
        if( args.regressor == "logistic" ):
            model = SklearnRegressor( model = LogisticRegression( penalty='l2', solver="sag", random_state=args.seed ), debug = args.debug )
        elif( args.regressor == "knn" ):
            model = SklearnRegressor( model = KNeighborsRegressor( n_neighbors = 3, p = 2, metric = 'minkowski' ), debug = args.debug )
        elif( args.regressor == "svm" ):
            model = SklearnRegressor( model = SVR( kernel = 'rbf', gamma = 0.1, C = 10.0 ), debug = args.debug )
        elif( args.regressor == "random_forest" ):
            model = SklearnRegressor( model = RandomForestRegressor( criterion = "mse", bootstrap = True, oob_score = True, n_estimators = 1000, n_jobs = -1, random_state = args.seed ), debug = args.debug )
        elif( args.regressor == "bagging" ):
            model = SklearnRegressor( model = BaggingRegressor( DecisionTreeRegressor(criterion = 'mse', max_depth = None, random_state = args.seed ) ), debug = args.debug )
        elif( args.regressor == "adaboost" ):
            model = SklearnRegressor( model = AdaBoostRegressor( DecisionTreeRegressor(criterion = 'mse', max_depth = None, random_state = args.seed ) ), debug = args.debug )
        elif( args.regressor == "xgboost" ):
            model = XGBoostRegressor( model = xgb.XGBRegressor( booster='gbtree', objective='reg:linear', eval_metric='rmse', learning_rate = 0.01 ), train_type = args.train_type, use_valid = True, debug = args.debug )
            model.load_params( "parames/xgboost_regressor_default.yml" )
        elif( args.regressor == "lightgbm" ):
            model = LightGBMRegressor( model = lgb.LGBMRegressor( objective='regression', metric='rmse' ), train_type = args.train_type, use_valid = True, debug = args.debug )
        elif( args.regressor == "catboost" ):
            model = CatBoostRegressor( model = catboost.CatBoostRegressor(), use_valid = True, debug = args.debug )

        # モデルのパラメータ設定
        model.set_params( **study.best_params )

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
    
    # k-fold CV で平均化
    y_preds_test = sum(y_preds_test) / len(y_preds_test)

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
    if( args.regressor in ["random_forest", "adaboost", "xgboost", "lightgbm", "catboost"] ):
        model.plot_importance( os.path.join(args.results_dir, args.exper_name, "feature_importances.png") )
        
    # 回帰対象
    _, axis = plt.subplots()
    sns.distplot(df_train['count'], label = "correct", ax=axis )
    sns.distplot(y_preds_train, label = "predict", ax=axis )
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "count.png"), dpi = 200, bbox_inches = 'tight' )

    #================================
    # Kaggle API での submit
    #================================
    # 提出用データに値を設定
    y_sub = list(map(int, y_preds_test))
    df_submission['count'] = y_sub
    df_submission.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)
    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
