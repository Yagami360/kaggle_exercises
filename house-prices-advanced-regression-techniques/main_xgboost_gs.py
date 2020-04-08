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
import xgboost as xgb
import optuna


def objective_wrapper(args, X_train, y_train):
    """
    objective に trial 以外の引数を指定可能にするためのラッパーメソッド
    """
    def objective(trial):
        #--------------------------------------------
        # ベイズ最適化でのチューニングパイパーパラメーター
        #--------------------------------------------
        params = {
            'booster': trial.suggest_categorical('booster', ['gbtree']),
            'objective': trial.suggest_categorical('objective', ['reg:linear']),
            "learning_rate" : trial.suggest_loguniform("learning_rate", 1e-8, 0.01),                      # ハイパーパラメーターのチューニング時は固定  
            "n_estimators" : trial.suggest_int("n_estimators", 950, 1050),                                # 
            'max_depth': trial.suggest_int("max_depth", 3, 9),                                            # 3 ~ 9 : 一様分布に従う。1刻み
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.1, 10.0),                  # 0.1 ~ 10.0 : 対数が一様分布に従う
            'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 0.95, 0.05),                    # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 0.95, 0.05),      # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
            'gamma': trial.suggest_loguniform("gamma", 1e-8, 1.0),                                        # 1e-8 ~ 1.0 : 対数が一様分布に従う
            'alpha': trial.suggest_float("alpha", 0.0, 0.0),                                              # デフォルト値としておく。余裕があれば変更
            'reg_lambda': trial.suggest_float("reg_lambda", 1.0, 1.0),                                    # デフォルト値としておく。余裕があれば変更
            'random_state': trial.suggest_int("random_state", 71, 71),
            'eval_metric': trial.suggest_categorical("eval_metric", ['rmse']),                            # 平均2乗平方根誤差
            "num_boost_round": 1000,            # 試行回数
            "early_stopping_rounds": 10,        # early stopping を行う繰り返し回数
        }

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

            # XGBoost 用データセットに変換
            if( args.train_type == "train" ):
                X_train_fold_dmat = xgb.DMatrix(X_train_fold, label=y_train_fold)
                X_valid_fold_dmat = xgb.DMatrix(X_valid_fold, label=y_valid_fold)
                X_test_dmat = xgb.DMatrix(X_test, label=y_train)

            #--------------------
            # モデルの定義
            #--------------------
            if( args.train_type == "fit" ):
                model = xgb.XGBRegressor(
                    booster = params['booster'],
                    objective = params['objective'],
                    learning_rate = params['learning_rate'],
                    n_estimators = params['n_estimators'],
                    max_depth = params['max_depth'],
                    min_child_weight = params['min_child_weight'],
                    subsample = params['subsample'],
                    colsample_bytree = params['colsample_bytree'],
                    gamma = params['gamma'],
                    alpha = params['alpha'],
                    reg_lambda = params['reg_lambda'],
                    random_state = params['random_state']                    
                )

            #--------------------
            # モデルの学習処理
            #--------------------
            if( args.train_type == "train" ):
                model = xgb.train(
                    params, X_train_fold_dmat, 
                    num_boost_round = params["num_boost_round"],
                    early_stopping_rounds = params["early_stopping_rounds"],
                    evals = [ (X_train_fold_dmat, 'train'), (X_valid_fold_dmat, 'val') ],
                )
            else:
                model.fit(X_train_fold, y_train_fold)

            #--------------------
            # モデルの推論処理
            #--------------------
            if( args.train_type == "train" ):
                y_pred_val[valid_index] = model.predict(X_valid_fold_dmat)
            else:
                y_pred_val[valid_index] = model.predict(X_valid_fold)

        if( args.target_norm ):
            rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_pred_val) ) ) 
        else:
            rmse = np.sqrt( mean_squared_error(y_train, y_pred_val) )

        return rmse

    return objective

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="datasets/input")
    parser.add_argument("--out_dir", type=str, default="datasets/output")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--submit_message", type=str, default="From Kaggle API Python Script")
    parser.add_argument("--competition_id", type=str, default="house-prices-advanced-regression-techniques")
    parser.add_argument('--train_type', choices=['train', 'fit'], default="train", help="XGBoost の学習タイプ")
    parser.add_argument("--n_splits", type=int, default=4, help="CV での学習用データセットの分割数")
    parser.add_argument("--n_splits_gs", type=int, default=4, help="ハイパーパラメーターチューニング時の CV での学習用データセットの分割数")
    parser.add_argument("--n_trials", type=int, default=100, help="Optuna での試行回数")
    parser.add_argument('--target_norm', action='store_true')
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # 警告非表示
    warnings.simplefilter('ignore', DeprecationWarning)

    # seed 値の固定
    np.random.seed(args.seed)
    random.seed(args.seed)

    #================================
    # データセットの読み込み
    #================================
    ds_train = pd.read_csv( os.path.join(args.in_dir, "train.csv" ) )
    ds_test = pd.read_csv( os.path.join(args.in_dir, "test.csv" ) )
    ds_sample_submission = pd.read_csv( os.path.join(args.in_dir, "sample_submission.csv" ) )
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
    ds_train.to_csv( os.path.join(args.out_dir, "train_preprocessed.csv"), index=True)
    ds_test.to_csv( os.path.join(args.out_dir, "test_preprocessed.csv"), index=True)
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
    evals_results = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train)):
        #--------------------
        # データセットの分割
        #--------------------
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        # XGBoost 用データセットに変換
        if( args.train_type == "train" ):
            X_train_fold_dmat = xgb.DMatrix(X_train_fold, label=y_train_fold)
            X_valid_fold_dmat = xgb.DMatrix(X_valid_fold, label=y_valid_fold)
            X_test_dmat = xgb.DMatrix(X_test, label=y_train)

        #--------------------
        # 回帰モデル定義
        #--------------------
        if( args.train_type == "fit" ):
            model = xgb.XGBRegressor(
                booster = study.best_params['booster'],
                objective = study.best_params['objective'],
                learning_rate = study.best_params['learning_rate'],
                n_estimators = study.best_params['n_estimators'],
                max_depth = study.best_params['max_depth'],
                min_child_weight = study.best_params['min_child_weight'],
                subsample = study.best_params['subsample'],
                colsample_bytree = study.best_params['colsample_bytree'],
                gamma = study.best_params['gamma'],
                alpha = study.best_params['alpha'],
                reg_lambda = study.best_params['reg_lambda'],
                random_state = study.best_params['random_state']
            )

        #--------------------
        # モデルの学習処理
        #--------------------
        if( args.train_type == "train" ):
            evals_result = {}
            model = xgb.train(
                study.best_params, X_train_fold_dmat, 
                num_boost_round = study.best_params["num_boost_round"],
                early_stopping_rounds = study.best_params["early_stopping_rounds"],
                evals = [ (X_train_fold_dmat, 'train'), (X_valid_fold_dmat, 'val') ],
                evals_result = evals_result
            )
            evals_results.append(evals_result)
        else:
            model.fit(X_train, y_train)

        #--------------------
        # モデルの推論処理
        #--------------------
        if( args.train_type == "train" ):
            y_pred_test = model.predict(X_test_dmat)
        else:
            y_pred_test = model.predict(X_test)

        y_preds.append(y_pred_test)

        if( args.train_type == "train" ):
            y_pred_val[valid_index] = model.predict(X_valid_fold_dmat)
        else:
            y_pred_val[valid_index] = model.predict(X_valid_fold)

        #print( "[{}] len(y_pred_fold) : {}".format(fold_id, len(y_pred_val)) )
    
    # 正解データとの平均2乗平方根誤差で評価
    if( args.target_norm ):
        rmse = np.sqrt( mean_squared_error( np.exp(y_train), np.exp(y_pred_val) ) ) 
    else:
        rmse = np.sqrt( mean_squared_error(y_train, y_pred_val) )

    print( "RMSE [val] : {:0.5f}".format(rmse) )

    # 重要特徴量
    if( args.train_type == "train" ):
        print( "[Feature Importances] : \n", model.get_fscore() )
    else:
        print( "[Feature Importances]" )
        for i, col in enumerate(X_train.columns):
            print( "{} : {:.4f}".format( col, model.feature_importances_[i] ) )

    # loss 値
    """
    if( args.train_type == "train" ):
        print( "[train loss] eval_result :", eval_result["train"][params_xgboost["eval_metric"]] )
        print( "[val loss] eval_result :", eval_result["val"][params_xgboost["eval_metric"]] )
    """

    #================================
    # 可視化処理
    #================================
    # 回帰対象
    sns.distplot(ds_train['SalePrice'] )
    sns.distplot(y_pred_val)
    if( args.target_norm ):
        plt.savefig( os.path.join(args.out_dir, "SalePrice_w_norm.png"), dpi = 300, bbox_inches = 'tight' )
    else:
        plt.savefig( os.path.join(args.out_dir, "SalePrice_wo_norm.png"), dpi = 300, bbox_inches = 'tight' )

    # loss
    if( args.train_type == "train" ):
        for i, evals_result in enumerate(evals_results):
            plt.plot(evals_result['train'][study.best_params["eval_metric"]], label='train / k={}'.format(i))
            plt.plot(evals_result['val'][study.best_params["eval_metric"]], label='val / k={}'.format(i))

        plt.xlabel('iters')
        plt.ylabel(study.best_params["eval_metric"])
        plt.xlim( [0,study.best_params["num_boost_round"]+1] )
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig( os.path.join(args.out_dir, "losees.png"), dpi = 300, bbox_inches = 'tight' )

    # 重要特徴量
    _, ax = plt.subplots(figsize=(8, 16))
    xgb.plot_importance(
        model,
        ax = ax,
        importance_type = 'gain',
        show_values = False
    )
    plt.tight_layout()
    plt.savefig( os.path.join(args.out_dir, "feature_importances.png"), dpi = 300, bbox_inches = 'tight' )

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

    sub.to_csv( os.path.join(args.out_dir, args.submit_file), index=False)

    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.out_dir, args.submit_file), args.submit_message, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )