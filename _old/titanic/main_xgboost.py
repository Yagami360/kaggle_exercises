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
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# 自作モジュール
from preprocessing import preprocessing

if __name__ == '__main__':
    """
    stratified k-fold cross validation で学習用データセットを分割して学習＆評価
    学習モデルは xgboost
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="xgboost_stratified_kfoldCV", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="titanic")
    parser.add_argument("--params_file", type=str, default="parames/xgboost_classifier_default.yml")
    parser.add_argument('--train_type', choices=['train', 'fit'], default="fit", help="XGBoost の学習タイプ")
    parser.add_argument("--n_splits", type=int, default=4, help="CV での学習用データセットの分割数")
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
    df_submission = pd.read_csv( os.path.join(args.dataset_dir, "gender_submission.csv" ) )
    if( args.debug ):
        print( "df_train.head() : \n", df_train.head() )
        print( "df_test.head() : \n", df_test.head() )
        print( "df_submission.head() : \n", df_submission.head() )
    
    #================================
    # 前処理
    #================================
    df_train, df_test = preprocessing( df_train, df_test, debug = args.debug )
    if( args.debug ):
        print( "df_train.head() : \n", df_train.head() )
        print( "df_test.head() : \n", df_test.head() )

    # 前処理後のデータセットを外部ファイルに保存
    df_train.to_csv( os.path.join(args.results_dir, args.exper_name, "train_preprocessed.csv"), index=True)
    df_test.to_csv( os.path.join(args.results_dir, args.exper_name, "test_preprocessed.csv"), index=True)

    #===========================================
    # 学習用データセットとテスト用データセットの設定
    #===========================================
    X_train = df_train.drop('Survived', axis = 1)
    X_test = df_test
    y_train = df_train['Survived']
    y_pred_train = np.zeros((len(y_train),))
    if( args.debug ):
        print( "len(X_train) : ", len(X_train) )
        print( "len(y_train) : ", len(y_train) )
        print( "len(y_pred_train) : ", len(y_pred_train) )

    #===========================================
    # k-fold CV による処理
    #===========================================
    # モデルのパラメータの読み込み
    with open( args.params_file ) as f:
        params = yaml.safe_load(f)
        model_params = params["model"]["model_params"]
        model_train_params = params["model"]["train_params"]
        if( args.debug ):
            print( "params :\n", params )

    # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    y_preds_test = []
    evals_results = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
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
        # モデル定義
        #--------------------
        if( args.train_type == "fit" ):
            model = xgb.XGBClassifier(
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

        #--------------------
        # モデルの学習処理
        #--------------------
        if( args.train_type == "train" ):
            evals_result = {}
            model = xgb.train(
                model_params, X_train_fold_dmat, 
                num_boost_round = model_train_params["num_boost_round"],
                early_stopping_rounds = model_train_params["early_stopping_rounds"],
                evals = [ (X_train_fold_dmat, 'train'), (X_valid_fold_dmat, 'valid') ],
                verbose_eval = model_train_params["num_boost_round"] // 50,
                evals_result = evals_result
            )
            evals_results.append(evals_result)
        else:
            model.fit(X_train, y_train)

        #--------------------
        # モデルの推論処理
        #--------------------
        # test
        if( args.train_type == "train" ):
            y_pred_test = model.predict(X_test_dmat)
            y_pred_test = np.where(y_pred_test > 0.5, 1, 0)
        else:
            y_pred_test = model.predict(X_test)

        y_preds_test.append(y_pred_test)

        # train - valid
        if( args.train_type == "train" ):
            y_pred_train[valid_index] = model.predict(X_valid_fold_dmat)
            y_pred_train = np.where(y_pred_train > 0.5, 1, 0)
        else:
            y_pred_train[valid_index] = model.predict(X_valid_fold)
    
    # k-fold CV で平均化
    y_preds_test = sum(y_preds_test) / len(y_preds_test)

    # accuracy
    accuracy = (y_train == y_pred_train).sum()/len(y_pred_train)
    print( "accuracy [k-fold CV train-valid] : {:0.5f}".format(accuracy) )

    #================================
    # 可視化処理
    #================================
    #------------------
    # 損失関数
    #------------------
    if( args.train_type == "train" ):
        fig = plt.figure()
        axis = fig.add_subplot(111)
        for i, evals_result in enumerate(evals_results):
            axis.plot(evals_result['train'][model_params["eval_metric"]], label='train / k={}'.format(i))
        for i, evals_result in enumerate(evals_results):
            axis.plot(evals_result['valid'][model_params["eval_metric"]], label='valid / k={}'.format(i))

        plt.xlabel('iters')
        plt.ylabel(model_params["eval_metric"])
        plt.xlim( [0,model_train_params["num_boost_round"]+1] )
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "losees.png"), dpi = 300, bbox_inches = 'tight' )

    #------------------
    # 重要特徴量
    #------------------
    _, ax = plt.subplots(figsize=(8, 4))
    xgb.plot_importance(
        model,
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
