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
from sklearn.model_selection import StratifiedKFold

# 機械学習モデル
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost

# 自作モジュール
from preprocessing import preprocessing, exploratory_data_analysis
from models import SklearnClassifier, XGBoostClassifier, LightGBMClassifier, CatBoostClassifier, KerasMLPClassifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="single_model", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="titanic")
    parser.add_argument("--classifier", choices=["logistic", "knn", "svm", "random_forest", "bagging", "adaboost", "xgboost", "lightgbm", "catboost", "mlp"], default="catboost", help="分類器モデルの種類")
    parser.add_argument("--params_file", type=str, default="")
    parser.add_argument('--train_type', choices=['train', 'fit'], default="fit", help="GDBTの学習タイプ")
    parser.add_argument("--n_splits", type=int, default=4, help="CV での学習用データセットの分割数")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--eda', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # 実験名を自動的に変更
    if( args.exper_name == "single_model" ):
        if( args.params_file == "" ):
            args.exper_name = args.exper_name + "_" + args.classifier
        else:
            args.exper_name = args.exper_name + "_" + args.classifier + "_" + args.params_file.split(".")[0]

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
    if( args.eda ):
        exploratory_data_analysis( args, df_train, df_test )

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
    # 学習用データセットとテスト用データセットの設定
    X_train = df_train.drop('Survived', axis = 1)
    X_test = df_test
    y_train = df_train['Survived']
    y_pred_train = np.zeros((len(y_train),))
    if( args.debug ):
        print( "len(X_train) : ", len(X_train) )
        print( "len(y_train) : ", len(y_train) )
        print( "len(y_pred_train) : ", len(y_pred_train) )

    #================================
    # モデルの学習 & 推論処理
    #================================    
    # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    y_preds_test = []
    k = 0
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
        # seed 値の固定
        np.random.seed(args.seed+k)
        random.seed(args.seed+k)
        
        #--------------------
        # データセットの分割
        #--------------------
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        #--------------------
        # モデルの定義
        #--------------------
        if( args.classifier == "logistic" ):
            model = SklearnClassifier( LogisticRegression( penalty='l2', solver="sag", random_state=args.seed ) )
        elif( args.classifier == "knn" ):
            model = SklearnClassifier( KNeighborsClassifier( n_neighbors = 3, p = 2, metric = 'minkowski' ) )
        elif( args.classifier == "svm" ):
            model = SklearnClassifier( SVC( kernel = 'rbf', gamma = 0.1, C = 10.0 ) )
        elif( args.classifier == "random_forest" ):
            model = SklearnClassifier( RandomForestClassifier( criterion = "gini", bootstrap = True, oob_score = True, n_estimators = 1000, n_jobs = -1, random_state = args.seed ) )
        elif( args.classifier == "bagging" ):
            model = SklearnClassifier( BaggingClassifier( DecisionTreeClassifier(criterion = 'entropy', max_depth = None, random_state = args.seed ) ) )
        elif( args.classifier == "adaboost" ):
            model = SklearnClassifier( AdaBoostClassifier( DecisionTreeClassifier(criterion = 'entropy', max_depth = None, random_state = args.seed ) ) )
        elif( args.classifier == "xgboost" ):
            model = XGBoostClassifier( model = xgb.XGBClassifier( booster='gbtree', objective='binary:logistic', eval_metric='logloss', learning_rate=0.01 ), train_type = args.train_type, use_valid = True, debug = args.debug )
            model.load_params( "parames/xgboost_classifier_default.yml" )
        elif( args.classifier == "lightgbm" ):
            model = LightGBMClassifier( model = lgb.LGBMClassifier( objective='binary', metric='binary_logloss' ), train_type = args.train_type, use_valid = True, debug = args.debug )
        elif( args.classifier == "catboost" ):
            model = CatBoostClassifier( model = catboost.CatBoostClassifier( loss_function="Logloss", random_seed = args.seed + k ), use_valid = True, debug = args.debug )
        elif( args.classifier == "mlp" ):
            model = KerasMLPClassifier( n_input_dim = len(X_train.columns), use_valid = True, debug = args.debug )

        # モデルのパラメータ設定
        if not( args.params_file == "" ):
            model.set_params( args.params_file )

        #--------------------
        # モデルの学習処理
        #--------------------
        model.fit(X_train_fold, y_train_fold, X_valid_fold, y_valid_fold)

        #--------------------
        # モデルの推論処理
        #--------------------
        y_pred_train[valid_index] = model.predict(X_valid_fold)
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

    # accuracy
    accuracy = (y_train == y_pred_train).sum()/len(y_pred_train)
    print( "accuracy [k-fold CV train-valid] : {:0.5f}".format(accuracy) )

    #================================
    # 可視化処理
    #================================
    # 重要特徴量
    if( args.classifier in ["random_forest", "adaboost", "xgboost", "lightgbm", "catboost"] ):
        model.plot_importance( os.path.join(args.results_dir, args.exper_name, "feature_importances.png") )

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
