# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
import yaml
import random
import warnings
from kaggle.api.kaggle_api_extended import KaggleApi

from sklearn.model_selection import StratifiedKFold

# models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost

# 自作モジュール
from preprocessing import preprocessing
from models import SklearnClassifier, XGBoostClassifier, CatBoostClassifier, KerasMLPClassifier
from models import WeightAverageEnsembleClassifier


if __name__ == '__main__':
    """
    stratified k-fold cross validation で学習用データセットを分割して学習＆評価
    学習モデルは xgboost
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="emsemble_average", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="titanic")
    parser.add_argument("--n_splits", type=int, default=4, help="k-fold CV での学習用データセットの分割数")
    parser.add_argument("--vote_method", choices=["majority_vote", "probability_vote"], default="probability_vote", help="固定値で平均化 or 確率値で平均化")
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

    #===========================================
    # 学習用データセットとテスト用データセットの設定
    #===========================================
    X_train = df_train.drop('Survived', axis = 1)
    X_test = df_test
    y_train = df_train['Survived']
    y_pred_val = np.zeros((len(y_train),))
    if( args.debug ):
        print( "len(X_train) : ", len(X_train) )
        print( "len(y_train) : ", len(y_train) )
        print( "len(y_pred_val) : ", len(y_pred_val) )

    #===========================================
    # k-fold CV による学習 & 推論処理
    #===========================================
    # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    y_preds_test = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
        #--------------------
        # データセットの分割
        #--------------------
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        #--------------------
        # モデル定義
        #--------------------
        logistic1 = SklearnClassifier( LogisticRegression() )
        logistic1.load_params( "parames/logstic_classifier_titanic.yml" )

        knn1 = SklearnClassifier( KNeighborsClassifier() )
        knn1.load_params( "parames/knn_classifier_titanic.yml" )

        svm1 = SklearnClassifier( SVC(probability=True) )
        svm1.load_params( "parames/svm_classifier_titanic.yml" )

        forest1 = SklearnClassifier( RandomForestClassifier() )
        forest1.load_params( "parames/random_forest_classifier_titanic.yml" )

        bagging1 = SklearnClassifier( BaggingClassifier( DecisionTreeClassifier(criterion = 'entropy', max_depth = None, random_state = args.seed ) ) )
        #bagging1.load_params( "parames/bagging_classifier_titanic.yml" )

        adaboost1 = SklearnClassifier( AdaBoostClassifier( DecisionTreeClassifier(criterion = 'entropy', max_depth = None, random_state = args.seed ) ) )
        #adaboost1.load_params( "parames/adaboostclassifier_titanic.yml" )

        xgboost1 = XGBoostClassifier( model = xgb.XGBClassifier, train_type = "fit", use_valid = True, debug = args.debug )
        xgboost1.load_params( "parames/xgboost_classifier_titanic.yml" )

        catboost1 = CatBoostClassifier( model = catboost.CatBoostClassifier(), use_valid = True, debug = args.debug )
        #catboost1.load_params( "parames/catboost_classifier_titanic.yml" )

        dnn1 = KerasMLPClassifier( n_input_dim = len(X_train.columns), use_valid = True, debug = args.debug )

        # アンサンブルモデル
        """
        model = WeightAverageEnsembleClassifier(
            classifiers  = [ xgboost, kNN, svm, forest, bagging, ada ],
            weights = [0.75, 0.25, 0.0, 0.25, 0.25, 0.25 ],
            vote_method = args.vote_method,
        )
        """
        model = WeightAverageEnsembleClassifier(
            classifiers  = [ logistic1, knn1, svm1, forest1, bagging1, adaboost1, xgboost1, catboost1, dnn1 ],
            weights = [ 0.01, 0.01, 0.02, 0.05, 0.15, 0.15, 0.3, 0.4, 0.05 ],
            vote_method = args.vote_method,
        )
        
        #--------------------
        # モデルの学習処理
        #--------------------
        model.fit(X_train_fold, y_train_fold, X_valid_fold, y_valid_fold )

        #--------------------
        # モデルの推論処理
        #--------------------
        y_pred_test = model.predict(X_test)
        y_preds_test.append(y_pred_test)

        y_pred_val[valid_index] = model.predict(X_valid_fold)
    
    # k-fold CV で平均化
    y_preds_test = sum(y_preds_test) / len(y_preds_test)

    # accuracy
    accuracy = (y_train == y_pred_val).sum()/len(y_pred_val)
    print( "accuracy [k-fold CV train-valid] : {:0.5f}".format(accuracy) )

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
