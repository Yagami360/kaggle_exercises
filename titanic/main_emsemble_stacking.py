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
from sklearn.svm import SVC                             # 
from sklearn.ensemble import BaggingClassifier          # バギング
from sklearn.ensemble import AdaBoostClassifier         # AdaBoost
from sklearn.ensemble import RandomForestClassifier     # 
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from models import EnsembleStackingModelClassifier


if __name__ == '__main__':
    """
    stratified k-fold cross validation で学習用データセットを分割して学習＆評価
    学習モデルは xgboost
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="emsemble_stacking", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="titanic")
    parser.add_argument("--params_file", type=str, default="parames/xgboost_classifier_titanic.yml")
    parser.add_argument("--n_splits", type=int, default=4, help="k-fold CV での学習用データセットの分割数")
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
    ds_submission = pd.read_csv( os.path.join(args.dataset_dir, "gender_submission.csv" ) )
    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )
        print( "ds_submission.head() : \n", ds_submission.head() )
    
    #================================
    # 前処理
    #================================
    # 無用なデータを除外
    ds_train.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
    ds_test.drop(['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # データを数量化
    ds_train['Sex'].replace(['male','female'], [0, 1], inplace=True)
    ds_test['Sex'].replace(['male','female'], [0, 1], inplace=True)

    ds_train['Embarked'].fillna(('S'), inplace=True)
    ds_train['Embarked'] = ds_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
    ds_test['Embarked'].fillna(('S'), inplace=True)
    ds_test['Embarked'] = ds_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # NAN 値を補完
    ds_train['Fare'].fillna(np.mean(ds_train['Fare']), inplace=True)
    ds_test['Fare'].fillna(np.mean(ds_test['Fare']), inplace=True)

    ds_train['Age'].fillna(np.mean(ds_train['Age']), inplace=True)
    ds_test['Age'].fillna(np.mean(ds_test['Age']), inplace=True)

    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )

    #===========================================
    # 学習用データセットとテスト用データセットの設定
    #===========================================
    X_train = ds_train.drop('Survived', axis = 1)
    X_test = ds_test
    y_train = ds_train['Survived']
    if( args.debug ):
        print( "len(X_train) : ", len(X_train) )
        print( "len(y_train) : ", len(y_train) )
        print( "len(X_test) : ", len(X_test) )

    #===========================================
    # モデルの学習 & 推論処理
    #===========================================
    # モデルのパラメータの読み込み
    with open( args.params_file ) as f:
        params = yaml.safe_load(f)
        xgboost_params = params["model"]["model_params"]
        xgboost_train_params = params["model"]["train_params"]
        if( args.debug ):
            print( "params :\n", params )

    #--------------------
    # モデル定義
    #--------------------
    logistic = LogisticRegression( penalty='l2', solver="sag", random_state=args.seed )
    kNN = KNeighborsClassifier( n_neighbors = 5, p = 2, metric = 'minkowski', n_jobs = -1 )
    svm = SVC( kernel = 'rbf', gamma = 10.0, C = 0.1, random_state = args.seed, probability = True )
    forest = RandomForestClassifier( criterion = "gini", bootstrap = True, n_estimators = 1001, n_jobs = -1, random_state = args.seed, oob_score = True )

    xgboost = XGBClassifier(
        booster = xgboost_params['booster'],
        objective = xgboost_params['objective'],
        learning_rate = xgboost_params['learning_rate'],
        n_estimators = xgboost_params['n_estimators'],
        max_depth = xgboost_params['max_depth'],
        min_child_weight = xgboost_params['min_child_weight'],
        subsample = xgboost_params['subsample'],
        colsample_bytree = xgboost_params['colsample_bytree'],
        gamma = xgboost_params['gamma'],
        alpha = xgboost_params['alpha'],
        reg_lambda = xgboost_params['reg_lambda'],
        random_state = xgboost_params['random_state']
    )

    model = EnsembleStackingModelClassifier(
        classifiers  = [ xgboost, kNN, svm, forest ],
        final_classifiers = logistic,
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
    y_preds = model.predict(X_test)

    #================================
    # Kaggle API での submit
    #================================
    # 提出用データに値を設定
    y_sub = list(map(int, y_preds))
    ds_submission['Survived'] = y_sub
    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)

    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
