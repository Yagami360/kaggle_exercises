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

from sklearn.model_selection import StratifiedKFold

# 機械学習モデル
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC                             # 
from sklearn.ensemble import BaggingClassifier          # バギング
from sklearn.ensemble import AdaBoostClassifier         # AdaBoost
from sklearn.ensemble import RandomForestClassifier     # 
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import StackingClassifier

# 自作クラス
from models import SklearnClassifier, XGBoostClassifier, KerasDNNClassifier, KerasResNetClassifier


if __name__ == '__main__':
    """
    stratified k-fold cross validation で学習用データセットを分割して学習＆評価
    学習モデルは xgboost
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="emsemble_stacking_sklearn", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="titanic")
    parser.add_argument("--params_file", type=str, default="parames/xgboost_classifier_titanic.yml")
    parser.add_argument("--n_splits", type=int, default=4, help="k-fold CV での学習用データセットの分割数")
    parser.add_argument('--output_type', choices=['fixed', 'proba'], default="fixed", help="出力形式（確定値 or 確率値）")
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
        model_params = params["model"]["model_params"]
        model_train_params = params["model"]["train_params"]
        if( args.debug ):
            print( "params :\n", params )

    xgboost1 = xgb.XGBClassifier(
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

    xgboost2 = xgb.XGBClassifier(
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
    # モデル定義
    #--------------------
    # アンサンブルモデル（２段）
    """
    model = StackingClassifier(
        estimators = {
            ("logistic", LogisticRegression( penalty='l2', solver="sag", random_state=args.seed )),
            ('knn', KNeighborsClassifier( n_neighbors = 5, p = 2, metric = 'minkowski', n_jobs = -1 )),
            ('svm', SVC( kernel = 'rbf', gamma = 10.0, C = 0.1, random_state = args.seed, probability = True ) ),
            ('forest', RandomForestClassifier( criterion = "gini", bootstrap = True, n_estimators = 1001, n_jobs = -1, random_state = args.seed, oob_score = True ) ),
            ('xgboost', xgboost1),
        }, 
        final_estimator = LogisticRegression( penalty='l2', solver="sag", random_state=args.seed ),
    )
    """
    
    # アンサンブルモデル（３段）
    model = StackingClassifier(
        estimators = {
            ("logistic", LogisticRegression( penalty='l2', solver="sag", random_state=args.seed )),
            ('knn', KNeighborsClassifier( n_neighbors = 5, p = 2, metric = 'minkowski', n_jobs = -1 )),
            ('svm', SVC( kernel = 'rbf', gamma = 10.0, C = 0.1, random_state = args.seed, probability = True ) ),
            ('forest', RandomForestClassifier( criterion = "gini", bootstrap = True, n_estimators = 1001, n_jobs = -1, random_state = args.seed, oob_score = True ) ),
            ('xgboost', xgboost1),
        }, 
        final_estimator = StackingClassifier(
            estimators = {
                ("logistic", LogisticRegression( penalty='l2', solver="sag", random_state=args.seed )),
                ('forest', RandomForestClassifier( criterion = "gini", bootstrap = True, n_estimators = 1001, n_jobs = -1, random_state = args.seed, oob_score = True ) ),
                ('xgboost', xgboost2),
            },
            final_estimator = LogisticRegression( penalty='l2', solver="sag", random_state=args.seed ),
            cv = args.n_splits,
            verbose = 1,
            n_jobs = -1,
        ),
        cv = args.n_splits,
        verbose = 1,
        n_jobs = -1,
    )

    #--------------------
    # モデルの学習処理
    #--------------------
    model.fit(X_train, y_train)

    #--------------------
    # モデルの推論処理
    #--------------------
    if( args.output_type == "fixed" ):
        y_preds_train = model.predict(X_train)
        y_preds_test = model.predict(X_test)
    else:
        y_preds_train = model.predict_proba(X_train)
        y_preds_test = model.predict_proba(X_test)

    print( "y_preds_train.shape: ", y_preds_train.shape )
    print( "y_preds_test.shape: ", y_preds_test.shape )

    accuracy = (y_train == y_preds_train).sum()/len(y_preds_train)
    print( "accuracy [k-fold CV train vs valid] : {:0.5f}".format(accuracy) )

    #================================
    # 可視化処理
    #================================
    # 分類対象の分布図
    fig = plt.figure()
    axis = fig.add_subplot(111)
    sns.distplot(ds_train['Survived'], label='correct' )
    sns.distplot(y_preds_train, label='predict' )
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "Survived.png"), dpi = 300, bbox_inches = 'tight' )

    """
    # loss
    fig = plt.figure()
    axis = fig.add_subplot(111)
    for i, evals_result in enumerate(xgboost.evals_results):
        axis.plot(evals_result['train'][xgboost.train_params["eval_metric"]], label='train / k={}'.format(i))
        axis.plot(evals_result['valid'][xgboost.train_params["eval_metric"]], label='valid / k={}'.format(i))

    plt.xlabel('iters')
    plt.ylabel(xgboost.train_params["eval_metric"])
    plt.xlim( [0,xgboost.train_params["num_boost_round"]+1] )
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "losses_xgboost.png"), dpi = 300, bbox_inches = 'tight' )

    # 重要特徴量
    _, ax = plt.subplots(figsize=(8, 4))
    xgb.plot_importance(
        xgboost.model,
        ax = ax,
        importance_type = 'gain',
        show_values = False
    )
    plt.tight_layout()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "feature_importances.png"), dpi = 300, bbox_inches = 'tight' )
    """
    #================================
    # Kaggle API での submit
    #================================
    # 提出用データに値を設定
    y_sub = list(map(int, y_preds_test))
    ds_submission['Survived'] = y_sub
    ds_submission.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)

    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
