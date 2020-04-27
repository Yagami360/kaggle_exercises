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
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import catboost

# 自作モジュール
from preprocessing import preprocessing
from models import SklearnClassifier, XGBoostClassifier, CatBoostClassifier, KerasMLPClassifier
from models import StackingEnsembleClassifier


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
    if( args.debug ):
        print( "len(X_train) : ", len(X_train) )
        print( "len(y_train) : ", len(y_train) )
        print( "len(X_test) : ", len(X_test) )

    #===========================================
    # モデルの学習 & 推論処理
    #===========================================
    #--------------------
    # モデル定義
    #--------------------
    logistic1 = SklearnClassifier( LogisticRegression() )
    logistic1.load_params( "parames/logstic_classifier_titanic.yml" )
    logistic2 = SklearnClassifier( LogisticRegression() )
    logistic2.load_params( "parames/logstic_classifier_titanic.yml" )
    logistic3 = SklearnClassifier( LogisticRegression() )
    logistic3.load_params( "parames/logstic_classifier_titanic.yml" )

    knn1 = SklearnClassifier( KNeighborsClassifier() )
    knn1.load_params( "parames/knn_classifier_titanic.yml" )

    svc1 = SklearnClassifier( SVC() )
    svc1.load_params( "parames/svm_classifier_titanic.yml" )

    forest1 = SklearnClassifier( RandomForestClassifier() )
    forest1.load_params( "parames/random_forest_classifier_titanic.yml" )

    bagging1 = SklearnClassifier( BaggingClassifier( DecisionTreeClassifier(criterion = 'entropy', max_depth = None, random_state = args.seed ) ) )
    bagging1.load_params( "parames/tuning_hyper_params_bagging.yml" )

    adaboost1 = SklearnClassifier( AdaBoostClassifier( DecisionTreeClassifier(criterion = 'entropy', max_depth = None, random_state = args.seed ) ) )
    adaboost1.load_params( "parames/tuning_hyper_params_adaboost.yml" )

    xgboost1 = XGBoostClassifier( model = xgb.XGBClassifier(), train_type = "fit", use_valid = True, debug = args.debug )
    xgboost1.load_params( "parames/xgboost_classifier_titanic.yml" )
    xgboost2 = XGBoostClassifier( model = xgb.XGBClassifier(), train_type = "fit", use_valid = True, debug = args.debug )
    xgboost2.load_params( "parames/xgboost_classifier_titanic2.yml" )

    catboost1 = CatBoostClassifier( model = catboost.CatBoostClassifier(), use_valid = True, debug = args.debug )
    catboost1.load_params( "parames/tuning_hyper_params_catboost.yml" )

    dnn1 = KerasMLPClassifier( n_input_dim = len(X_train.columns), use_valid = True, debug = args.debug )
    dnn2 = KerasMLPClassifier( n_input_dim = 8, use_valid = True, debug = args.debug )

    # アンサンブルモデル（２段）
    """
    model = StackingEnsembleClassifier(
        classifiers  = [ knn1, logistic1, svc1, forest1, xgboost1, dnn1 ],
        final_classifiers = logistic2,
        n_splits = args.n_splits,
        seed = args.seed,
    )
    """

    # アンサンブルモデル（３段）
    model = StackingEnsembleClassifier(
        classifiers  = [ knn1, logistic1, svc1, forest1, bagging1, adaboost1, xgboost1, dnn1 ],
        second_classifiers  = [ logistic2, xgboost2, catboost1 ],
        final_classifiers = logistic3,
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
    y_preds_train = model.y_preds_train
    y_preds_test = model.y_preds_test

    # accuracy
    accuracy = (y_train == y_preds_train).sum()/len(y_preds_train)
    print( "ccuracy [k-fold CV train vs valid] : {:0.5f}".format(accuracy) )

    #================================
    # 可視化処理
    #================================
    #------------------
    # 分類対象の分布図
    #------------------
    fig = plt.figure()
    axis = fig.add_subplot(111)
    sns.distplot(df_train['Survived'], label='correct' )
    sns.distplot(model.y_preds_train, label='predict' )
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "Survived.png"), dpi = 300, bbox_inches = 'tight' )

    #------------------
    # 損失関数値
    #------------------
    """
    # DNN    
    fig = plt.figure()
    axis = fig.add_subplot(111)
    for i, evals_result in enumerate(dnn1.evals_results):
        print( evals_result.keys() )
        print( evals_result['accuracy'][0:10] )

        #axis.plot(evals_result['train'][xgboost1.model_params["eval_metric"]], label='train / k={}'.format(i))
        #axis.plot(evals_result['valid'][xgboost1.model_params["eval_metric"]], label='valid / k={}'.format(i))

    plt.xlabel('epoches')
    plt.set_title( "DNN" )
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "losses_dnn.png"), dpi = 300, bbox_inches = 'tight' )
    """

    # XGBoost
    """
    fig = plt.figure()
    axis = fig.add_subplot(111)
    for i, evals_result in enumerate(xgboost1.evals_results):
        axis.plot(evals_result['train'][xgboost1.model_params["eval_metric"]], label='train / k={}'.format(i))

    for i, evals_result in enumerate(xgboost1.evals_results):
        axis.plot(evals_result['valid'][xgboost1.model_params["eval_metric"]], label='valid / k={}'.format(i))

    axis.set_title( "xgboost" )
    plt.xlabel('iters')
    plt.ylabel(xgboost1.model_params["eval_metric"])
    plt.xlim( [0,xgboost1.train_params["num_boost_round"]+1] )
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "losses_xgboost.png"), dpi = 300, bbox_inches = 'tight' )
    """
    
    #------------------
    # 重要特徴量
    #------------------
    _, ax = plt.subplots(figsize=(8, 4))
    xgb.plot_importance(
        xgboost1.model,
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
