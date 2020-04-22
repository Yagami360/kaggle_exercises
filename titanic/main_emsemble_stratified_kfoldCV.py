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
from xgboost import XGBClassifier

from models import EnsembleBlendingClassifier


if __name__ == '__main__':
    """
    stratified k-fold cross validation で学習用データセットを分割して学習＆評価
    学習モデルは xgboost
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="emsemble_stratified_kfoldCV", help="実験名")
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
    ds_gender_submission = pd.read_csv( os.path.join(args.dataset_dir, "gender_submission.csv" ) )
    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )
        print( "ds_gender_submission.head() : \n", ds_gender_submission.head() )
    
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
    y_pred_val = np.zeros((len(y_train),))
    if( args.debug ):
        print( "len(X_train) : ", len(X_train) )
        print( "len(y_train) : ", len(y_train) )
        print( "len(y_pred_val) : ", len(y_pred_val) )

    #===========================================
    # k-fold CV による学習 & 推論処理
    #===========================================
    # モデルのパラメータの読み込み
    with open( args.params_file ) as f:
        params = yaml.safe_load(f)
        xgboost_params = params["model"]["model_params"]
        xgboost_train_params = params["model"]["train_params"]
        if( args.debug ):
            print( "params :\n", params )

    # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    y_preds = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
        #--------------------
        # データセットの分割
        #--------------------
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        #--------------------
        # モデル定義
        #--------------------
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

        kNN = KNeighborsClassifier(
                n_neighbors = 5,
                p = 2,
                metric = 'minkowski'
            )

        svm = SVC( 
                kernel = 'rbf',     # rbf : RFBカーネルでのカーネルトリックを指定
                gamma = 10.0,       # RFBカーネル関数のγ値
                C = 0.1,            # C-SVM の C 値
                random_state = args.seed,   #
                probability = True  # 学習後の predict_proba method による予想確率を有効にする
        )

        forest = RandomForestClassifier(
                    criterion = "gini",     # 不純度関数 [purity]
                    bootstrap = True,       # 決定木の構築に、ブートストラップサンプルを使用するか否か（default:True）
                    n_estimators = 1001,    # 弱識別器（決定木）の数
                    n_jobs = -1,            # The number of jobs to run in parallel for both fit and predict ( -1 : 全てのCPUコアで並列計算)
                    random_state = args.seed,       #
                    oob_score = True        # Whether to use out-of-bag samples to estimate the generalization accuracy.(default=False)
                )

        decition_tree = DecisionTreeClassifier(
                            criterion = 'entropy',       # 不純度として, 交差エントロピー
                            max_depth = None,            # None : If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.(default=None)
                            random_state = args.seed
                        )

        bagging = BaggingClassifier(
                    base_estimator = decition_tree,   # 弱識別器をして決定木を設定
                    n_estimators = 1001,              # バギングを構成する弱識別器の数
                    max_samples = 1.0,                # The number of samples to draw from X to train each base estimator.
                                                      # If float, then draw max_samples * X.shape[0] samples.
                                                      # base_estimator に設定した弱識別器の内, 使用するサンプルの割合
                                                      # 
                    max_features = 1.0,               # The number of features to draw from X to train each base estimator.
                                                        # If float, then draw max_features * X.shape[1] features.
                    bootstrap = True,                 # ブートストラップサンプリングを行う 
                    bootstrap_features = False,       #
                    n_jobs = -1, 
                    random_state = args.seed
                )
        
        ada = AdaBoostClassifier(
                base_estimator = decition_tree,       # 弱識別器をして決定木を設定
                n_estimators = 1001,                  # バギングを構成する弱識別器の数 
                learning_rate = 0.01,                 # 
                random_state = args.seed              #
            )

        """
        model = EnsembleModelClassifier(
            classifiers  = [ xgboost, kNN, svm, forest, bagging, ada ],
            weights = [0.75, 0.25, 0.0, 0.25, 0.25, 0.25 ],
            vote_method = "majority_vote",
        )
        """
        model = EnsembleBlendingClassifier(
            classifiers  = [ xgboost, kNN, svm, forest ],
            weights = [0.75, 0.05, 0.05, 0.25, ],
            vote_method = "majority_vote",
        )

        #--------------------
        # モデルの学習処理
        #--------------------
        model.fit(X_train_fold, y_train_fold)

        #--------------------
        # モデルの推論処理
        #--------------------
        y_pred_test = model.predict(X_test)
        y_preds.append(y_pred_test)
        #print( "[{}] len(y_pred_test) : {}".format(fold_id, len(y_pred_test)) )

        y_pred_val[valid_index] = model.predict(X_valid_fold)
        #print( "[{}] len(y_pred_fold) : {}".format(fold_id, len(y_pred_val)) )
    
    accuracy = (y_train == y_pred_val).sum()/len(y_pred_val)
    print( "accuracy [val] : {:0.5f}".format(accuracy) )

    #================================
    # Kaggle API での submit
    #================================
    # 提出用データに値を設定
    y_sub = sum(y_preds) / len(y_preds)
    sub = ds_gender_submission
    sub['Survived'] = list(map(int, y_sub))
    sub.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)

    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
