import os
import argparse
import numpy as np
import pandas as pd
import random
import warnings
from kaggle.api.kaggle_api_extended import KaggleApi

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

# XGBoost のデフォルトハイパーパラメーター（グリッドサーチ用）
params_xgboost = {
    'booster': ['gbtree'],
    'objective': ['binary:logistic'],
    "learning_rate" : [0.01],                         # ハイパーパラメーターのチューニング時は 0.1 で固定  
    "n_estimators" : [1000,1050,2000],
    'max_depth': [6],                         # 3 ~ 9 : 一様分布に従う。1刻み
    'min_child_weight': [0.1],           # 0.1 ~ 10.0 : 対数が一様分布に従う
    'subsample': [0.6],                   # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
    'colsample_bytree': [0.6],            # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
    'gamma': [0.001, 0.01],                       # 1e-8 ~ 1.0 : 対数が一様分布に従う
    'alpha': [0.0],                       # デフォルト値としておく。余裕があれば変更
    'reg_lambda': [1.0],                  # デフォルト値としておく。余裕があれば変更
    'random_state': [71],
}

if __name__ == '__main__':
    """
    stratified k-fold cross validation で学習用データセットを分割して学習＆評価
    学習モデルは xgboost
    グリッドサーチ
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="xgboost_stratified_kfoldCV_gridsearch", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="titanic")
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
    ds_train = pd.read_csv( os.path.join(args.datasets, "train.csv" ) )
    ds_test = pd.read_csv( os.path.join(args.datasets, "test.csv" ) )
    ds_gender_submission = pd.read_csv( os.path.join(args.datasets, "gender_submission.csv" ) )
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

    #==============================
    # 学習用データセットの分割
    #==============================
    # 学習用データセットとテスト用データセットの設定
    X_train = ds_train.drop('Survived', axis = 1)
    X_test = ds_test
    y_train = ds_train['Survived']
    y_pred_val = np.zeros((len(y_train),))
    if( args.debug ):
        print( "len(X_train) : ", len(X_train) )
        print( "len(y_train) : ", len(y_train) )
        print( "len(y_pred_val) : ", len(y_pred_val) )

    # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
    kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    #==============================
    # モデルの定義
    #==============================
    model = XGBClassifier()

    #==============================
    # grid search
    #==============================
    # グリッドサーチを行う,GridSearchCV クラスのオブジェクト作成
    gs = GridSearchCV(
            estimator = model,              # 推定器
            param_grid = params_xgboost,    # グリッドサーチの対象パラメータ
            scoring = 'accuracy',           # 
            cv = kf,                        # クロスバディゲーション
            n_jobs = -1                     # 全てのCPUで並列処理
    )

    # グリッドサーチを行う
    gs = gs.fit( X_train, y_train )

    # グリッドサーチの結果を print
    print( "sklearn.model_selection.GridSearchCV.best_score_ : \n", gs.best_score_ )        # 指定したモデルの内, 最もよいスコアを出したモデルのスコア
    print( "sklearn.model_selection.GridSearchCV.best_params_ : \n", gs.best_params_ )      # 最もよいスコアを出したモデルのパラメータ
    #print( "sklearn.model_selection.GridSearchCV.grid_scores_ : \n",gs.grid_scores_ )       # 全ての情報

    #================================
    # 最良モデルでの学習 & 推論
    #================================    
    # 最もよいスコアを出したモデルを抽出し, テストデータを評価
    model_best = gs.best_estimator_

    y_preds = []
    for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
        #--------------------
        # データセットの分割
        #--------------------
        X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
        y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

        #--------------------
        # モデルの学習処理
        #--------------------
        model_best.fit(X_train_fold, y_train_fold)

        #--------------------
        # モデルの推論処理
        #--------------------
        y_pred_test = model_best.predict(X_test)
        y_preds.append(y_pred_test)
        #print( "[{}] len(y_pred_test) : {}".format(fold_id, len(y_pred_test)) )

        y_pred_val[valid_index] = model_best.predict(X_valid_fold)
        #print( "[{}] len(y_pred_fold) : {}".format(fold_id, len(y_pred_val)) )
    
    accuracy = (y_train == y_pred_val).sum()/len(y_pred_val)
    print( "accuracy [val] : {:0.5f}".format(accuracy) )

    #================================
    # Kaggle API での submit
    #================================
    if( args.submit ):
        # 提出用データに値を設定
        y_sub = sum(y_preds) / len(y_preds)
        sub = ds_gender_submission
        sub['Survived'] = list(map(int, y_sub))
        sub.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)

        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
