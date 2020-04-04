import os
import argparse
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import random
import warnings
from kaggle.api.kaggle_api_extended import KaggleApi

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# XGBoost のデフォルトハイパーパラメーター
params_xgboost = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    "learning_rate" : 0.01,             # ハイパーパラメーターのチューニング時は 0.1 で固定  
    "n_estimators" : 1050,
    'max_depth': 5,                     # 3 ~ 9 : 一様分布に従う。1刻み
    'min_child_weight': 1,              # 0.1 ~ 10.0 : 対数が一様分布に従う
    'subsample': 0.8,                   # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
    'colsample_bytree': 0.8,            # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
    'gamma': 0.0,                       # 1e-8 ~ 1.0 : 対数が一様分布に従う
    'alpha': 0.0,                       # デフォルト値としておく。余裕があれば変更
    'reg_lambda': 1.0,                  # デフォルト値としておく。余裕があれば変更
    'random_state': 71,
}

if __name__ == '__main__':
    """
    hold-out 法で学習用データセットを分割して評価
    学習モデルは xgboost
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="input")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--submit_message", type=str, default="From Kaggle API Python Script")
    parser.add_argument("--competition_id", type=str, default="titanic")
    parser.add_argument("--val_rate", type=float, default=0.25, help="hold-out 法での検証用データセットの割合")
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
    ds_gender_submission = pd.read_csv( os.path.join(args.in_dir, "gender_submission.csv" ) )
    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )
        print( "ds_gender_submission.head() : \n", ds_gender_submission.head() )
    
    # データのプロファイリング
    #profile = ProfileReport(ds_train)
    #profile.to_file(outputfile= os.path.join(args.ourdir, "train_csv.html") )

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

    age_avg = ds_train['Age'].mean()
    age_std = ds_train['Age'].std()
    ds_train['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)
    ds_test['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )
        
    #================================
    # submit 時の処理
    #================================
    if( args.submit ):
        #--------------------------------
        # データセットの分割
        #--------------------------------
        # 学習用データセットとテスト用データセットの設定
        X_train = ds_train.drop('Survived', axis = 1)
        y_train = ds_train['Survived']
        X_test = ds_test
        if( args.debug ):
            print( "X_train.head() : \n", X_train.head() )
            print( "y_train.head() : \n", y_train.head() )
            print( "X_test.head() : \n", X_test.head() )
            print( "len(X_train) : ", len(X_train) )
            print( "len(y_train) : ", len(y_train) )

        #--------------------------------
        # モデルの定義
        #--------------------------------
        model = XGBClassifier(
            booster = params_xgboost['booster'],
            objective = params_xgboost['objective'],
            learning_rate = params_xgboost['learning_rate'],
            n_estimators = params_xgboost['n_estimators'],
            max_depth = params_xgboost['max_depth'],
            min_child_weight = params_xgboost['min_child_weight'],
            subsample = params_xgboost['subsample'],
            colsample_bytree = params_xgboost['colsample_bytree'],
            gamma = params_xgboost['gamma'],
            alpha = params_xgboost['alpha'],
            reg_lambda = params_xgboost['reg_lambda'],
            random_state = params_xgboost['random_state']
        )
 
        #--------------------------------
        # 学習処理
        #--------------------------------
        model.fit(X_train, y_train)    

        #--------------------------------
        # 推論処理
        #--------------------------------
        y_pred = model.predict(X_test)
        y_sub = y_pred
        print( "len(y_pred) : ", len(y_pred) ) 
        print( "y_pred : ", y_pred[:100] ) 

    #================================
    # 非 submit 時の処理
    #================================
    else:
        # 学習用データセットとテスト用データセットの設定
        X_train = ds_train.drop('Survived', axis = 1)
        y_train = ds_train['Survived']
        X_test = ds_test
        if( args.debug ):
            print( "X_train.head() : \n", X_train.head() )
            print( "y_train.head() : \n", y_train.head() )
            print( "X_test.head() : \n", X_test.head() )

        # stratify 引数で y_train を指定することで、y_train のデータ (0 or 1) の割合を保ったままデータセットを2つに分割
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args.val_rate, random_state=args.seed, stratify=y_train)
        if( args.debug ):
            print( "X_valid.head() : \n", X_valid.head() )
            print( "y_valid.head() : \n", y_valid.head() )
            print( "len(X_valid) : ", len(X_valid) )
            print( "len(y_valid) : ", len(y_valid) )

        #--------------------------------
        # モデルの定義
        #--------------------------------
        model = XGBClassifier(
            booster = params_xgboost['booster'],
            objective = params_xgboost['objective'],
            learning_rate = params_xgboost['learning_rate'],
            n_estimators = params_xgboost['n_estimators'],
            max_depth = params_xgboost['max_depth'],
            min_child_weight = params_xgboost['min_child_weight'],
            subsample = params_xgboost['subsample'],
            colsample_bytree = params_xgboost['colsample_bytree'],
            gamma = params_xgboost['gamma'],
            alpha = params_xgboost['alpha'],
            reg_lambda = params_xgboost['reg_lambda'],
            random_state = params_xgboost['random_state']
        )
        
        #--------------------------------
        # モデルの学習処理
        #--------------------------------
        model.fit(X_train, y_train)

        #--------------------------------
        # モデルの推論処理
        #--------------------------------
        y_pred = model.predict(X_valid)
        print( "y_pred : ", y_pred[:100] )
        print( "y_pred : ", len(y_pred) )

        # 正解率の計算
        print( "number of classified samples : ", (y_valid == y_pred).sum() )
        print( "accuracy : {:0.5f}".format( (y_valid == y_pred).sum()/len(y_pred) ) )
        #print( "accuracy : ", accuracy_score(y_valid, y_pred) )

    #================================
    # Kaggle API での submit
    #================================
    if( args.submit ):
        # 提出用データに値を設定
        sub = ds_gender_submission
        sub['Survived'] = list(map(int, y_sub))
        sub.to_csv( os.path.join(args.out_dir, args.submit_file), index=False)

        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.out_dir, args.submit_file), args.submit_message, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
        
