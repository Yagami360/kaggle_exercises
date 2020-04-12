import os
import argparse
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import random
from kaggle.api.kaggle_api_extended import KaggleApi

# scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="datasets/input")
    parser.add_argument("--out_dir", type=str, default="datasets/output")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--submit_message", type=str, default="From Kaggle API Python Script")
    parser.add_argument("--competition_id", type=str, default="titanic")
    parser.add_argument("--val_rate", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    # seed 値の固定
    np.random.seed(args.seed)
    random.seed(args.seed)

    #================================
    # データセットの読み込み
    #================================
    ds_train = pd.read_csv( os.path.join(args.in_dir, "train.csv" ) )
    ds_test = pd.read_csv( os.path.join(args.in_dir, "test.csv" ) )
    ds_gender_submission = pd.read_csv( os.path.join(args.in_dir, "gender_submission.csv" ) )
    """
    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )
        print( "ds_gender_submission.head() : \n", ds_gender_submission.head() )
    """

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
    """
    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )
    """

    #================================
    # データセットの分割
    #================================
    # 学習用データセットとテスト用データセットの設定
    X_train = ds_train.drop('Survived', axis = 1)
    y_train = ds_train['Survived']
    X_test = ds_test
    if( args.debug ):
        print( "X_train.head() : \n", X_train.head() )
        print( "X_test.head() : \n", X_test.head() )
        print( "y_train.head() : \n", y_train.head() )
        print( "len(X_train) : ", len(X_train) )
        print( "len(X_test) : ", len(X_test) )
        print( "len(y_train) : ", len(y_train) )

    # hold-out 法で、学習用データセットを学習用と検証用に分割
    if not( args.submit ):
        # stratify 引数で y_train を指定することで、割合を保ったままデータセットを2つに分割
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args.val_rate, random_state=args.seed, stratify=y_train)
        if( args.debug ):
            print( "X_valid.head() : \n", X_valid.head() )
            print( "y_valid.head() : \n", y_valid.head() )
            print( "len(X_valid) : ", len(X_valid) )
            print( "len(y_valid) : ", len(y_valid) )

    #================================
    # モデルの定義
    #================================
    # ロジスティクス回帰
    model = LogisticRegression( penalty='l2', solver="sag", random_state=args.seed )

    #================================
    # モデルの学習処理
    #================================
    model.fit(X_train, y_train)

    #================================
    # モデルの推論処理
    #================================
    if( args.submit ):
        y_pred = model.predict(X_test)
        print( "y_pred : ", y_pred[:100] ) 
    else:
        y_pred = model.predict(X_valid)
        print( "y_pred : ", y_pred[:100] )

        # 正解率の計算
        print( "number of classified samples", (y_valid == y_pred).sum() )
        print( "accuracy : {:0.5f}".format( (y_valid == y_pred).sum()/len(y_pred) ) )

    #================================
    # Kaggle API での submit
    #================================
    if( args.submit ):
        # 提出用データに値を設定
        sub = ds_gender_submission
        sub['Survived'] = list(map(int, y_pred))
        sub.to_csv( os.path.join(args.out_dir, args.submit_file), index=False)

        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.out_dir, args.submit_file), args.submit_message, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
