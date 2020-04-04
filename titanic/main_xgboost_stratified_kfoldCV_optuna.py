import os
import argparse
import numpy as np
import pandas as pd
import random
import warnings
from kaggle.api.kaggle_api_extended import KaggleApi

from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import optuna


def objective_wrapper(args, X_train, y_train):
    """
    objective に trial 以外の引数を指定可能にするためのラッパーメソッド
    """
    def objective(trial):
        #--------------------------------------------
        # ベイズ最適化でのチューニングパイパーパラメーター
        #--------------------------------------------
        params = {
            'booster': trial.suggest_categorical('booster', ['gbtree']),
            'objective': trial.suggest_categorical('objective', ['binary:logistic']),
            "learning_rate" : trial.suggest_loguniform("learning_rate", 1e-8, 0.01),                      # ハイパーパラメーターのチューニング時は固定  
            "n_estimators" : trial.suggest_int("n_estimators", 950, 1050),                                # 
            'max_depth': trial.suggest_int("max_depth", 3, 9),                                            # 3 ~ 9 : 一様分布に従う。1刻み
            'min_child_weight': trial.suggest_loguniform('min_child_weight', 0.1, 10.0),                  # 0.1 ~ 10.0 : 対数が一様分布に従う
            'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 0.95, 0.05),                    # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
            'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.6, 0.95, 0.05),      # 0.6 ~ 0.95 : 一様分布に従う。0.05 刻み
            'gamma': trial.suggest_loguniform("gamma", 1e-8, 1.0),                                        # 1e-8 ~ 1.0 : 対数が一様分布に従う
            'alpha': trial.suggest_float("alpha", 0.0, 0.0),                                              # デフォルト値としておく。余裕があれば変更
            'reg_lambda': trial.suggest_float("reg_lambda", 1.0, 1.0),                                         # デフォルト値としておく。余裕があれば変更
            'random_state': trial.suggest_int("random_state", 71, 71),
        }

        #--------------------------------------------
        # stratified k-fold CV での評価
        #--------------------------------------------
        # k-hold cross validation で、学習用データセットを学習用と検証用に分割したもので評価
        kf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

        for fold_id, (train_index, valid_index) in enumerate(kf.split(X_train, y_train)):
            #--------------------
            # データセットの分割
            #--------------------
            X_train_fold, X_valid_fold = X_train.iloc[train_index], X_train.iloc[valid_index]
            y_train_fold, y_valid_fold = y_train.iloc[train_index], y_train.iloc[valid_index]

            #--------------------
            # モデルの定義
            #--------------------
            model = XGBClassifier()

            #--------------------
            # モデルの学習処理
            #--------------------
            model.fit(X_train_fold, y_train_fold)

            #--------------------
            # モデルの推論処理
            #--------------------
            y_pred_val[valid_index] = model.predict(X_valid_fold)
            #print( "[{}] len(y_pred_fold) : {}".format(fold_id, len(y_pred_val)) )
        
        accuracy = (y_train == y_pred_val).sum()/len(y_pred_val)

        """
        #--------------------------------------------
        # データセット分割なしでの評価
        #--------------------------------------------
        # モデルの定義
        model = XGBClassifier()

        # モデルの学習処理
        model.fit(X_train, y_train)

        # 推論処理
        y_pred = model.predict(X_train)

        # 最良モデルの判断基準となるスコア
        accuracy = (y_train == y_pred).sum()/len(y_pred)
        """

        return accuracy

    return objective

if __name__ == '__main__':
    """
    stratified k-fold cross validation で学習用データセットを分割して学習＆評価
    学習モデルは xgboost
    Optuna によるハイパーパラメーターのチューニング
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="input")
    parser.add_argument("--out_dir", type=str, default="output")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--submit_message", type=str, default="From Kaggle API Python Script")
    parser.add_argument("--competition_id", type=str, default="titanic")
    parser.add_argument("--n_splits", type=int, default=4, help="CV での学習用データセットの分割数")
    parser.add_argument("--n_trials", type=int, default=100, help="Optuna での試行回数")
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

    #==============================
    # Optuna によるハイパーパラメーターのチューニング
    #==============================
    study = optuna.create_study(direction="maximize")
    study.optimize(objective_wrapper(args, X_train,y_train), n_trials=args.n_trials)
    print('best params : ', study.best_params)
    #print('best best_trial : ', study.best_trial)
    
    #================================
    # 最良モデルでの学習 & 推論
    #================================    
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
        # モデルの定義
        #--------------------
        model_best = XGBClassifier(
            booster = study.best_params['booster'],
            objective = study.best_params['objective'],
            learning_rate = study.best_params['learning_rate'],
            n_estimators = study.best_params['n_estimators'],
            max_depth = study.best_params['max_depth'],
            min_child_weight = study.best_params['min_child_weight'],
            subsample = study.best_params['subsample'],
            colsample_bytree = study.best_params['colsample_bytree'],
            gamma = study.best_params['gamma'],
            alpha = study.best_params['alpha'],
            reg_lambda = study.best_params['reg_lambda'],
            random_state = study.best_params['random_state']            
        )

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
        sub.to_csv( os.path.join(args.out_dir, args.submit_file), index=False)

        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.out_dir, args.submit_file), args.submit_message, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
