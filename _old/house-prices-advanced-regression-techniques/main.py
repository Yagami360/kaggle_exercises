import os
import argparse
import numpy as np
import pandas as pd
import random
import warnings
from kaggle.api.kaggle_api_extended import KaggleApi

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="datasets/input")
    parser.add_argument("--out_dir", type=str, default="datasets/output")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--submit_message", type=str, default="From Kaggle API Python Script")
    parser.add_argument("--competition_id", type=str, default="house-prices-advanced-regression-techniques")
    parser.add_argument("--n_splits", type=int, default=4, help="CV での学習用データセットの分割数")
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
    ds_sample_submission = pd.read_csv( os.path.join(args.in_dir, "sample_submission.csv" ) )
    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )
        print( "ds_sample_submission.head() : \n", ds_sample_submission.head() )
    
    #================================
    # 前処理
    #================================
    # 無用なデータを除外
    #ds_train.drop(['xxx', 'xxx', 'xxx'], axis=1, inplace=True)
    #ds_test.drop(['xxx', 'xxx', 'xxx'], axis=1, inplace=True)

    # ラベル情報のエンコード
    """
    for c in cols:
        label_encoder = LabelEncoder() 
        label_encoder.fit(list(all_data[c].values)) 
        all_data[c] = label_encoder.transform(list(all_data[c].values))
    """

    # NAN 値の埋め合わせ

    """
    # MSSubClass : The building class / 建物クラス
    pass

    # MSZoning : The general zoning classification / 一般的なゾーニングの分類
    ds_train['MSZoning'].replace(["RL","RM", "FV", "RH", "C (all)"], [0, 1, 2, 3, 4], inplace=True)
    ds_test['MSZoning'].replace(["RL","RM", "FV", "RH", "C (all)"], [0, 1, 2, 3, 4], inplace=True)

    # LotFrontage :  Linear feet of street connected to property / 土地に接続された道路の直線フィート
    ds_train['LotFrontage'].fillna(np.mean(ds_train['LotFrontage']), inplace=True)
    ds_test['LotFrontage'].fillna(np.mean(ds_train['LotFrontage']), inplace=True)

    # LotFrontage : Lot size in square feet/ 平方フィートのロットサイズ
    pass

    # Street
    ds_train['Street'].replace(["Pave","Grvl"], [0, 1], inplace=True)
    ds_test['Street'].replace(["Pave","Grvl"], [0, 1], inplace=True)

    # Alley : Type of alley access / 路地アクセスの種類
    ds_train['Alley'].fillna(('NA'), inplace=True)
    ds_train['Alley'] = ds_train['Alley'].map( {'NA': 0, 'Pave': 1, 'Grvl': 2} ).astype(int)
    ds_test['Alley'].fillna(('NA'), inplace=True)
    ds_test['Alley'] = ds_test['Alley'].map( {'NA': 0, 'Pave': 1, 'Grvl': 2} ).astype(int)

    # LotShape : General shape of property / 物件の一般的な形状
    ds_train['LotShape'].replace(["Reg","IR1","IR2","IR3"], [0, 1, 2, 3], inplace=True)
    ds_test['LotShape'].replace(["Reg","IR1","IR2","IR3"], [0, 1, 2, 3], inplace=True)
    """

    if( args.debug ):
        print( "ds_train.head() : \n", ds_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )

    #===========================================
    # k-fold CV による処理
    #===========================================
    """
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
    """

    #================================
    # Kaggle API での submit
    #================================
    """
    # 提出用データに値を設定
    y_sub = sum(y_preds) / len(y_preds)
    sub = ds_sample_submission
    sub['Survived'] = list(map(int, y_sub))
    sub.to_csv( os.path.join(args.out_dir, args.submit_file), index=False)

    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.out_dir, args.submit_file), args.submit_message, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
    """