import os
import argparse
import numpy as np
import pandas as pd
import yaml
from pandas_profiling import ProfileReport
import random
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from kaggle.api.kaggle_api_extended import KaggleApi

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import xgboost as xgb


if __name__ == '__main__':
    """
    hold-out 法で学習用データセットを分割して評価
    学習モデルは xgboost
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exper_name", default="xgboost", help="実験名")
    parser.add_argument("--dataset_dir", type=str, default="datasets")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--competition_id", type=str, default="titanic")
    parser.add_argument("--params_file", type=str, default="parames/xgboost_classifier_default.yml")
    parser.add_argument('--train_type', choices=['train', 'fit'], default="train", help="XGBoost の学習タイプ")
    parser.add_argument("--val_rate", type=float, default=0.25, help="hold-out 法での検証用データセットの割合")
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
    ds_test = pd.read_csv( os.path.join(args.dataset_dir, "test.csv" ) )
    ds_gender_submission = pd.read_csv( os.path.join(args.dataset_dir, "gender_submission.csv" ) )
    if( args.debug ):
        print( "df_train.head() : \n", df_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )
        print( "ds_gender_submission.head() : \n", ds_gender_submission.head() )
    
    #================================
    # 前処理
    #================================
    # 無用なデータを除外
    df_train.drop(['Name', 'PassengerId'], axis=1, inplace=True)
    ds_test.drop(['Name', 'PassengerId'], axis=1, inplace=True)
    df_train.drop(['SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
    ds_test.drop(['SibSp', 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

    # 全特徴量を一括で処理
    for col in df_train.columns:
        if( args.debug ):
            print( "df_train[{}].dtypes ] : {}".format(col, df_train[col].dtypes))

        # 目的変数
        if( col in ["Survived"] ):
            continue

        #-----------------------------
        # 欠損値の埋め合わせ
        #-----------------------------
        # NAN 値の埋め合わせ（平均値）
        if( col in ["Age", 'Fare'] ):
            df_train[col].fillna(np.mean(df_train[col]), inplace=True)
            ds_test[col].fillna(np.mean(df_train[col]), inplace=True)
        # NAN 値の埋め合わせ（ゼロ値）/ int 型
        elif( df_train[col].dtypes in ["int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"] ):
            df_train[col].fillna(0, inplace=True)
            ds_test[col].fillna(0, inplace=True)
        # NAN 値の埋め合わせ（ゼロ値）/ float 型
        elif( df_train[col].dtypes in ["float16", "float32", "float64", "float128"] ):
            df_train[col].fillna(0.0, inplace=True)
            ds_test[col].fillna(0.0, inplace=True)
        # NAN 値の補完（None値）/ object 型
        else:
            df_train[col] = df_train[col].fillna('NA')
            ds_test[col] = ds_test[col].fillna('NA')

        #-----------------------------
        # 正規化処理
        #-----------------------------
        """
        #if( df_train[col].dtypes != "object" ):
        if( df_train[col].dtypes in ["float16", "float32", "float64", "float128"] ):
            scaler = StandardScaler()
            scaler.fit( df_train[col].values.reshape(-1,1) )
            df_train[col] = scaler.fit_transform( df_train[col].values.reshape(-1,1) )
            ds_test[col] = scaler.fit_transform( ds_test[col].values.reshape(-1,1) )
        """
        
        #-----------------------------
        # ラベル情報のエンコード
        #-----------------------------
        if( df_train[col].dtypes == "object" ):
            label_encoder = LabelEncoder()
            label_encoder.fit(list(df_train[col]))
            df_train[col] = label_encoder.transform(list(df_train[col]))

            label_encoder = LabelEncoder()
            label_encoder.fit(list(ds_test[col]))
            ds_test[col] = label_encoder.transform(list(ds_test[col]))

    # 前処理後のデータセットを外部ファイルに保存
    df_train.to_csv( os.path.join(args.results_dir, args.exper_name, "train_preprocessed.csv"), index=True)
    ds_test.to_csv( os.path.join(args.results_dir, args.exper_name, "test_preprocessed.csv"), index=True)
    if( args.debug ):
        print( "df_train.head() : \n", df_train.head() )
        print( "ds_test.head() : \n", ds_test.head() )

    #================================
    # データセットの分割
    #================================
    # 学習用データセットとテスト用データセットの設定
    X_train = df_train.drop('Survived', axis = 1)
    y_train = df_train['Survived']
    X_test = ds_test
    if( args.debug ):
        print( "X_train.head() : \n", X_train.head() )
        print( "y_train.head() : \n", y_train.head() )
        print( "X_test.head() : \n", X_test.head() )

    # stratify 引数で y_train を指定することで、y_train のデータ (0 or 1) の割合を保ったままデータセットを2つに分割
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=args.val_rate, random_state=args.seed, stratify=y_train)

    # XGBoost 用データセットに変換
    if( args.train_type == "train" ):
        X_train_dmat = xgb.DMatrix(X_train, label=y_train)
        X_valid_dmat = xgb.DMatrix(X_valid, label=y_valid)
        X_test_dmat = xgb.DMatrix(X_test, label=y_train)

    #================================
    # モデルの定義
    #================================
    # モデルのパラメータの読み込み
    with open( args.params_file ) as f:
        params = yaml.safe_load(f)
        model_params = params["model"]["model_params"]
        model_train_params = params["model"]["train_params"]
        if( args.debug ):
            print( "params :\n", params )

    # モデルの定義
    if( args.train_type == "fit" ):
        model = xgb.XGBClassifier(
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

    #================================
    # モデルの学習処理
    #================================
    if( args.train_type == "train" ):
        evals_result = {}
        model = xgb.train(
            model_params, X_train_dmat, 
            num_boost_round = model_train_params["num_boost_round"],
            early_stopping_rounds = model_train_params["early_stopping_rounds"],
            evals = [ (X_train_dmat, 'train'), (X_valid_dmat, 'val') ],
            evals_result = evals_result
        )
    else:
        model.fit(X_train, y_train)

    #================================
    # モデルの推論処理
    #================================
    if( args.train_type == "train" ):
        y_pred_prob = model.predict(X_test_dmat)
    else:
        y_pred_prob = model.predict(X_test)

    y_pred = np.where(y_pred_prob > 0.5, 1, 0)
    print( "y_pred : ", y_pred[:100] )
    print( "y_pred : ", len(y_pred) )

    if( args.train_type == "train" ):
        y_pred_prob_val = model.predict(X_valid_dmat)
    else:
        y_pred_prob_val = model.predict(X_valid)

    y_pred_val = np.where(y_pred_prob_val > 0.5, 1, 0)

    # 正解率の計算
    print( "accuracy [val] : {:0.5f}".format( (y_valid == y_pred_val).sum()/len(y_pred_val) ) )

    # 重要特徴量
    if( args.train_type == "train" ):
        print( "[Feature Importances] : \n", model.get_fscore() )
    else:
        print( "[Feature Importances]" )
        for i, col in enumerate(X_train.columns):
            print( "{} : {:.4f}".format( col, model.feature_importances_[i] ) )

    #================================
    # 可視化処理
    #================================
    # loss
    if( args.train_type == "train" ):
        plt.plot(evals_result['train'][model_train_params["eval_metric"]], label='train')
        plt.plot(evals_result['val'][model_train_params["eval_metric"]], label='val')

        plt.xlabel('iters')
        plt.ylabel(model_train_params["eval_metric"])
        plt.xlim( [0,model_train_params["num_boost_round"]+1] )
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig( os.path.join(args.results_dir, args.exper_name, "losees.png"), dpi = 300, bbox_inches = 'tight' )

    # 重要特徴量
    _, ax = plt.subplots(figsize=(8, 4))
    xgb.plot_importance(
        model,
        ax = ax,
        importance_type = 'gain',
        show_values = False
    )
    plt.tight_layout()
    plt.savefig( os.path.join(args.results_dir, args.exper_name, "feature_importances.png"), dpi = 300, bbox_inches = 'tight' )

    #================================
    # Kaggle API での submit
    #================================
    if( args.submit ):
        # 提出用データに値を設定
        sub = ds_gender_submission
        sub['Survived'] = list(map(int, y_sub))
        sub.to_csv( os.path.join(args.results_dir, args.exper_name, args.submit_file), index=False)

        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.results_dir, args.exper_name, args.submit_file), args.exper_name, args.competition_id)
        #os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )
        
