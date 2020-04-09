import os
import argparse
import numpy as np
import pandas as pd
import random
import warnings
from matplotlib import pyplot as plt
import seaborn as sns
from kaggle.api.kaggle_api_extended import KaggleApi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir", type=str, default="datasets/input")
    parser.add_argument("--out_dir", type=str, default="datasets/output")
    parser.add_argument("--submit_file", type=str, default="submission.csv")
    parser.add_argument("--submit_message", type=str, default="From Kaggle API Python Script")
    parser.add_argument("--competition_id", type=str, default="house-prices-advanced-regression-techniques")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)



    #================================
    # Kaggle API での submit
    #================================
    # 提出用データに値を設定
    y_sub = sum(y_preds) / len(y_preds)
    sub = ds_sample_submission
    if( args.target_norm ):
        sub['label'] = list(map(float, np.exp(y_sub)))
    else:
        sub['label'] = list(map(float, y_sub))

    sub.to_csv( os.path.join(args.out_dir, args.submit_file), index=False)
    
    if( args.submit ):
        # Kaggle-API で submit
        api = KaggleApi()
        api.authenticate()
        api.competition_submit( os.path.join(args.out_dir, args.submit_file), args.submit_message, args.competition_id)
        os.system('kaggle competitions submissions -c {}'.format(args.competition_id) )