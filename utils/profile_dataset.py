# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
from kaggle.api.kaggle_api_extended import KaggleApi

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str, default="../datasets/input/train.csv")
    parser.add_argument("--out_file", type=str, default="../datasets/output/dataset_report.html")
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    if( args.debug ):
        for key, value in vars(args).items():
            print('%s: %s' % (str(key), str(value)))

    """
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    """

    df = pd.read_csv( args.in_file )
    profile = ProfileReport(df)
    #profile = ProfileReport(df, title='Pandas Profiling Report', html={'style':{'full_width':True}})
    profile.to_file(outputfile=args.out_file)
