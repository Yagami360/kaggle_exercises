import os
import numpy as np
import random
import pandas as pd
import re
import math
import cv2
from skimage.transform import resize

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# keras
import keras
from keras.utils import Sequence
from keras.utils import to_categorical

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

def load_dataset(
    dataset_dir, 
    image_height_org = 101, image_width_org = 101,
    image_height = 128, image_width = 128, n_channels = 1,
    n_samplings = -1,
    debug = True,
):
    df_train = pd.read_csv( os.path.join(dataset_dir, "train.csv"), index_col='id' )
    if( debug ):
        print( df_train.head() )

    df_depth = pd.read_csv( os.path.join(dataset_dir, "depths.csv"), index_col='id' )
    if( debug ):
        print( df_depth.head() )

    # サンプリングする画像名リスト
    train_image_names = df_train.index.values
    if( n_samplings != -1 ):
        train_image_names = train_image_names[0: min(n_samplings, len(train_image_names))]

    for i, name in enumerate(train_image_names):
        train_image_names[i] = name + ".png"

    # X_train_img
    X_train_img = np.zeros( (len(train_image_names), image_height, image_width, n_channels), dtype=np.float32 )
    for i, name in enumerate(train_image_names):
        img = cv2.imread( os.path.join( dataset_dir, "train", "images", name ), cv2.IMREAD_GRAYSCALE ) / 255    # 0.0f ~ 1.0f
        #img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_LANCZOS4 )                # shape = [H,W,C]
        img = resize(img, (image_height, image_width), mode='constant', preserve_range=True)
        X_train_img[i] = img.reshape( (image_height, image_width, n_channels))

    # y_train（マスク画像）
    y_train_mask = np.zeros( (len(train_image_names), image_height, image_width, 1), dtype=np.float32 )
    for i, name in enumerate(train_image_names):
        img = cv2.imread( os.path.join( dataset_dir, "train", "masks", name ), cv2.IMREAD_GRAYSCALE ) / 255     # 0.0f ~ 1.0f
        #img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_NEAREST )                 # shape = [H,W,C]
        img = resize(img, (image_height, image_width), mode='constant', preserve_range=True)
        y_train_mask[i] = img.reshape( (image_height, image_width, 1))

    # X_test
    test_image_names = sorted( [f for f in os.listdir(os.path.join( dataset_dir, "test", "images")) if f.endswith(IMG_EXTENSIONS)] )
    if( n_samplings != -1 ):
        test_image_names = test_image_names[0: min(n_samplings, len(test_image_names))]

    X_test_img = np.zeros( (len(test_image_names), image_height, image_width, n_channels), dtype=np.float32 )
    for i, name in enumerate(test_image_names):
        img = cv2.imread( os.path.join( dataset_dir, "test", "images", name ), cv2.IMREAD_GRAYSCALE ) / 255     # 0.0f ~ 1.0f
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_LANCZOS4 )                # shape = [H,W,C]
        X_test_img[i] = img.reshape( (image_height, image_width, n_channels))

    # depth
    scaler = StandardScaler()
    scaler.fit( df_depth["z"].values.reshape(-1,1) )
    df_depth["z"] = scaler.transform( df_depth["z"].values.reshape(-1,1) )
    if( debug ):
        print( df_depth.head() )

    X_train_depth = np.zeros((len(train_image_names), 1), dtype=np.float32)
    for i, name in enumerate(train_image_names):   
        id = name.split(".png")[0]
        X_train_depth[i] = df_depth.loc[id,"z"]

    X_test_depth = np.zeros((len(test_image_names), 1), dtype=np.float32)
    for i, name in enumerate(test_image_names):   
        id = name.split(".png")[0]
        X_test_depth[i] = df_depth.loc[id,"z"]

    # クラスラベルを追加（画像内で塩が含まれる割合のクラス）
    """
    def cov_to_class(val):    
        for i in range(0, 11):
            if val * 10 <= i :
                return i

    coverage = np.zeros( (len(train_image_names), image_height, image_width, n_channels), dtype=np.float32 )
    
    np.sum(y_train_mask) / ( image_height_org * image_width_org )
    
    coverage_class = []
    cov_to_class(coverage)
    """

    return X_train_img, y_train_mask, X_test_img, X_train_depth, X_test_depth, train_image_names, test_image_names
