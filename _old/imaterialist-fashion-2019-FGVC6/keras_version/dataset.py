import os
import numpy as np
import random
import pandas as pd
import re
import math
import cv2

# keras
import keras
from keras.utils import Sequence
from keras.utils import to_categorical

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

def make_mask_img( df_segument, n_classes = 46 ):
    print( df_segument.head() )
     
    # セグメンテーションマスク画像
    mask_width = df_segument.at[0, "Width"]
    mask_height = df_segument.at[0, "Height"]
    #print( "seg_width={}, seg_height={}".format(mask_width, mask_height) )
    mask_image = np.full(mask_width*mask_height, n_classes-1, dtype=np.int32)
    for encoded_pixels, class_id in zip(df_segument["EncodedPixels"].values, df_segument["ClassId"].values):
        pixels = list(map(int, encoded_pixels.split(" ")))
        #print( "encoded_pixels={}, class_id={}".format(encoded_pixels,class_id) )
        #print( "pixels={}".format(pixels) )
        for i in range(0,len(pixels), 2):
            start_pixel = pixels[i]-1 #index from 0
            len_mask = pixels[i+1]-1
            end_pixel = start_pixel + len_mask
            if int(class_id) < n_classes - 1:
                mask_image[start_pixel:end_pixel] = int(class_id)
            
    mask_image = mask_image.reshape((mask_height, mask_width), order='F')
    return mask_image

def load_dataset(
    args,
    dataset_dir, 
    image_height = 512, image_width = 512, n_channels = 3, n_classes = 46,
    one_hot_encode = True,
    n_samplings = -1,
):
    df_train_pairs = pd.read_csv( os.path.join(dataset_dir, "train.csv") )
    df_segument = df_train_pairs
    #df_segument['CategoryId'] = df_train_pairs['ClassId'].str.split('_').str[0]
    #df_image = df_segument.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x)).reset_index()
    #df_size = df_segument.groupby('ImageId')['Height', 'Width'].mean()
    #df_image = df_image.join(df_size , on='ImageId' )
    df_train_pairs = df_segument
    print( df_train_pairs.shape )
    print( df_train_pairs.head() )

    image_names = df_train_pairs["ImageId"]
    image_names = image_names[0: min(n_samplings, len(image_names))]

    # セグメンテーションマスク画像
    """
    for i, name in enumerate(image_names):
        mask_image = make_mask_img( df_segument.loc[i], n_classes )
        cv2.imwrite( os.path.join(args.results_dir, args.exper_name, name ), mask_image )
    """
    
    # X_train
    X_train = np.zeros( (len(image_names), image_height, image_width, 3), dtype=np.uint8 )
    for i, name in enumerate(image_names):
        img = cv2.imread( os.path.join( dataset_dir, "train", name ) )
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]
        X_train[i] = img

    # y_train
    if( one_hot_encode ):
        y_train = np.zeros( (len(image_names), n_classes), dtype=np.uint8 )
    else:
        y_train = np.zeros( (len(image_names), 1), dtype=np.uint8 )

    for i, name in enumerate(image_names):
        if( one_hot_encode ):
            y_train[i] = to_categorical( df_train_pairs["ClassId"].iloc[i], n_classes )
        else:
            y_train[i] = df_train_pairs["ClassId"].iloc[i]

    # X_test
    image_names = sorted( [f for f in os.listdir(os.path.join( dataset_dir, "test")) if f.endswith(IMG_EXTENSIONS)], key=lambda s: int(re.search(r'\d+', s).group()) )
    image_names = image_names[0: min(n_samplings, len(image_names))]
    X_test = np.zeros( (len(image_names), image_height, image_width, 3), dtype=np.uint8 )
    for i, name in enumerate(image_names):
        img = cv2.imread( os.path.join( dataset_dir, "test", name ) )
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_LANCZOS4 )  # shape = [H,W,C]
        X_test[i] = img

    return X_train, y_train, X_test


