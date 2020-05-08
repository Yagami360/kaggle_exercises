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

def load_dataset(
    dataset_dir, 
    image_height = 512, image_width = 512, n_channels = 3, n_classes = 46,
    one_hot_encode = True,
    n_samplings = -1,
):
    df_train_pairs = pd.read_csv( os.path.join(dataset_dir, "train.csv") )
    #df_train_pairs['CategoryId'] = df_train_pairs['ClassId'].str.split('_').str[0]
    #df_train_pairs = df_train_pairs.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x))
    print( df_train_pairs.shape )
    print( df_train_pairs.head() )

    # X_train
    image_names = df_train_pairs["ImageId"]
    image_names = image_names[0: min(n_samplings, len(image_names))]
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


