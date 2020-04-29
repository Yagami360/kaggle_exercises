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

def load_dataset(
    dataset_dir, 
    image_height = 32, image_width = 32, n_classes = 10,
    one_hot_encode = True,
):
    df_train = pd.read_csv( os.path.join(dataset_dir, "train.csv") )
    df_test = pd.read_csv( os.path.join(dataset_dir, "test.csv") )

    X_train = ( df_train.drop(labels = ["label"],axis = 1).values.reshape(-1, 28, 28, 1) ) / 255.0
    X_test = ( df_test.values.reshape(-1, 28, 28, 1) ) / 255.0

    # resize (事前学習済み resnet 等に入力出来るサイズするため)
    X_train_resize = np.zeros( shape=(len(X_train), image_height, image_width, 3) )
    for i in range(len(X_train)):
        X_train_cv = cv2.resize(X_train[i], dsize = (image_width,image_height), interpolation = cv2.INTER_AREA) 
        X_train_resize[i,:,:,0] = X_train_cv
        X_train_resize[i,:,:,1] = X_train_cv
        X_train_resize[i,:,:,2] = X_train_cv

    X_test_resize = np.zeros( shape=(len(X_test), image_height, image_width, 3) )
    for i in range(len(X_test)):
        X_test_cv = cv2.resize(X_test[i], dsize = (image_width,image_height), interpolation = cv2.INTER_AREA) 
        X_test_resize[i,:,:,0] = X_test_cv
        X_test_resize[i,:,:,1] = X_test_cv
        X_test_resize[i,:,:,2] = X_test_cv

    y_train = df_train["label"]
    if( one_hot_encode ):
        y_train = to_categorical(y_train, num_classes = 10)
        
    return X_train_resize, y_train, X_test_resize


