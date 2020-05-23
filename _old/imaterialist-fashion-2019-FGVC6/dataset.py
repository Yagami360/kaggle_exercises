import os
from tqdm import tqdm
import numpy as np
import random
import pandas as pd
import re
import math
from PIL import Image, ImageDraw, ImageOps
import cv2
#from skimage.transform import resize

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# PyTorch
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.utils import save_image

from utils import set_random_seed

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

def save_mask_images( df_mask, n_classes = 47, save_dir = "" ):
    for i in tqdm( range(len(df_mask)), desc="saving mask images"):
        name = df_mask.iloc[i]["ImageId"]
        height = df_mask.iloc[i]["Height"]
        width = df_mask.iloc[i]["Width"]
        encoded_pixels = df_mask.iloc[i]["EncodedPixels"]
        class_id = df_mask.iloc[i]["ClassId"]
        #print( "name={}".format(name) )
        #print( "height={}, width={}".format(height, width) )

        # encoded_pixels: 3655609 3 3658608 9 3661608 14 3664607 ...
        # pixels : [3655609, 3, 3658608, 9, 3661608, 14, ...]
        pixels = list(map(int, encoded_pixels.split(" ")))
        #print( "encoded_pixels={}".format(encoded_pixels) )
        #print( "pixels={}".format(pixels) )

        # セグメンテーションマスク画像    
        mask_image = np.full(height*width, n_classes-1, dtype=np.int32)

        for i in range(0,len(pixels), 2):
            start_pixel = pixels[i]-1 #index from 0
            len_mask = pixels[i+1]-1
            end_pixel = start_pixel + len_mask
            if int(class_id) < n_classes - 1:
                mask_image[start_pixel:end_pixel] = int(class_id)
            
        mask_image = mask_image.reshape((height, width), order='F')
        cv2.imwrite( os.path.join(save_dir, name), mask_image )

    return

"""
def load_dataset(
    args,
    dataset_dir, 
    image_height = 512, image_width = 512, n_channels = 3, n_classes = 47,
    one_hot_encode = True,
    n_samplings = -1,
):
    df_train_pairs = pd.read_csv( os.path.join(dataset_dir, "train.csv") )
    df_mask = df_train_pairs
    #df_mask['CategoryId'] = df_train_pairs['ClassId'].str.split('_').str[0]
    #df_image = df_mask.groupby('ImageId')['EncodedPixels', 'CategoryId'].agg(lambda x: list(x)).reset_index()
    #df_size = df_mask.groupby('ImageId')['Height', 'Width'].mean()
    #df_image = df_image.join(df_size , on='ImageId' )
    df_train_pairs = df_mask
    print( df_train_pairs.shape )
    print( df_train_pairs.head() )

    image_names = df_train_pairs["ImageId"]
    image_names = image_names[0: min(n_samplings, len(image_names))]

    # セグメンテーションマスク画像
    if( args.save_mask ):
        if not os.path.isdir( os.path.join(args.dataset_dir, "train_mask") ):
            os.mkdir( os.path.join(args.dataset_dir, "train_mask") )

        save_mask_images( df_mask, n_classes, os.path.join(args.dataset_dir, "train_mask") )
    else:
        pass

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
"""

class ImaterialistDataset(data.Dataset):
    def __init__(self, args, dataset_dir, datamode = "train", image_height = 512, image_width = 512, n_classes = 47, data_augument = False, debug = False ):
        super(ImaterialistDataset, self).__init__()
        self.args = args
        self.dataset_dir = dataset_dir
        self.datamode = datamode
        self.image_height = image_height
        self.image_width = image_width
        self.n_classes = n_classes
        self.data_augument = data_augument
        self.debug = debug

        self.df_train = pd.read_csv( os.path.join(self.dataset_dir, "train.csv") )
        #self.df_train = pd.read_csv( os.path.join(self.dataset_dir, "train.csv"), index_col='ImageId' )
        self.image_names = sorted( [f for f in os.listdir( os.path.join(self.dataset_dir, "train")) if f.endswith(IMG_EXTENSIONS)] )

        mean = [ 0.5 for i in range(args.n_channels) ]
        std = [ 0.5 for i in range(args.n_channels) ]

        # transform
        if( data_augument ):
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
#                    transforms.RandomResizedCrop( (args.image_height, args.image_width) ),
                    transforms.RandomHorizontalFlip(),
#                    transforms.RandomVerticalFlip(),
#                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.BICUBIC ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( mean, std ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( mean, std ),
                ]
            )

        if( self.debug ):
            print( "self.dataset_dir :", self.dataset_dir)
            print( "len(self.image_names) :", len(self.image_names))
            print( "self.df_train.head() \n:", self.df_train.head())

        return

    def __len__(self):
        return len(self.image_names)

    def get_mask_image(self, df_mask, image_height = 512, image_width = 512, n_channels = 1, n_classes = 47 ):
        height = df_mask["Height"]  
        width = df_mask["Width"]
        encoded_pixels = df_mask["EncodedPixels"]
        class_id = df_mask["ClassId"]

        # encoded_pixels: 3655609 3 3658608 9 3661608 14 3664607 ...
        # pixels : [3655609, 3, 3658608, 9, 3661608, 14, ...]
        pixels = list(map(int, encoded_pixels.split(" ")))
        #print( "encoded_pixels={}".format(encoded_pixels) )
        #print( "pixels={}".format(pixels) )

        mask_image = np.full(height*width, n_classes-1, dtype=np.int32)
        for i in range(0,len(pixels), 2):
            start_pixel = pixels[i]-1   #index from 0
            len_mask = pixels[i+1]-1
            end_pixel = start_pixel + len_mask
            if int(class_id) < n_classes - 1:
                mask_image[start_pixel:end_pixel] = int(class_id)
            
        mask_image = mask_image.reshape((height, width), order='F')
        if( n_channels == 1):
            mask_image = Image.fromarray(mask_image).convert("L")
        else:
            mask_image = Image.fromarray(mask_image).convert("RGB")

        return mask_image

    def get_mask_image2(self, df_mask, image_height = 512, image_width = 512, n_channels = 1, n_classes = 47 ):
        """
        train.csv の EncodedPixels 値からマスク画像を生成
        同名の ImageId に対して、複数のラベル値とマスク画像が存在するので、ラベル値分のチャンネルをもつ [H,W,C=ラベル数] のマスク画像となる
        """
        height = df_mask.iloc[0]["Height"]  
        width = df_mask.iloc[0]["Width"]
        #print( "height={}, width={}".format(height, width) )
        
        # セグメンテーションマスク画像    
        mask_image = np.full(height*width, n_classes-1, dtype=np.int32)
        for encoded_pixels, class_id in zip(df_mask["EncodedPixels"].values, df_mask["ClassId"].values):
            # encoded_pixels: 3655609 3 3658608 9 3661608 14 3664607 ...
            # pixels : [3655609, 3, 3658608, 9, 3661608, 14, ...]
            pixels = list(map(int, encoded_pixels.split(" ")))
            #print( "encoded_pixels={}".format(encoded_pixels) )
            #print( "pixels={}".format(pixels) )

            for i in range(0,len(pixels), 2):
                start_pixel = pixels[i]-1   #index from 0
                len_mask = pixels[i+1]-1
                end_pixel = start_pixel + len_mask
                if int(class_id) < n_classes - 1:
                    mask_image[start_pixel:end_pixel] = int(class_id)
                
            # ? : チャンネル次元が消えるので、同名の ImageId に対して存在する複数のラベル値とマスク画像が消える。（最初の ImageID のみのラベル値とマスク画像になｒｙ）
            mask_image = mask_image.reshape((height, width), order='F')

        if( n_channels == 1):
            mask_image = Image.fromarray(mask_image).convert("L")
        else:
            mask_image = Image.fromarray(mask_image).convert("RGB")

        return mask_image

    def __getitem__(self, index):
        image_name = self.image_names[index]
        
        #-------------
        # image
        #-------------
        image = Image.open(os.path.join(self.dataset_dir, self.datamode, image_name)).convert('RGB')
        self.seed_da = random.randint(0,10000)
        if( self.data_augument ):
            set_random_seed( self.seed_da )

        image = self.transform(image)

        #-------------
        # mask
        #-------------
        if( self.datamode == "train" ):
            mask = self.get_mask_image( self.df_train.iloc[index], image_height = 512, image_width = 512, n_channels = self.args.n_channels, n_classes = self.n_classes )
            #mask = self.get_mask_images( self.df_train.loc[image_namea], image_height = 512, image_width = 512, n_channels = self.args.n_channels, n_classes = self.n_classes )
            if( self.data_augument ):
                set_random_seed( self.seed_da )

            mask = self.transform(mask)

        #-------------
        # class id
        #-------------
        class_id = self.df_train.iloc[index]["ClassId"]

        if( self.datamode == "train" ):
            results_dict = {
                "image_name" : image_name,
                "image" : image,
                "mask" : mask,
                "class_id" : class_id,
            }
        else:
            results_dict = {
                "image_name" : image_name,
                "image" : image,
            }

        return results_dict


class ImaterialistDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, n_workers = 4, pin_memory = True):
        super(ImaterialistDataLoader, self).__init__()
        self.data_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size = batch_size, 
                shuffle = shuffle,
                num_workers = n_workers,
                pin_memory = pin_memory,
        )

        self.dataset = dataset
        self.batch_size = batch_size
        self.data_iter = self.data_loader.__iter__()

    def next_batch(self):
        try:
            batch = self.data_iter.__next__()
        except StopIteration:
            self.data_iter = self.data_loader.__iter__()
            batch = self.data_iter.__next__()

        return batch