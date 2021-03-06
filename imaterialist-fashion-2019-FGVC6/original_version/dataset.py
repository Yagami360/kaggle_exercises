import os
from tqdm import tqdm
import numpy as np
import shutil
import random
import pandas as pd
import re
import math
from PIL import Image, ImageDraw, ImageOps
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# PyTorch
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.utils import save_image

from utils.utils import set_random_seed
from utils.utils import split_masks, concat_masks

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)

def save_masks( dataset_dir, save_dir, n_classes, image_height = 256, image_width = 192, resize = True ):
    df_train = pd.read_csv( os.path.join(dataset_dir, "train.csv"), index_col='ImageId' )
    df_mask = df_train.groupby('ImageId')['EncodedPixels', 'ClassId'].agg(lambda x: list(x))
    df_size = df_train.groupby('ImageId')['Height', 'Width'].mean()
    df_train = df_mask.join(df_size, on='ImageId')
    #print( "df_train.head() \n:", df_train.head())
    for i in tqdm(range(len(df_train)), desc="saving mask images"):
        image_name = df_train.index[i]
        height = df_train.iloc[i]["Height"]
        width = df_train.iloc[i]["Width"]
        #print( "image_name={}, height={}, width={}".format(image_name, height, width) )

        encoded_pixels = df_train.iloc[i]["EncodedPixels"]
        class_ids = df_train.iloc[i]["ClassId"]
        for encoded_pixel, class_id in zip(encoded_pixels, class_ids):    
            if int(class_id) < n_classes - 1:
                mask_image = np.zeros( (height*width), dtype=np.int32)
                split_pixel = list(map(int, encoded_pixel.split(" ")))
                for i in range(0,len(split_pixel), 2):
                    start_pixel = split_pixel[i]-1
                    len_mask = split_pixel[i+1]-1
                    end_pixel = start_pixel + len_mask
                    mask_image[start_pixel:end_pixel] = int(class_id)

                mask_image = mask_image.reshape((height, width), order='F')
                if( resize ):
                    mask_image = cv2.resize(mask_image, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite( os.path.join(save_dir, image_name.split(".jpg")[0] + "_c{}".format(class_id) + ".png"), mask_image )
    return


class ImaterialistDataset(data.Dataset):
    def __init__(self, args, dataset_dir, datamode = "train", image_height = 512, image_width = 512, n_classes = 92, data_augument = False, debug = False ):
        super(ImaterialistDataset, self).__init__()
        self.args = args
        self.dataset_dir = dataset_dir
        self.datamode = datamode
        self.image_height = image_height
        self.image_width = image_width
        self.n_classes = n_classes
        self.data_augument = data_augument
        self.debug = debug

        self.df_train = pd.read_csv( os.path.join(self.dataset_dir, "train.csv"), index_col='ImageId' )
        df_mask = self.df_train.groupby('ImageId')['EncodedPixels', 'ClassId'].agg(lambda x: list(x))
        df_size = self.df_train.groupby('ImageId')['Height', 'Width'].mean()
        self.df_train = df_mask.join(df_size, on='ImageId')

        self.image_names = sorted( [f for f in os.listdir( os.path.join(self.dataset_dir, self.datamode)) if f.endswith(IMG_EXTENSIONS)] )

        # test データに対する file name, image height, image width のデータフレーム
        image_heights = []
        image_widths = []
        for image_name in self.image_names:
            image = Image.open(os.path.join(self.dataset_dir, self.datamode, image_name))
            image_heights.append( image.height )
            image_widths.append( image.width )

        self.df_test = pd.DataFrame( {'Height':image_heights, 'Width':image_widths}, index = self.image_names )
        self.df_test.index.names = ['ImageId']

        # transform
        mean = [ 0.5 for i in range(args.n_in_channels) ]
        std = [ 0.5 for i in range(args.n_in_channels) ]
        if( data_augument ):
            self.transform = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
#                    transforms.RandomResizedCrop( (args.image_height, args.image_width) ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.BICUBIC ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( mean, std ),
                ]
            )

            self.transform_mask = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
#                    transforms.RandomResizedCrop( (args.image_height, args.image_width) ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.BICUBIC ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5], [0.5] ),
                ]
            )
            
            self.transform_mask_woToTernsor = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
#                    transforms.RandomResizedCrop( (args.image_height, args.image_width) ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.BICUBIC ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
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

            self.transform_mask = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( [0.5], [0.5] ),
                ]
            )

            self.transform_mask_woToTernsor = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                ]
            )


        if( self.debug ):
            print( "self.dataset_dir :", self.dataset_dir)
            print( "len(self.image_names) :", len(self.image_names))
            print( "self.df_train.head() \n:", self.df_train.head())
            print( "self.df_test.head() \n:", self.df_test.head())

        return

    def __len__(self):
        return len(self.image_names)

    def get_mask_image(self, df_mask, n_channels, n_classes ):
        #print( df_mask.head() )
        height = df_mask["Height"]
        width = df_mask["Width"]
        encoded_pixels = df_mask["EncodedPixels"]
        class_ids = df_mask["ClassId"]
        #print( "encoded_pixels={}".format(encoded_pixels) )
        #print( "class_ids={}".format(class_ids) )

        mask_image = np.zeros( (height*width, n_classes), dtype=np.int32)
        for encoded_pixel, class_id in zip(encoded_pixels, class_ids):
            #print( "encoded_pixel={}".format(encoded_pixel) )
            # encoded_pixel : 3655609 3 3658608 9 3661608 14 3664607 ...
            # split_pixel : [3655609, 3, 3658608, 9, 3661608, 14, ...]
            split_pixel = list(map(int, encoded_pixel.split(" ")))
            #print( "split_pixel={}".format(split_pixel) )
            #print( "class_id={}".format(class_id) )
            for i in range(0,len(split_pixel), 2):
                start_pixel = split_pixel[i]-1
                len_mask = split_pixel[i+1]-1
                end_pixel = start_pixel + len_mask

                #print( "mask_image.shape={}".format(mask_image.shape) )
                #print( "start_pixel={}".format(start_pixel) )
                #print( "end_pixel={}".format(end_pixel) )
                if int(class_id) < n_classes - 1:
                    mask_image[start_pixel:end_pixel, int(class_id)] = int(class_id)

        mask_image = mask_image.reshape((height, width, n_classes), order='F')
        return mask_image

    def get_mask_image_from_dir(self, df_mask, n_channels, n_classes, load_mask_dir, image_name ):
        class_ids = df_mask["ClassId"]
        mask_image = np.zeros( (self.image_height, self.image_width, n_classes), dtype=np.int32)
        for class_id in zip(class_ids):
            class_id = int(class_id[0])
            if class_id < n_classes - 1:
                image_np = cv2.imread(os.path.join(load_mask_dir, image_name.split(".jpg")[0] + "_c{}".format(str(class_id)) + ".png"), cv2.IMREAD_GRAYSCALE)
                mask_image[:, :, class_id] = image_np

        return mask_image

    def __getitem__(self, index):
        #print( "index : ", index )
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
            if( self.args.load_masks_from_dir ):
                mask_split_np = self.get_mask_image_from_dir( self.df_train.loc[image_name], n_channels = self.args.n_in_channels, n_classes = self.n_classes, load_mask_dir = os.path.join(self.dataset_dir, "train_masks"), image_name = image_name )
            else:
                mask_split_np = self.get_mask_image( self.df_train.loc[image_name], n_channels = self.args.n_in_channels, n_classes = self.n_classes )

            # １枚の画像中に複数のラベル値があるマスク画（int 型）/ 0 ~ n_classes
            mask_np = concat_masks( mask_split_np, n_classes = self.n_classes )
            #print( "mask_np.shape : ", mask_np.shape )
            #print( "min(mask_np)={}, max(mask_np)={}".format(np.min(mask_np), np.max(mask_np)) )

            # １枚の画像中に複数のラベル値があるマスク画像（int 型）/ 0 ~ n_classes
            mask = torch.from_numpy( np.asarray(self.transform_mask_woToTernsor( Image.fromarray(mask_np) )) ).float()
            #print( "mask.shape : ", mask.shape )
            #print( "torch.min(mask)={}, torch.max(mask)={}".format(torch.min(mask), torch.max(mask)) )

        if( self.datamode == "train" ):
            results_dict = {
                "image_name" : image_name,
                "image" : image,
                "mask" : mask,
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