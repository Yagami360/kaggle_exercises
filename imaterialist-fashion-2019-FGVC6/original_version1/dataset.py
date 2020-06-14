import os
from tqdm import tqdm
import numpy as np
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

from utils import set_random_seed

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)


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
        mean = [ 0.5 for i in range(args.n_channels) ]
        std = [ 0.5 for i in range(args.n_channels) ]
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

            self.transform_mask = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
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
            self.transform_mask = transforms.Compose(
                [
                    transforms.Resize( (args.image_height, args.image_width), interpolation=Image.NEAREST ),
                    transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                    transforms.ToTensor(),
                    transforms.Normalize( mean, std ),
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

    def get_mask_image(self, df_mask, n_channels = 1, n_classes = 92 ):
        #print( df_mask.head() )
        if( len(df_mask.shape) == 1 ):
            height = df_mask["Height"]
            width = df_mask["Width"]
        else:
            height = df_mask["Height"].iloc[0]
            width = df_mask["Width"].iloc[0]

        #print( "height={}, width={}".format(height, width) )
        mask_image = np.zeros(height*width, dtype=np.int32)
        if( len(df_mask.shape) == 1 ):
            encoded_pixels = df_mask["EncodedPixels"]
            class_id = df_mask["ClassId"]

            # encoded_pixels: 3655609 3 3658608 9 3661608 14 3664607 ...
            # pixels : [3655609, 3, 3658608, 9, 3661608, 14, ...]
            pixels = list(map(int, encoded_pixels.split(" ")))
            #print( "encoded_pixels={}".format(encoded_pixels) )
            #print( "pixels={}".format(pixels) )
            #print( "class_id={}".format(class_id) )
            for i in range(0,len(pixels), 2):
                start_pixel = pixels[i]-1
                len_mask = pixels[i+1]-1
                end_pixel = start_pixel + len_mask
                if int(class_id) < n_classes - 1:
                    mask_image[start_pixel:end_pixel] = int(class_id)
        else:
            for index in range(len(df_mask)):
                encoded_pixels = df_mask.iloc[index]["EncodedPixels"]
                class_id = df_mask.iloc[index]["ClassId"]
                pixels = list(map(int, encoded_pixels.split(" ")))
                for i in range(0,len(pixels), 2):
                    start_pixel = pixels[i]-1
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
            mask = self.get_mask_image( self.df_train.loc[image_name], n_channels = self.args.n_channels, n_classes = self.n_classes )
            if( self.data_augument ):
                set_random_seed( self.seed_da )

            mask = self.transform_mask(mask)

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