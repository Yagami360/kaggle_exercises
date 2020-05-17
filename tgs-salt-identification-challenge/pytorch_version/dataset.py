import os
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
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_LANCZOS4 )                # shape = [H,W,C]
        #img = resize(img, (image_height, image_width), mode='constant', preserve_range=True)
        X_train_img[i] = img.reshape( (image_height, image_width, n_channels))

    # y_train（マスク画像）
    y_train_mask = np.zeros( (len(train_image_names), image_height, image_width, 1), dtype=np.float32 )
    for i, name in enumerate(train_image_names):
        img = cv2.imread( os.path.join( dataset_dir, "train", "masks", name ), cv2.IMREAD_GRAYSCALE ) / 255     # 0.0f ~ 1.0f
        img = cv2.resize( img, (image_height, image_width), interpolation = cv2.INTER_NEAREST )                 # shape = [H,W,C]
        #img = resize(img, (image_height, image_width), mode='constant', preserve_range=True)
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


class TGSSaltDataset(data.Dataset):
    def __init__(self, args, root_dir, datamode = "train", data_augument = False, debug = False ):
        super(TGSSaltDataset, self).__init__()
        self.args = args
        self.datamode = datamode
        self.data_augument = data_augument
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.debug = debug
        self.dataset_dir = os.path.join( root_dir, datamode )
        self.image_names = sorted( [f for f in os.listdir( os.path.join(self.dataset_dir, "images")) if f.endswith(IMG_EXTENSIONS)] )

        mean = [ 0.5 for i in range(args.n_channels) ]
        std = [ 0.5 for i in range(args.n_channels) ]

        # transform
        if( data_augument ):
            self.transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop( (args.image_height, args.image_width) ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
#                    transforms.RandomAffine( degrees = (-10,10),  translate=(0.0, 0.0), scale = (1.00,1.00), resample=Image.BICUBIC ),
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

        #
        self.df_train = pd.read_csv( os.path.join(root_dir, "train.csv"), index_col='id' )

        # depth
        self.df_depth = pd.read_csv( os.path.join(root_dir, "depths.csv"), index_col='id' )
        scaler = StandardScaler()
        scaler.fit( self.df_depth["z"].values.reshape(-1,1) )
        self.df_depth["z"] = scaler.transform( self.df_depth["z"].values.reshape(-1,1) )

        if( self.debug ):
            print( "self.dataset_dir :", self.dataset_dir)
            print( "len(self.image_names) :", len(self.image_names))
            print( "self.image_names[0:5] :", self.image_names[0:5])
            print( "self.df_train.head() \n:", self.df_train.head())
            print( "self.df_depth[0:5] \n:", self.df_depth[0:5] )

        return

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        
        # image
        if( self.args.n_channels == 1 ):
            image = Image.open(os.path.join(self.dataset_dir, "images", image_name)).convert('L')
        else:
            image = Image.open(os.path.join(self.dataset_dir, "images", image_name)).convert('RGB')

        self.seed_da = random.randint(0,10000)
        if( self.data_augument ):
            set_random_seed( self.seed_da )

        image = self.transform(image)

        # mask
        if( self.datamode == "train" ):
            if( self.args.n_channels == 1 ):
                mask = Image.open(os.path.join(self.dataset_dir, "masks", image_name)).convert('L')
            else:
                mask = Image.open(os.path.join(self.dataset_dir, "masks", image_name)).convert('RGB')

            if( self.data_augument ):
                set_random_seed( self.seed_da )

            mask = self.transform(mask)

        # depth
        depth = np.zeros( (1, 1, 1, 1) )
        depth[:,0] = np.array( self.df_depth.loc[image_name.split(".png")[0],"z"] )
        depth = torch.from_numpy( depth[:,0] ).float()

        if( self.datamode == "train" ):
            results_dict = {
                "image_name" : image_name,
                "image" : image,
                "mask" : mask,
                "depth" : depth,
            }
        else:
            results_dict = {
                "image_name" : image_name,
                "image" : image,
                "depth" : depth,
            }

        return results_dict


class TGSSaltDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, n_workers = 4, pin_memory = True):
        super(TGSSaltDataLoader, self).__init__()
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