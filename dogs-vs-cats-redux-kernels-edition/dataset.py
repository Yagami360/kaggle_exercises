# coding=utf-8
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import cv2

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.utils import save_image

IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif',
    '.JPG', '.JPEG', '.PNG', '.PPM', '.BMP', '.PGM', '.TIF',
)


class DogsVSCatsDataset(data.Dataset):
    """
    kaggle コンペ dogs-vs-cats 用データセットクラス
    """
    def __init__(self, args, root_dir, datamode = "train", debug = False ):
        super(DogsVSCatsDataset, self).__init__()
        self.args = args

        # データをロードした後に行う各種前処理の関数を構成を指定する。
        self.transform = transforms.Compose(
            [
                transforms.Resize( (args.image_height, args.image_width), interpolation=Image.LANCZOS ),
                transforms.CenterCrop( size = (args.image_height, args.image_width) ),
                transforms.ToTensor(),   # Tensor に変換
            ]
        )

        self.image_height = args.image_height
        self.image_width = args.image_width
        self.dataset_dir = os.path.join( root_dir, datamode )
        self.image_names = sorted( [f for f in os.listdir(self.dataset_dir) if f.endswith(IMG_EXTENSIONS)] )
        self.debug = debug
        if( self.debug ):
            print( "self.dataset_dir :", self.dataset_dir)
            print( "len(self.image_names) :", len(self.image_names))
            print( "self.image_names[0:5] :", self.image_names[0:5])

        return

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        raw_image = Image.open(os.path.join(self.dataset_dir, image_name)).convert('RGB')
        image = self.transform(raw_image)

        if( image_name[0:3] == "cat." ):
            #targets = torch.eye(2)[0].float()
            targets = torch.zeros(1).squeeze().long()
        elif( image_name[0:3] == "dog." ):
            #targets = torch.eye(2)[1].float()
            targets = torch.ones(1).squeeze().long()
        else:
            #targets = torch.eye(2)[0].float()
            targets = torch.zeros(1).squeeze().long()

        results_dict = {
            "image_name" : image_name,
            "image" : image,
            "targets" : targets,
        }
        return results_dict


class DogsVSCatsDataLoader(object):
    def __init__(self, dataset, batch_size = 1, shuffle = True, pin_memory = True):
        super(DogsVSCatsDataLoader, self).__init__()
        self.data_loader = torch.utils.data.DataLoader(
                dataset, 
                batch_size = batch_size, 
                shuffle = shuffle,
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