import os
import numpy as np
#from apex import amp

# PyTorch
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

# 自作クラス
from networks import MyResNet18, ResNet18, ResNet50


class ResNetClassifier( nn.Module ):
    """
    ResNet の分類器
    """
    def __init__(self, device, network_type = "resnet50", n_classes = 2, pretrained = True ):
        if( network_type == "my_resnet18" ):
            self.model = MyResNet18( n_in_channels = 3, n_fmaps = 64, n_classes = 2 ).to(device)
        elif( network_type == "resnet18" ):
            self.model = ResNet18( n_classes, pretrained ).to(device)
        else:
            self.model = ResNet50( n_classes, pretrained ).to(device)

        return

    def fit( self, X_train, y_train ):
        return

    def predict( self, X_test ):
        return

    def predict_prob( self, X_test ):
        return
