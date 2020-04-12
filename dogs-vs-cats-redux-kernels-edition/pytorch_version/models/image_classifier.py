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
from utils import save_checkpoint, load_checkpoint


class ImageClassifierDNN( nn.Module ):
    """
    ディープラーニングベースの画像分類器
    """
    def __init__(self, device, network_type = "resnet50", n_classes = 2, pretrained = True, train_only_fc = True ):
        super( ImageClassifierDNN, self ).__init__()
        self.device = device
        self.network_type = network_type
        self.n_classes = n_classes
        self.pretrained = pretrained
        self.train_only_fc = train_only_fc

        # モデルの定義
        if( network_type == "my_resnet18" ):
            self.model = MyResNet18( n_in_channels = 3, n_fmaps = 64, n_classes = 2 ).to(device)
        elif( network_type == "resnet18" ):
            self.model = ResNet18( n_classes, pretrained, train_only_fc ).to(device)
        else:
            self.model = ResNet50( n_classes, pretrained, train_only_fc ).to(device)

        # optimizer の設定
        pass

        # loss 関数の設定
        pass

        return

    def load_check_point( self, load_checkpoints_path ):
        if not load_checkpoints_path == '' and os.path.exists(load_checkpoints_path):
            load_checkpoint(self.model, self.device, load_checkpoints_path )
            print( "load check points" )
        return

    def set_optimizer(self):
        # 未実装
        return

    def set_loss_fn(self):
        # 未実装
        return

    def fit( self, X_train, y_train ):
        """
        [args]
            X_train : <tensor> 学習用画像
            X_train : <tensor> ラベル値
        """
        self.model.train()

        # 未実装
        return

    def predict( self, X_test ):
        self.model.eval()

        # データをモデルに流し込む
        with torch.no_grad():
            output = self.model( X_test )

        # 確率値が最大のラベル 0~9 を予想ラベルとする。
        _, predict = torch.max( output.data, dim = 1 )
        predict = predict.detach().cpu().numpy()
        return predict


    def predict_proba( self, X_test ):
        """
        [args]
            X_test : <tensor> テスト用画像 / shape=(N,C,H,W)
        """
        self.model.eval()

        # データをモデルに流し込む
        with torch.no_grad():
            output = self.model( X_test )

        # 確率値で出力
        predict_proba = F.softmax(output, dim=1)[:, 1]
        predict_proba = predict_proba.detach().cpu().numpy()
        return predict_proba
