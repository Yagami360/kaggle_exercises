# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50

#====================================
# ResNet-18
#====================================
class BasicBlock( nn.Module ):
    """
    """
    def __init__( 
        self,
        n_in_channels = 3,
        n_out_channels = 3,
        stride = 1,
    ):
        """
        [Args]
            n_in_channels : <int> 入力画像のチャンネル数
            n_out_channels : <int> 出力画像のチャンネル数
            stride : <int>
        """
        super( BasicBlock, self ).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d( n_in_channels, n_out_channels, kernel_size=3, stride=stride, padding=1 ),
            nn.BatchNorm2d( n_out_channels ),
            nn.LeakyReLU( 0.2, inplace=True ),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d( n_out_channels, n_out_channels, kernel_size=3, stride=1, padding=1 ),
            nn.BatchNorm2d( n_out_channels ),
        )

        # shortcut connection は、恒等写像
        self.shortcut_connections = nn.Sequential()

        # 入出力次元が異なる場合は、ゼロパディングで、次元の不一致箇所を０で埋める。
        if( n_in_channels != n_out_channels ):
            self.shortcut_connections = nn.Sequential(
                nn.Conv2d( n_in_channels, n_out_channels, kernel_size=1, stride=stride, padding=0,bias=False),
                nn.BatchNorm2d( n_out_channels )
            )

        return

    def forward( self, x ):
        out = self.layer1(x)
        out = self.layer2(out)

        # shortcut connection からの経路を加算
        out += self.shortcut_connections(x)
        return out


class MyResNet18( nn.Module ):
    def __init__( 
        self,
        n_in_channels = 3,
        n_fmaps = 64,
        n_classes = 10
    ):
        super( MyResNet18, self ).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d( n_in_channels, n_fmaps, kernel_size=7, stride=2, padding=3 ),
            nn.BatchNorm2d( n_fmaps ),
            nn.LeakyReLU( 0.2, inplace=True ),
            nn.MaxPool2d( kernel_size=3, stride=2, padding=1 )
        )
        
        self.layer1 = nn.Sequential(
                BasicBlock(
                    n_in_channels = n_fmaps, n_out_channels = n_fmaps, stride = 1
                ),
                BasicBlock(
                    n_in_channels = n_fmaps, n_out_channels = n_fmaps, stride = 1
                ),          
        )

        self.layer2 = nn.Sequential(
                BasicBlock(
                    n_in_channels = n_fmaps, n_out_channels = n_fmaps*2, stride = 2
                ),
                BasicBlock(
                    n_in_channels = n_fmaps*2, n_out_channels = n_fmaps*2, stride = 1
                ),          
        )

        self.layer3 = nn.Sequential(
                BasicBlock(
                    n_in_channels = n_fmaps*2, n_out_channels = n_fmaps*4, stride = 2
                ),
                BasicBlock(
                    n_in_channels = n_fmaps*4, n_out_channels = n_fmaps*4, stride = 1
                ),          
        )

        self.layer4 = nn.Sequential(
                BasicBlock(
                    n_in_channels = n_fmaps*4, n_out_channels = n_fmaps*8, stride = 2
                ),
                BasicBlock(
                    n_in_channels = n_fmaps*8, n_out_channels = n_fmaps*8, stride = 1
                ),          
        )

        self.avgpool = nn.AvgPool2d( 7, stride=1 )
        self.fc_layer = nn.Linear( n_fmaps*8, n_classes )
        return

    def forward( self, x ):
        out = self.layer0(x)    # 224x224
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # 7x7
        #out = torch.squeeze(out)
        out = self.avgpool(out) # 1x1
        out = out.view( out.size(0), -1 )
        out = self.fc_layer(out)
        return out

#====================================
# pretrained ResNet-18
#====================================
class ResNet18( nn.Module ):
    def __init__( 
        self,
        n_classes = 2,
        pretrained = True,
        train_only_fc = False,
    ):
        super( ResNet18, self ).__init__()
        self.resnet18 = resnet18( pretrained )

        # 事前学習済み resnet の出力層の n_classes 数を変更
        self.resnet18.fc = nn.Linear( 512, n_classes )

        # 主力層のみ学習対象にする
        self.train_only_fc = train_only_fc
        if( self.train_only_fc ):
            for name, param in self.resnet50.named_parameters():
                if name in ['fc.weight', 'fc.bias']:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        return

    def parameters(self):
        params_to_train = []
        if( self.train_only_fc ):
            for name, param in self.resnet50.named_parameters():
                if name in ['fc.weight', 'fc.bias']:
                    params_to_train.append(param)

        else:
            for name, param in self.resnet50.named_parameters():
                params_to_train.append(param)

        return params_to_train
        
    def forward( self, x ):
        out = self.resnet18(x)
        return out

#====================================
# pretrained ResNet-50
#====================================
class ResNet50( nn.Module ):
    def __init__( 
        self,
        n_classes = 2,
        pretrained = True,
        train_only_fc = False,
    ):
        super( ResNet50, self ).__init__()
        self.resnet50 = resnet50( pretrained )

        # 事前学習済み resnet の出力層の n_classes 数を変更
        self.resnet50.fc = nn.Linear( 2048, n_classes )

        # 主力層のみ学習対象にする
        self.train_only_fc = train_only_fc
        if( self.train_only_fc ):
            for name, param in self.resnet50.named_parameters():
                if name in ['fc.weight', 'fc.bias']:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        return

    def parameters(self):
        params_to_train = []
        if( self.train_only_fc ):
            for name, param in self.resnet50.named_parameters():
                if name in ['fc.weight', 'fc.bias']:
                    params_to_train.append(param)

        else:
            for name, param in self.resnet50.named_parameters():
                params_to_train.append(param)

        return params_to_train

    def forward( self, x ):
        out = self.resnet50(x)
        return out
