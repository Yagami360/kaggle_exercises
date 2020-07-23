# -*- coding:utf-8 -*-
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import models
from torchvision.models import resnet34, resnet50

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#====================================
# UNet
#====================================
class UNet4( nn.Module ):
    def __init__(
        self,
        n_in_channels = 3, n_out_channels = 3, n_fmaps = 64,
    ):
        super( UNet4, self ).__init__()

        def conv_block( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                nn.BatchNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True ),

                nn.Conv2d( out_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                nn.BatchNorm2d( out_dim ),
            )
            return model

        def dconv_block( in_dim, out_dim ):
            model = nn.Sequential(
                nn.ConvTranspose2d( in_dim, out_dim, kernel_size=3, stride=2, padding=1,output_padding=1 ),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU( 0.2, inplace=True ),
            )
            return model

        # Encoder（ダウンサンプリング）
        self.conv1 = conv_block( n_in_channels, n_fmaps )
        self.pool1 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
        self.conv2 = conv_block( n_fmaps*1, n_fmaps*2 )
        self.pool2 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
        self.conv3 = conv_block( n_fmaps*2, n_fmaps*4 )
        self.pool3 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )
        self.conv4 = conv_block( n_fmaps*4, n_fmaps*8 )
        self.pool4 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )

        #
        self.bridge=conv_block( n_fmaps*8, n_fmaps*16 )

        # Decoder（アップサンプリング）
        self.dconv1 = dconv_block( n_fmaps*16, n_fmaps*8 )
        self.up1 = conv_block( n_fmaps*16, n_fmaps*8 )
        self.dconv2 = dconv_block( n_fmaps*8, n_fmaps*4 )
        self.up2 = conv_block( n_fmaps*8, n_fmaps*4 )
        self.dconv3 = dconv_block( n_fmaps*4, n_fmaps*2 )
        self.up3 = conv_block( n_fmaps*4, n_fmaps*2 )
        self.dconv4 = dconv_block( n_fmaps*2, n_fmaps*1 )
        self.up4 = conv_block( n_fmaps*2, n_fmaps*1 )

        # 出力層
        self.out_layer = nn.Sequential(
		    nn.Conv2d( n_fmaps, n_out_channels, 3, 1, 1 ),
		)
        return

    def forward( self, input ):
        # Encoder（ダウンサンプリング）
        conv1 = self.conv1( input )
        pool1 = self.pool1( conv1 )
        conv2 = self.conv2( pool1 )
        pool2 = self.pool2( conv2 )
        conv3 = self.conv3( pool2 )
        pool3 = self.pool3( conv3 )
        conv4 = self.conv4( pool3 )
        pool4 = self.pool4( conv4 )

        #
        bridge = self.bridge( pool4 )

        # Decoder（アップサンプリング）& skip connection
        dconv1 = self.dconv1(bridge)

        concat1 = torch.cat( [dconv1,conv4], dim=1 )
        up1 = self.up1(concat1)

        dconv2 = self.dconv2(up1)
        concat2 = torch.cat( [dconv2,conv3], dim=1 )

        up2 = self.up2(concat2)
        dconv3 = self.dconv3(up2)
        concat3 = torch.cat( [dconv3,conv2], dim=1 )

        up3 = self.up3(concat3)
        dconv4 = self.dconv4(up3)
        concat4 = torch.cat( [dconv4,conv1], dim=1 )

        up4 = self.up4(concat4)

        # 出力層
        output = self.out_layer( up4 )
        return output

#====================================
# UNet with Resnet
#====================================
class UNet4ResNet34( nn.Module ):
    def __init__(
        self,
        n_in_channels = 3, n_out_channels = 3, n_fmaps = 64,
        pretrained = True,
    ):
        super( UNet4ResNet34, self ).__init__()

        def conv_block( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                nn.BatchNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True ),

                nn.Conv2d( out_dim, out_dim, kernel_size=3, stride=1, padding=1 ),
                nn.BatchNorm2d( out_dim ),
            )
            return model

        def dconv_block( in_dim, out_dim ):
            model = nn.Sequential(
                nn.ConvTranspose2d( in_dim, out_dim, kernel_size=3, stride=2, padding=1,output_padding=1 ),
                nn.BatchNorm2d(out_dim),
                nn.LeakyReLU( 0.2, inplace=True ),
            )
            return model

        self.resnet = resnet34( pretrained )
        self.resnet.conv1 = nn.Conv2d( n_in_channels, n_fmaps, kernel_size=7, stride=2, padding=3, bias=False )
        #print( self.resnet )

        self.conv0 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu
        )

        # Encoder（ダウンサンプリング）
        self.conv1 = nn.Sequential( self.resnet.layer1 )
        self.conv2 = nn.Sequential( self.resnet.layer2 )
        self.conv3 = nn.Sequential( self.resnet.layer3 )
        self.conv4 = nn.Sequential( self.resnet.layer4 )

        #
        self.bridge = conv_block( n_fmaps*8, n_fmaps*16 )
        self.bridge_pool = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )

        # Decoder（アップサンプリング）
        self.dconv1 = dconv_block( n_fmaps*16, n_fmaps*8 )
        self.up1 = conv_block( n_fmaps*16, n_fmaps*8 )
        self.dconv2 = dconv_block( n_fmaps*8, n_fmaps*4 )
        self.up2 = conv_block( n_fmaps*8, n_fmaps*4 )
        self.dconv3 = dconv_block( n_fmaps*4, n_fmaps*2 )
        self.up3 = conv_block( n_fmaps*4, n_fmaps*2 )
        self.dconv4 = dconv_block( n_fmaps*2, n_fmaps*1 )
        self.up4 = conv_block( n_fmaps*2, n_fmaps*1 )

        self.dconv0 = dconv_block( n_fmaps*1, n_fmaps*1 )

        # 出力層
        self.out_layer = nn.Sequential(
		    nn.Conv2d( n_fmaps, n_out_channels, 3, 1, 1 ),
		)
        return

    def forward( self, input ):
        """
        conv1.shape :  torch.Size([32, 32, 128, 128])
        pool1.shape :  torch.Size([32, 32, 64, 64])
        conv4.shape :  torch.Size([32, 256, 16, 16])
        pool4.shape :  torch.Size([32, 256, 8, 8])
        dconv1.shape :  torch.Size([32, 256, 16, 16])
        """
        conv0 = self.conv0( input )
        #print( "conv0.shape : ", conv0.shape )

        # Encoder（ダウンサンプリング）
        conv1 = self.conv1( conv0 )
        #print( "conv1.shape : ", conv1.shape )

        conv2 = self.conv2( conv1 )
        #print( "conv2.shape : ", conv2.shape )

        conv3 = self.conv3( conv2 )
        #print( "conv3.shape : ", conv3.shape )

        conv4 = self.conv4( conv3 )
        #print( "conv4.shape : ", conv4.shape )

        #
        bridge = self.bridge( conv4 )
        bridge = self.bridge_pool(bridge)
        #print( "bridge.shape : ", bridge.shape )

        # Decoder（アップサンプリング）& skip connection
        dconv1 = self.dconv1(bridge)
        #print( "dconv1.shape : ", dconv1.shape )
        concat1 = torch.cat( [dconv1,conv4], dim=1 )
        up1 = self.up1(concat1)
        #print( "up1.shape : ", up1.shape )

        dconv2 = self.dconv2(up1)
        #print( "dconv2.shape : ", dconv2.shape )
        concat2 = torch.cat( [dconv2,conv3], dim=1 )
        up2 = self.up2(concat2)
        #print( "up2.shape : ", up2.shape )

        dconv3 = self.dconv3(up2)
        #print( "dconv3.shape : ", dconv3.shape )
        concat3 = torch.cat( [dconv3,conv2], dim=1 )
        up3 = self.up3(concat3)
        #print( "up3.shape : ", up3.shape )

        dconv4 = self.dconv4(up3)
        #print( "dconv4.shape : ", dconv4.shape )
        concat4 = torch.cat( [dconv4,conv1], dim=1 )
        up4 = self.up4(concat4)
        #print( "up4.shape : ", up4.shape )

        dconv0 = self.dconv0(up4)
        #print( "dconv0.shape : ", dconv0.shape )

        # 出力層
        output = self.out_layer( dconv0 )
        return output

#====================================
# U-Net Baseline by PyTorch in FGVC6 resize
# https://www.kaggle.com/go1dfish/u-net-baseline-by-pytorch-in-fgvc6-resize
#====================================
class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

    
class UNetFGVC6(nn.Module):
    def __init__(self, n_channels=3, n_classes=3):
        super(UNetFGVC6, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)
        return

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

