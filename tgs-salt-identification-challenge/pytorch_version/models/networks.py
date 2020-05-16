# -*- coding:utf-8 -*-
import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision
from torchvision import models


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
    """
    UNet 構造での生成器
    """
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
		    nn.Tanh(),
#		    nn.Sigmoid(),
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


class UNet5( nn.Module ):
    def __init__(
        self,
        n_in_channels = 3, n_out_channels = 3, n_fmaps = 32,
    ):
        super( UNet5, self ).__init__()

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
        self.conv5 = conv_block( n_fmaps*8, n_fmaps*16 )
        self.pool5 = nn.MaxPool2d( kernel_size=2, stride=2, padding=0 )

        #
        self.bridge=conv_block( n_fmaps*16, n_fmaps*32 )

        # Decoder（アップサンプリング）
        self.dconv1 = dconv_block( n_fmaps*32, n_fmaps*16 )
        self.up1 = conv_block( n_fmaps*32, n_fmaps*16 )
        self.dconv2 = dconv_block( n_fmaps*16, n_fmaps*8 )
        self.up2 = conv_block( n_fmaps*16, n_fmaps*8 )
        self.dconv3 = dconv_block( n_fmaps*8, n_fmaps*4 )
        self.up3 = conv_block( n_fmaps*8, n_fmaps*4 )
        self.dconv4 = dconv_block( n_fmaps*4, n_fmaps*2 )
        self.up4 = conv_block( n_fmaps*4, n_fmaps*2 )
        self.dconv5 = dconv_block( n_fmaps*2, n_fmaps*1 )
        self.up5 = conv_block( n_fmaps*2, n_fmaps*1 )
        
        # 出力層
        self.out_layer = nn.Sequential(
		    nn.Conv2d( n_fmaps, n_out_channels, 3, 1, 1 ),
		    nn.Tanh(),
#		    nn.Sigmoid(),
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
        conv5 = self.conv4( pool4 )
        pool5 = self.pool4( conv5 )

        #
        bridge = self.bridge( pool5 )

        # Decoder（アップサンプリング）& skip connection
        dconv1 = self.dconv1(bridge)

        concat1 = torch.cat( [dconv1,conv5], dim=1 )
        up1 = self.up1(concat1)

        dconv2 = self.dconv2(up1)
        concat2 = torch.cat( [dconv2,conv4], dim=1 )

        up2 = self.up2(concat2)
        dconv3 = self.dconv3(up2)
        concat3 = torch.cat( [dconv3,conv3], dim=1 )

        up3 = self.up3(concat3)
        dconv4 = self.dconv4(up3)
        concat4 = torch.cat( [dconv4,conv2], dim=1 )

        up4 = self.up4(concat4)
        dconv5 = self.dconv5(up4)
        concat5 = torch.cat( [dconv5,conv1], dim=1 )

        up5 = self.up5(concat5)

        # 出力層
        output = self.out_layer( up5 )

        return output


#====================================
# UNet BottleNeck
#====================================
class UNet4BottleNeck( nn.Module ):
    """
    UNet のボトルネック部分で depth を concat する生成器
    """
    def __init__(
        self,
        n_in_channels = 3, n_out_channels = 3, n_fmaps = 64,
    ):
        super( UNet4BottleNeck, self ).__init__()

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
        self.bridge=conv_block( n_fmaps*8+1, n_fmaps*16 )

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
		    nn.Tanh(),
#		    nn.Sigmoid(),
		)
        return

    def forward( self, input, depth ):
        # Encoder（ダウンサンプリング）
        conv1 = self.conv1( input )
        pool1 = self.pool1( conv1 )
        conv2 = self.conv2( pool1 )
        pool2 = self.pool2( conv2 )
        conv3 = self.conv3( pool2 )
        pool3 = self.pool3( conv3 )
        conv4 = self.conv4( pool3 )
        pool4 = self.pool4( conv4 )

        # ボトルネック部分で concat
        depth = depth.expand(depth.shape[0], depth.shape[1], pool4.shape[2], pool4.shape[3] )
        concat = torch.cat( [pool4, depth], dim=1)
        bridge = self.bridge( concat )

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
# MG-VTON
#====================================
class MGVTONResGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, padding_type='zero', affine=True):
        assert (n_blocks >= 0)
        super(MGVTONResGenerator, self).__init__()
        activation = nn.ReLU(True)

        p = 0
        if padding_type == 'reflect':
            model = [nn.ReflectionPad2d(3)]
        elif padding_type == 'replicate':
            model = [nn.ReplicationPad2d(3)]
        elif padding_type == 'zero':
            p = 3
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        model = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=p), nn.BatchNorm2d(ngf, affine=affine), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      nn.BatchNorm2d(ngf * mult * 2, affine=affine), activation]

        ### resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, affine=affine)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                      nn.BatchNorm2d(int(ngf * mult / 2),affine=affine), activation]
        
        p = 0
        if padding_type == 'reflect':
            model += [nn.ReflectionPad2d(3)]
        elif padding_type == 'replicate':
            model += [nn.ReplicationPad2d(3)]
        elif padding_type == 'zero':
            p = 3
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=p), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, activation=nn.ReLU(True), affine=True, use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, activation, affine, use_dropout)

    def build_conv_block(self, dim, padding_type, activation, affine, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       nn.BatchNorm2d(dim,affine=affine),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       nn.BatchNorm2d(dim,affine=affine)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        #print( "[ResnetBlock] x.size() :", x.size() )
        out = x + self.conv_block(x)
        return out

#------------------------------------
# GANimation の生成器
#------------------------------------
class NetworkBase(nn.Module):
    def __init__(self):
        super(NetworkBase, self).__init__()
        self._name = 'BaseNetwork'

    @property
    def name(self):
        return self._name

    def init_weights(self):
        self.apply(self._weights_init_fn)

    def _weights_init_fn(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0)
        elif classname.find('BatchNorm2d') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _get_norm_layer(self, norm_type='batch'):
        if norm_type == 'batch':
            norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
        elif norm_type == 'instance':
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
        elif norm_type =='batchnorm2d':
            norm_layer = nn.BatchNorm2d
        else:
            raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

        return norm_layer

class GANimationGenerator(NetworkBase):
    """Generator. Encoder-Decoder Architecture."""
    def __init__(self, input_nc, output_nc, conv_dim=64, repeat_num=6):
        super(GANimationGenerator, self).__init__()
        self._name = 'generator_wgan'

        layers = []
        layers.append(nn.Conv2d(input_nc, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.InstanceNorm2d(conv_dim, affine=True))
        layers.append(nn.ReLU(inplace=True))

        # Down-Sampling
        curr_dim = conv_dim
        for i in range(2):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim*2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim * 2

        # Bottleneck
        for i in range(repeat_num):
            layers.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))

        # Up-Sampling
        for i in range(2):
            layers.append(nn.ConvTranspose2d(curr_dim, curr_dim//2, kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.InstanceNorm2d(curr_dim//2, affine=True))
            layers.append(nn.ReLU(inplace=True))
            curr_dim = curr_dim // 2

        self.main = nn.Sequential(*layers)

        layers = []
        layers.append(nn.Conv2d(curr_dim, output_nc, kernel_size=7, stride=1, padding=3, bias=False))
        layers.append(nn.Tanh())
        self.img_reg = nn.Sequential(*layers)
        return

    def forward(self, x ):
        # replicate spatially and concatenate domain information
        #cloth_size = cloth_size.unsqueeze(2).unsqueeze(3)
        #cloth_size = cloth_size.expand(cloth_size.size(0), cloth_size.size(1), x.size(2), x.size(3))
        #pose_size = pose_size.unsqueeze(2).unsqueeze(3)
        #pose_size = pose_size.expand(pose_size.size(0), pose_size.size(1), x.size(2), x.size(3))
        #x = torch.cat([x, cloth_size, pose_size], dim=1)

        features = self.main(x)
        return self.img_reg(features)

class ResidualBlock(nn.Module):
    """Residual Block."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(dim_out, affine=True))

    def forward(self, x):
        return x + self.main(x)


#====================================
# 識別器
#====================================
class PatchGANDiscriminator( nn.Module ):
    """
    PatchGAN の識別器
    """
    def __init__(
        self,
        n_in_channels = 3,
        n_fmaps = 32
    ):
        super( PatchGANDiscriminator, self ).__init__()

        # 識別器のネットワークでは、Patch GAN を採用するが、
        # patchを切り出したり、ストライドするような処理は、直接的には行わない
        # その代りに、これを畳み込みで表現する。
        # つまり、CNNを畳み込んで得られる特徴マップのある1pixelは、入力画像のある領域(Receptive field)の影響を受けた値になるが、
        # 裏を返せば、ある1pixelに影響を与えられるのは、入力画像のある領域だけ。
        # そのため、「最終出力をあるサイズをもった特徴マップにして、各pixelにて真偽判定をする」ことと 、「入力画像をpatchにして、各patchの出力で真偽判定をする」ということが等価になるためである。
        def discriminator_block1( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride=2, padding=1 ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        def discriminator_block2( in_dim, out_dim ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride=2, padding=1 ),
                nn.InstanceNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        #self.layer1 = discriminator_block1( n_in_channels * 2, n_fmaps )
        self.layer1 = discriminator_block1( n_in_channels, n_fmaps )
        self.layer2 = discriminator_block2( n_fmaps, n_fmaps*2 )
        self.layer3 = discriminator_block2( n_fmaps*2, n_fmaps*4 )
        self.layer4 = discriminator_block2( n_fmaps*4, n_fmaps*8 )

        self.output_layer = nn.Sequential(
            nn.ZeroPad2d( (1, 0, 1, 0) ),
            nn.Conv2d( n_fmaps*8, 1, 4, padding=1, bias=False )
        )

    def forward(self, input ):
        #output = torch.cat( [x, y], dim=1 )
        output = self.layer1( input )
        output = self.layer2( output )
        output = self.layer3( output )
        output = self.layer4( output )
        output = self.output_layer( output )
        output = output.view(-1)
        return output


class MultiscaleDiscriminator(nn.Module):
    """
    Pix2Pix-HD のマルチスケール識別器
    """
    def __init__(
        self,
        n_in_channels = 3,
        n_fmaps = 64,
        n_dis = 3,                # 識別器の数
#        n_layers = 3,        
    ):
        super( MultiscaleDiscriminator, self ).__init__()
        self.n_dis = n_dis
        #self.n_layers = n_layers
        
        def discriminator_block1( in_dim, out_dim, stride, padding ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride, padding ),
                nn.LeakyReLU( 0.2, inplace=True ),
            )
            return model

        def discriminator_block2( in_dim, out_dim, stride, padding ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride, padding ),
                nn.InstanceNorm2d( out_dim ),
                nn.LeakyReLU( 0.2, inplace=True )
            )
            return model

        def discriminator_block3( in_dim, out_dim, stride, padding ):
            model = nn.Sequential(
                nn.Conv2d( in_dim, out_dim, 4, stride, padding ),
            )
            return model

        # マルチスケール識別器で、入力画像を 1/2 スケールにする層
        self.downsample_layer = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

        # setattr() を用いて self オブジェクトを動的に生成することで、各 Sequential ブロックに名前をつける
        for i in range(self.n_dis):
            setattr( self, 'scale'+str(i)+'_layer0', discriminator_block1( n_in_channels, n_fmaps, 2, 2) )
            setattr( self, 'scale'+str(i)+'_layer1', discriminator_block2( n_fmaps, n_fmaps*2, 2, 2) )
            setattr( self, 'scale'+str(i)+'_layer2', discriminator_block2( n_fmaps*2, n_fmaps*4, 2, 2) )
            setattr( self, 'scale'+str(i)+'_layer3', discriminator_block2( n_fmaps*4, n_fmaps*8, 1, 2) )
            setattr( self, 'scale'+str(i)+'_layer4', discriminator_block3( n_fmaps*8, 1, 1, 2) )

        """
        # この方法だと、各 Sequential ブロックに名前をつけられない（連番になる）
        self.layers = nn.ModuleList()
        for i in range(self.n_dis):
            self.layers.append( discriminator_block1( n_in_channels*2, n_fmaps, 2, 2) )
            self.layers.append( discriminator_block2( n_fmaps, n_fmaps*2, 2, 2) )
            self.layers.append( scdiscriminator_block2( n_fmaps*2, n_fmaps*4, 2, 2)ale_layer )
            self.layers.append( discriminator_block2( n_fmaps*4, n_fmaps*8, 1, 2) )
            self.layers.append( discriminator_block3( n_fmaps*8, 1, 1, 2) )
        """
        return

    def forward(self, input ):
        """
        [Args]
            input : 入力画像 <torch.Float32> shape =[N,C,H,W]
        [Returns]
            outputs_allD : shape=[n_dis, n_layers=5, tensor=[N,C,H,W] ]
        """
        #input = torch.cat( [x, y], dim=1 )

        outputs_allD = []
        for i in range(self.n_dis):
            if i > 0:
                # 入力画像を 1/2 スケールにする
                input = self.downsample_layer(input)

            scale_layer0 = getattr( self, 'scale'+str(i)+'_layer0' )
            scale_layer1 = getattr( self, 'scale'+str(i)+'_layer1' )
            scale_layer2 = getattr( self, 'scale'+str(i)+'_layer2' )
            scale_layer3 = getattr( self, 'scale'+str(i)+'_layer3' )
            scale_layer4 = getattr( self, 'scale'+str(i)+'_layer4' )

            outputs_oneD = []
            outputs_oneD.append( scale_layer0(input) )
            outputs_oneD.append( scale_layer1(outputs_oneD[-1]) )
            outputs_oneD.append( scale_layer2(outputs_oneD[-1]) )
            outputs_oneD.append( scale_layer3(outputs_oneD[-1]) )
            outputs_oneD.append( scale_layer4(outputs_oneD[-1]) )
            outputs_allD.append( outputs_oneD )

        return outputs_allD

#------------------------------------
# GANimation の識別器
#------------------------------------
class GANimationDiscriminator(NetworkBase):
    """Discriminator. PatchGAN."""
    def __init__(self, n_in_channels = 3, n_fmaps=64, feat_dim=1, image_size=128, repeat_num=6):
        super(GANimationDiscriminator, self).__init__()
        self._name = 'discriminator_wgan'

        layers = []
        layers.append(nn.Conv2d(n_in_channels, n_fmaps, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01, inplace=True))

        curr_dim = n_fmaps
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            curr_dim = curr_dim * 2

        k_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(curr_dim, feat_dim, kernel_size=k_size, bias=False)

    def forward(self, x):
        h = self.main(x)
        out_real = self.conv1(h)
        out_feat = self.conv2(h)

        #print( "out_real.shape : ", out_real.shape )
        out_real = torch.mean(out_real, dim=2 )
        out_real = torch.mean(out_real, dim=2 )
        #print( "out_real.shape : ", out_real.shape )

        out_feat = torch.mean(out_feat, dim=2 )
        out_feat = torch.mean(out_feat, dim=2 )
        #print( "out_feat.shape : ", out_feat.shape )
        return out_real, out_feat
