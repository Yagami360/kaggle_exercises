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


