# -*- coding:utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

#=============================================
# パース画像用クロスエントロピー loss
#=============================================
class ParsingCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(ParsingCrossEntropyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        return

    def forward(self, input, target):
        # input
        input = input.transpose(0, 1)
        c = input.size()[0]
        n = input.size()[1] * input.size()[2] * input.size()[3]
        input = input.contiguous().view(c, n)
        input = input.transpose(0, 1)

        # target
        [_, argmax] = target.max(dim=1)
        target = argmax.view(n)

        return self.loss_fn(input, target)

#=============================================
# VGG loss
#=============================================
class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out

class VGGLoss(nn.Module):
    def __init__(self, device, layids = None,  ):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().to(device)
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.layids = layids

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.layids is None:
            self.layids = list(range(len(x_vgg)))
        for i in self.layids:
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


#============================================
# GAN Adv loss
#============================================
class VanillaGANLoss(nn.Module):
    def __init__(self, w_sigmoid_D = True ):
        super(VanillaGANLoss, self).__init__()
        # when use sigmoid in Discriminator
        if( w_sigmoid_D ):
            self.loss_fn = nn.BCELoss()            
        # when not use sigmoid in Discriminator
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()
        return

    def forward_D(self, d_real, d_fake):
        real_ones_tsr = torch.ones( d_real.shape ).cuda()
        fake_zeros_tsr = torch.zeros( d_fake.shape ).cuda()
        loss_D_real = self.loss_fn( d_real, real_ones_tsr )
        loss_D_fake = self.loss_fn( d_fake, fake_zeros_tsr )
        loss_D = loss_D_real + loss_D_fake

        return loss_D, loss_D_real, loss_D_fake

    def forward_G(self, d_fake):
        real_ones_tsr =  torch.ones( d_fake.shape ).cuda()
        loss_G = self.loss_fn( d_fake, real_ones_tsr )
        return loss_G

    def forward(self, d_real, d_fake, dis_or_gen = True ):
        # Discriminator 用の loss
        if dis_or_gen:
            loss, _, _ = self.forward_D( d_real, d_fake )
        # Generator 用の loss
        else:
            loss = self.forward_G( d_fake )

        return loss


class LSGANLoss(nn.Module):
    def __init__(self ):
        super(LSGANLoss, self).__init__()
        self.loss_fn = nn.MSELoss()       
        return

    def forward_D(self, d_real, d_fake):
        real_ones_tsr = torch.ones( d_real.shape ).cuda()
        fake_zeros_tsr = torch.zeros( d_fake.shape ).cuda()
        loss_D_real = self.loss_fn( d_real, real_ones_tsr )
        loss_D_fake = self.loss_fn( d_fake, fake_zeros_tsr )
        loss_D = loss_D_real + loss_D_fake
        return loss_D, loss_D_real, loss_D_fake

    def forward_G(self, d_fake):
        real_ones_tsr =  torch.ones( d_fake.shape ).cuda()
        loss_G = self.loss_fn( d_fake, real_ones_tsr )
        return loss_G

    def forward(self, d_real, d_fake, dis_or_gen = True ):
        # Discriminator 用の loss
        if dis_or_gen:
            loss, _, _ = self.forward_D( d_real, d_fake )
        # Generator 用の loss
        else:
            loss = self.forward_G( d_fake )

        return loss


class HingeGANLoss(nn.Module):
    """
    GAN の Hinge loss
        −min(x−1,0)     if D and real
        −min(−x−1,0)    if D and fake
        −x              if G
    """
    def __init__(self):
        super(HingeGANLoss, self).__init__()
        return

    def forward_D(self, d_real, d_fake):
        zeros_tsr =  torch.zeros( d_real.shape ).cuda()
        loss_D_real = - torch.mean( torch.min(d_real - 1, zeros_tsr) )
        #loss_D_fake = - torch.mean( torch.min(-d_fake - 1, zeros_tsr) )
        loss_D_fake = - torch.mean( torch.min(-d_real - 1, zeros_tsr) )
        loss_D = loss_D_real + loss_D_fake
        return loss_D, loss_D_real, loss_D_fake

    def forward_G(self, d_fake):
        real_ones_tsr =  torch.ones( d_fake.shape ).cuda()
        loss_G = - torch.mean(d_fake)
        return loss_G

    def forward(self, d_real, d_fake, dis_or_gen = True ):
        # Discriminator 用の loss
        if dis_or_gen:
            loss, _, _ = self.forward_D( d_real, d_fake )
        # Generator 用の loss
        else:
            loss = self.forward_G( d_fake )

        return loss


#=============================
# GANimation の conditional expression loss
#=============================
class ConditionalExpressionLoss(nn.Module):
    def __init__(self):
        super(ConditionalExpressionLoss, self).__init__()
        self.loss_fn = nn.MSELoss()
        return

    """
    def forward(self, d_size_real, d_size_fake, size ):
        loss_fake = self.loss_fn( d_size_fake, size )
        loss_real = self.loss_fn( d_size_real, size )
        loss = loss_real + loss_fake
        return loss
    """
    def forward(self, d_size, size ):
        loss = self.loss_fn( d_size, size )
        return loss