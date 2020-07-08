import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.aspp import build_aspp
from models.decoder import build_decoder
from models.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', n_in_channels = 1, output_stride=16, num_classes = 1, sync_bn=True, freeze_bn=False, pretrained_backbone=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, n_in_channels, output_stride, BatchNorm, pretrained_backbone)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)
        self.activate_tanh = nn.Tanh()
        self.activate_sigmoid = nn.Sigmoid()

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        output = self.activate_tanh( x )
        output_mask = self.activate_sigmoid( x )
        return output, output_mask, x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()


class DeepLabBottleNeck(nn.Module):
    def __init__( self, backbone='resnet', n_in_channels = 1, output_stride=16, num_classes = 1, n_bottleneck_channels=1, sync_bn=True, freeze_bn=False, pretrained_backbone=False):
        super(DeepLabBottleNeck, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, n_in_channels, output_stride, BatchNorm, pretrained_backbone)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm, n_bottleneck_channels)
        self.activate_tanh = nn.Tanh()
        self.activate_sigmoid = nn.Sigmoid()

        self.freeze_bn = freeze_bn

    def forward(self, input, bottle_neck ):
        x, low_level_feat = self.backbone(input)
        #print( "x.shape : ", x.shape )          # torch.Size([32, 2048, 8, 8])
        x = self.aspp(x)

        # bottle neck 部分で concat
        #print( "x.shape : ", x.shape )              # torch.Size([32, 256, 8, 8])
        bottle_neck = bottle_neck.expand(bottle_neck.shape[0], bottle_neck.shape[1], x.shape[2], x.shape[3] )
        concat = torch.cat( [x, bottle_neck], dim=1)
        #print( "concat.shape : ", concat.shape )    # torch.Size([32, 257, 8, 8])

        #
        x = self.decoder(concat, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        output = self.activate_tanh( x )
        output_mask = self.activate_sigmoid( x )
        return output, output_mask, x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == "__main__":
    model = DeepLab(backbone='mobilenet', output_stride=16)
    model.eval()
    input = torch.rand(1, 3, 513, 513)
    output = model(input)
    print(output.size())


