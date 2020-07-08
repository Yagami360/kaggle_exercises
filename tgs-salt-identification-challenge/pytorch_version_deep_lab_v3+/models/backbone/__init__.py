from models.backbone import resnet, xception, drn, mobilenet

def build_backbone(backbone, n_in_channels, output_stride, BatchNorm, pretrained_backbone):
    if backbone == 'resnet':
        return resnet.ResNet101(n_in_channels, output_stride, BatchNorm, pretrained=pretrained_backbone)
    elif backbone == 'xception':
        return xception.AlignedXception(output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(output_stride, BatchNorm)
    else:
        raise NotImplementedError
