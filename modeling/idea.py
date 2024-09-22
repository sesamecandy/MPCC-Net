# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn
import torch.nn.functional as F

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.senet import SENet, SEResNetBottleneck, SEBottleneck, SEResNeXtBottleneck
from .backbones.resnet_ibn_a import resnet50_ibn_a
from .backbones.batchnorm2d_wei import Batchnorm2d_wei

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Idea(nn.Module):
    in_planes = 2048

    def __init__(self, num_classes, last_stride, model_path, neck, neck_feat, model_name, pretrain_choice):
        super(Idea, self).__init__()
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, 
                               block=BasicBlock, 
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck, 
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.base = ResNet(last_stride=last_stride, 
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])
            
        elif model_name == 'se_resnet50':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 6, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnet101':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 4, 23, 3], 
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'se_resnet152':
            self.base = SENet(block=SEResNetBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=1, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)  
        elif model_name == 'se_resnext50':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 6, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride) 
        elif model_name == 'se_resnext101':
            self.base = SENet(block=SEResNeXtBottleneck,
                              layers=[3, 4, 23, 3], 
                              groups=32, 
                              reduction=16,
                              dropout_p=None, 
                              inplanes=64, 
                              input_3x3=False,
                              downsample_kernel_size=1, 
                              downsample_padding=0,
                              last_stride=last_stride)
        elif model_name == 'senet154':
            self.base = SENet(block=SEBottleneck, 
                              layers=[3, 8, 36, 3],
                              groups=64, 
                              reduction=16,
                              dropout_p=0.2, 
                              last_stride=last_stride)
        elif model_name == 'resnet50_ibn_a':
            self.base = resnet50_ibn_a(last_stride)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat

        if self.neck == 'no':
            self.classifier = nn.Conv2d(self.in_planes, self.num_classes, kernel_size=1, padding=0, bias=False)
            self.classifier.apply(weights_init_kaiming)  # new add by luo
        elif self.neck == 'bnneck':
            self.bottleneck = Batchnorm2d_wei(num_features=self.in_planes)
            self.classifier = nn.Conv2d(self.in_planes, self.num_classes, kernel_size=1, padding=0)
            self.classifier.apply(weights_init_kaiming)

    def forward(self, x, label=None):
        if self.training:
            mode = 'True'
        else:
            mode = 'False'
        xs = F.upsample_bilinear(x, [336, 112])  # 288,96     #增强之后的图像
        xf = xs.flip(-1)

        base = self.base(x)
        base_map = torch.mean(base, dim=1, keepdim=True)
        base = base * base_map

        base_s = self.base(xs)
        base_map_s = torch.mean(base_s, dim=1, keepdim=True)
        base_s = base_s * base_map_s

        base_f = self.base(xf)
        base_map_f = torch.mean(base_f, dim=1, keepdim=True)
        base_f = base_f * base_map_f

        if self.neck == 'bnneck':
            base_BN = self.bottleneck(base, is_train=mode)
            base_BN_s = self.bottleneck(base_s, is_train=mode)
            base_BN_f = self.bottleneck(base_f, is_train=mode)
        if self.neck == 'no':
            global_feat = self.gap(base)
            global_feat = global_feat.view(global_feat.shape[0], -1)
            feat = global_feat
        elif self.neck == 'bnneck':     #如果bnneck为True，则将feat经过bn再进行test
            global_feat = self.gap(base)
            global_feat = global_feat.view(global_feat.shape[0], -1)
            feat = self.gap(base_BN)
            feat = feat.view(feat.shape[0], -1)

            global_feat_s = self.gap(base_s)
            global_feat_s = global_feat_s.view(global_feat_s.shape[0], -1)
            feat_s = self.gap(base_BN_s)
            feat_s = feat_s.view(feat_s.shape[0], -1)

            global_feat_f = self.gap(base_f)
            global_feat_f = global_feat_f.view(global_feat_f.shape[0], -1)
            feat_f = self.gap(base_BN_f)
            feat_f = feat_f.view(feat_f.shape[0], -1)

        if self.training:
            if self.neck == 'no':
                cls_score = self.gap(self.classifier(base)).view(feat.shape[0], -1)
            elif self.neck == 'bnneck':
                score_map = self.classifier(base_BN)
                localization_map_normed = self.get_atten_map(feature_maps=score_map, gt_labels=label, normalize=True)
                cls_score = self.gap(score_map).view(feat.shape[0], -1)
                #cls_score = self.gap(self.classifier(cls_score)).view(feat.shape[0], -1)

                score_map_s = self.classifier(base_BN_s)
                localization_map_normed_s = self.get_atten_map(feature_maps=score_map_s, gt_labels=label, normalize=True)
                cls_score_s = self.gap(score_map_s).view(feat_s.shape[0], -1)

                score_map_f = self.classifier(base_BN_f)
                localization_map_normed_f = self.get_atten_map(feature_maps=score_map_f, gt_labels=label, normalize=True)
                cls_score_f = self.gap(score_map_f).view(feat_f.shape[0], -1)
            return [cls_score,cls_score_s,cls_score_f],\
                [global_feat,global_feat_s,global_feat_f],\
                [localization_map_normed,localization_map_normed_s,localization_map_normed_f],\
                [base_map,base_s,base_f]  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

    def normalize_atten_maps(self, atten_maps):
        atten_shape = atten_maps.size()

        # --------------------------
        batch_mins, _ = torch.min(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        batch_maxs, _ = torch.max(atten_maps.view(atten_shape[0:-2] + (-1,)), dim=-1, keepdim=True)
        atten_normed = torch.div(atten_maps.view(atten_shape[0:-2] + (-1,)) - batch_mins,
                                 batch_maxs - batch_mins)
        atten_normed = atten_normed.view(atten_shape)

        return atten_normed

    def get_atten_map(self, feature_maps, gt_labels, normalize=True):
        label = gt_labels

        feature_map_size = feature_maps.size()
        batch_size = feature_map_size[0]

        atten_map = torch.zeros([feature_map_size[0], 1, feature_map_size[2], feature_map_size[3]])
        atten_map = (atten_map.cuda())
        for batch_idx in range(batch_size):
            atten_map[batch_idx, 0, :, :] = (feature_maps[batch_idx, label.data[batch_idx], :, :])

        if normalize:
            atten_map = self.normalize_atten_maps(atten_map)

        return atten_map