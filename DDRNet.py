"""
Paper:      Deep Dual-resolution Networks for Real-time and Accurate Semantic 
            Segmentation of Road Scenes
Url:        https://arxiv.org/abs/2101.06085
Create by:  zh320
Date:       2023/07/29
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

from modules import conv1x1, ConvBNAct, Activation, SegHead
import numpy as np

Label = namedtuple( 'Label' , [

        'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                        # We use them to uniquely name a class

        'id'          , # An integer ID that is associated with this label.
                        # The IDs are used to represent the label in ground truth images
                        # An ID of -1 means that this label does not have an ID and thus
                        # is ignored when creating ground truth images (e.g. license plate).
                        # Do not modify these IDs, since exactly these IDs are expected by the
                        # evaluation server.

        'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                        # ground truth images with train IDs, using the tools provided in the
                        # 'preparation' folder. However, make sure to validate or submit results
                        # to our evaluation server using the regular IDs above!
                        # For trainIds, multiple labels might have the same ID. Then, these labels
                        # are mapped to the same class in the ground truth images. For the inverse
                        # mapping, we use the label that is defined first in the list below.
                        # For example, mapping all void-type classes to the same ID in training,
                        # might make sense for some approaches.
                        # Max value is 255!

        'category'    , # The name of the category that this label belongs to

        'categoryId'  , # The ID of this category. Used to create ground truth images
                        # on category level.

        'hasInstances', # Whether this label distinguishes between single instances or not

        'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                        # during evaluations or not

        'color'       , # The color of this label
        ] )

    #------------------

 
colormap = {0:(128, 64,128), 1:(244, 35,232), 2:( 70, 70, 70), 3:(102,102,156),
            4:(190,153,153), 5:(153,153,153), 6:(250,170, 30), 7:(220,220,  0),
            8:(107,142, 35), 9:(152,251,152), 10:( 70,130,180), 11:(220, 20, 60),
            12:(255,  0,  0), 13:(  0,  0,142), 14:(  0,  0, 70), 15:(  0, 60,100),
            16:(  0, 80,100), 17:(  0,  0,230), 18:(119, 11, 32)}

# self.labels = [
#         #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
#         Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#         Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#         Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#         Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#         Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
#         Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
#         Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
#         Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
#         Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
#         Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
#         Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
#         Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
#         Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
#         Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
#         Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
#         Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
#         Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
#         Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
#         Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
#         Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
#         Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
#         Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
#         Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
#         Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
#         Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
#         Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
#         Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
#         Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
#         Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
#         Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
#         Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
#         Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
#         Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
#         Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
#         Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
#     ]

class DDRNet(nn.Module):
    def __init__(self, num_class=1, n_channel=3, arch_type='DDRNet-23-slim', act_type='relu', 
                    use_aux=True):
        super(DDRNet, self).__init__()
        arch_hub = {'DDRNet-23-slim': {'init_channel': 32, 'repeat_times': [2, 2, 2, 0, 2, 1]},
                    'DDRNet-23': {'init_channel': 64, 'repeat_times': [2, 2, 2, 0, 2, 1]},
                    'DDRNet-39': {'init_channel': 64, 'repeat_times': [3, 4, 3, 3, 3, 1]},
                    }
        if arch_type not in arch_hub.keys():
            raise ValueError(f'Unsupport architecture type: {arch_type}.\n')
            
        init_channel = arch_hub[arch_type]['init_channel']
        repeat_times = arch_hub[arch_type]['repeat_times']
        self.use_aux = use_aux
        self.num_class = num_class

        self.conv1 = ConvBNAct(n_channel, init_channel, 3, 2, act_type=act_type)
        self.conv2 = Stage2(init_channel, repeat_times[0], act_type)
        self.conv3 = Stage3(init_channel, repeat_times[1], act_type)
        self.conv4 = Stage4(init_channel, repeat_times[2], repeat_times[3], act_type)
        self.conv5 = Stage5(init_channel, repeat_times[4], repeat_times[5], act_type)
        self.seg_head = SegHead(init_channel*4, num_class, act_type)
        if self.use_aux:
            self.aux_head = SegHead(init_channel*2, num_class, act_type)

        self.labels = [
        #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
        Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
        Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
        Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
        Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
        Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
        Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
        Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
        Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
        Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
        Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
        Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
        Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
        Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
        Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
        Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
        Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
        Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
        Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
        Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
        Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
        Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
        Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
        Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
        Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
        Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
        Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
        Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
        Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
        Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
        Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
        Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
    ]

    def forward(self, x, is_training=False):
        size = x.size()[2:]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x_low, x_high = self.conv4(x)
        
        if self.use_aux:
            x_aux = self.aux_head(x_high)
            # x_aux = F.interpolate(x_aux, size, mode='bilinear', align_corners=True)
            
        x = self.conv5(x_low, x_high)
        x = self.seg_head(x)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)

        if self.use_aux and is_training:
            return x, (x_aux,)
        else:
            return x


class Stage2(nn.Module):
    def __init__(self, init_channel, repeat_times, act_type='relu'):
        super(Stage2, self).__init__()
        in_channels = init_channel
        out_channels = init_channel
        
        layers = [ConvBNAct(in_channels, out_channels, 3, 2, act_type=act_type)]
        for _ in range(repeat_times):
            layers.append(RB(out_channels, out_channels, 1, act_type))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


def build_blocks(block, in_channels, out_channels, stride, repeat_times, act_type):
    layers = [block(in_channels, out_channels, stride, act_type=act_type)]
    for _ in range(1, repeat_times):
        layers.append(block(out_channels, out_channels, 1, act_type=act_type))
    return nn.Sequential(*layers)


class Stage3(nn.Module):
    def __init__(self, init_channel, repeat_times, act_type='relu'):
        super(Stage3, self).__init__()
        in_channels = init_channel
        out_channels = init_channel * 2

        self.conv = build_blocks(RB, in_channels, out_channels, 2, repeat_times, act_type)

    def forward(self, x):
        return self.conv(x)


class Stage4(nn.Module):
    def __init__(self, init_channel, repeat_times1, repeat_times2, act_type='relu'):
        super(Stage4, self).__init__()
        in_channels = init_channel * 2
        low_res_channels = init_channel * 4
        high_res_channels = init_channel * 2
        if low_res_channels < high_res_channels:
            raise ValueError('Low resolution channel should be more than high resolution channel.\n')

        self.low_conv1 = build_blocks(RB, in_channels, low_res_channels, 2, repeat_times1, act_type)
        self.high_conv1 = build_blocks(RB, in_channels, high_res_channels, 1, repeat_times1, act_type)
        self.bilateral_fusion1 = BilateralFusion(low_res_channels, high_res_channels, 2)

        self.extra_conv = repeat_times2 > 0
        if self.extra_conv:
            self.low_conv2 = build_blocks(RB, low_res_channels, low_res_channels, 1, repeat_times2, act_type)
            self.high_conv2 = build_blocks(RB, high_res_channels, high_res_channels, 1, repeat_times2, act_type)
            self.bilateral_fusion2 = BilateralFusion(low_res_channels, high_res_channels, 2)

    def forward(self, x):
        x_low = self.low_conv1(x)
        x_high = self.high_conv1(x)
        x_low, x_high = self.bilateral_fusion1(x_low, x_high)

        if self.extra_conv:
            x_low = self.low_conv2(x_low)
            x_high = self.high_conv2(x_high)
            x_low, x_high = self.bilateral_fusion2(x_low, x_high)

        return x_low, x_high


class Stage5(nn.Module):
    def __init__(self, init_channel, repeat_times1, repeat_times2, act_type='relu'):
        super(Stage5, self).__init__()
        low_in_channels = init_channel * 4
        high_in_channels = init_channel * 2
        low_res_channels1 = init_channel * 8
        high_res_channels1 = init_channel * 2
        low_res_channels2 = init_channel * 16
        high_res_channels2 = init_channel * 4
        if (low_in_channels < high_in_channels) or (low_res_channels1 < high_res_channels1) or (low_res_channels2 < high_res_channels2):
            raise ValueError('Low resolution channel should be more than high resolution channel.\n')

        self.low_conv1 = build_blocks(RB, low_in_channels, low_res_channels1, 2, repeat_times1, act_type)
        self.high_conv1 = build_blocks(RB, high_in_channels, high_res_channels1, 1, repeat_times1, act_type)
        self.bilateral_fusion = BilateralFusion(low_res_channels1, high_res_channels1, 4)

        self.low_conv2 = build_blocks(RBB, low_res_channels1, low_res_channels2, 2, repeat_times2, act_type)
        self.high_conv2 = build_blocks(RBB, high_res_channels1, high_res_channels2, 1, repeat_times2, act_type)
        self.dappm = DAPPM(low_res_channels2, high_res_channels2)

    def forward(self, x_low, x_high):
        size = x_high.size()[2:]

        x_low = self.low_conv1(x_low)
        x_high = self.high_conv1(x_high)
        x_low, x_high = self.bilateral_fusion(x_low, x_high)

        x_low = self.low_conv2(x_low)
        x_low = self.dappm(x_low)
        x_low = F.interpolate(x_low, size, mode='bilinear', align_corners=True)

        x_high = self.high_conv2(x_high) + x_low

        return x_high


class RB(nn.Module):
    # Building sequential residual basic blocks, codes are based on
    # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    def __init__(self, in_channels, out_channels, stride=1, act_type='relu'):
        super(RB, self).__init__()
        self.downsample = (stride > 1) or (in_channels != out_channels)
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, stride, act_type=act_type)
        self.conv2 = ConvBNAct(out_channels, out_channels, 3, 1, act_type='none')

        if self.downsample:
            self.conv_down = ConvBNAct(in_channels, out_channels, 1, stride, act_type='none')
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_down(x)
        out += identity
        out = self.act(out)

        return out


class RBB(nn.Module):
    # Building single residual bottleneck block, codes are based on
    # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    def __init__(self, in_channels, out_channels, stride=1, act_type='relu'):
        super(RBB, self).__init__()
        self.downsample = (stride > 1) or (in_channels != out_channels)
        self.conv1 = ConvBNAct(in_channels, in_channels, 1, act_type=act_type)
        self.conv2 = ConvBNAct(in_channels, in_channels, 3, stride, act_type=act_type)
        self.conv3 = ConvBNAct(in_channels, out_channels, 1, act_type='none')

        if self.downsample:
            self.conv_down = ConvBNAct(in_channels, out_channels, 1, stride, act_type='none')
        self.act = Activation(act_type)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if self.downsample:
            identity = self.conv_down(x)
        out += identity
        out = self.act(out)

        return out


class BilateralFusion(nn.Module):
    def __init__(self, low_res_channels, high_res_channels, stride, act_type='relu'):
        super(BilateralFusion, self).__init__()
        self.conv_low = ConvBNAct(low_res_channels, high_res_channels, 1, act_type='none')
        self.conv_high = ConvBNAct(high_res_channels, low_res_channels, 3, stride, act_type='none')
        self.act = Activation(act_type)

    def forward(self, x_low, x_high):
        size = x_high.size()[2:]
        fuse_low = self.conv_low(x_low)
        fuse_high = self.conv_high(x_high)
        x_low = self.act(x_low + fuse_high)

        fuse_low = F.interpolate(fuse_low, size, mode='bilinear', align_corners=True)
        x_high = self.act(x_high + fuse_low)

        return x_low, x_high


class DAPPM(nn.Module):
    def __init__(self, in_channels, out_channels, act_type='relu'):
        super(DAPPM, self).__init__()
        hid_channels = int(in_channels // 4)
        
        self.conv0 = ConvBNAct(in_channels, out_channels, 1, act_type=act_type)
        self.conv1 = ConvBNAct(in_channels, hid_channels, 1, act_type=act_type)
        self.pool2 = self._build_pool_layers(in_channels, hid_channels, 5, 2)
        self.conv2 = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.pool3 = self._build_pool_layers(in_channels, hid_channels, 9, 4)
        self.conv3 = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.pool4 = self._build_pool_layers(in_channels, hid_channels, 17, 8)
        self.conv4 = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.pool5 = self._build_pool_layers(in_channels, hid_channels, -1, -1)
        self.conv5 = ConvBNAct(hid_channels, hid_channels, 3, act_type=act_type)
        self.conv_last = ConvBNAct(hid_channels*5, out_channels, 1, act_type=act_type)
        
    def _build_pool_layers(self, in_channels, out_channels, kernel_size, stride):
        layers = []
        if kernel_size == -1:
            layers.append(nn.AdaptiveAvgPool2d(1))
        else:
            padding = (kernel_size - 1) // 2
            layers.append(nn.AvgPool2d(kernel_size, stride, padding))
        layers.append(conv1x1(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        size = x.size()[2:]
        y0 = self.conv0(x)
        y1 = self.conv1(x)
        
        y2 = self.pool2(x)
        y2 = F.interpolate(y2, size, mode='bilinear', align_corners=True)
        y2 = self.conv2(y1 + y2)
        
        y3 = self.pool3(x)
        y3 = F.interpolate(y3, size, mode='bilinear', align_corners=True)
        y3 = self.conv3(y2 + y3)
        
        y4 = self.pool4(x)
        y4 = F.interpolate(y4, size, mode='bilinear', align_corners=True)
        y4 = self.conv4(y3 + y4)
    
        y5 = self.pool5(x)
        y5 = F.interpolate(y5, size, mode='bilinear', align_corners=True)
        y5 = self.conv5(y4 + y5)
        
        x = self.conv_last(torch.cat([y1, y2, y3, y4, y5], dim=1)) + y0
    
        return x