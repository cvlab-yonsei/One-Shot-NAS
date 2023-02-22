from collections import OrderedDict

import torch
import torch.nn as nn

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2

class MBInvertedConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None):
        super(MBInvertedConvLayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim, affine=False, track_running_stats=False)),
                ('act', nn.ReLU6(inplace=True)),
            ]))

        pad = get_same_padding(self.kernel_size)
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim, affine=False, track_running_stats=False)),
            ('act', nn.ReLU6(inplace=True)),
        ]))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False)),
        ]))

    def forward(self, x):
        if self.inverted_bottleneck:
            x = self.inverted_bottleneck(x)
        x = self.depth_conv(x)
        x = self.point_linear(x)
        return x
    
class MBInvertedConvLayer_with_shortcut(nn.Module):

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, expand_ratio=6, mid_channels=None):
        super(MBInvertedConvLayer_with_shortcut, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.kernel_size = kernel_size
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.mid_channels = mid_channels

        if self.mid_channels is None:
            feature_dim = round(self.in_channels * self.expand_ratio)
        else:
            feature_dim = self.mid_channels

        if self.expand_ratio == 1:
            self.inverted_bottleneck = None
        else:
            self.inverted_bottleneck = nn.Sequential(OrderedDict([
                ('conv', nn.Conv2d(self.in_channels, feature_dim, 1, 1, 0, bias=False)),
                ('bn', nn.BatchNorm2d(feature_dim, affine=False, track_running_stats=False)),
                ('act', nn.ReLU6(inplace=True)),
            ]))

        pad = get_same_padding(self.kernel_size)
        self.depth_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, feature_dim, kernel_size, stride, pad, groups=feature_dim, bias=False)),
            ('bn', nn.BatchNorm2d(feature_dim, affine=False, track_running_stats=False)),
            ('act', nn.ReLU6(inplace=True)),
        ]))

        self.point_linear = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(feature_dim, out_channels, 1, 1, 0, bias=False)),
            ('bn', nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False)),
        ]))

    def forward(self, x):
        skip_x = x
        if self.inverted_bottleneck:
            conv_x = self.inverted_bottleneck(x)
            conv_x = self.depth_conv(conv_x)
            conv_x = self.point_linear(conv_x)
            
        else:
            conv_x = self.depth_conv(x)
            conv_x = self.point_linear(conv_x)
        return skip_x + conv_x
    
class Zero_with_shortcut(nn.Module):

    def __init__(self):
        super(Zero_with_shortcut, self).__init__()

    def forward(self, x):
        return x
    
    
class MobileNetV2_search(nn.Module):
    def __init__(self, n_class=1000):
        super(MobileNetV2_search, self).__init__()
    
    
        width_stages = [24,40,80,96,192,320]
        n_cell_stages = [4,4,4,4,4,1]
        stride_stages = [2,2,2,1,2,1]
        
        input_channel = 32
        first_cell_width = 16
        
        self.first_conv = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(3, input_channel, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)),
            ('bn', nn.BatchNorm2d(32, affine=False, track_running_stats=False)),
            ('act', nn.ReLU6(inplace=True))
        ]))
        
        self.first_block_conv = MBInvertedConvLayer(input_channel, first_cell_width, 3, 1, 1)
        input_channel = first_cell_width
        
        self.features = torch.nn.ModuleList()
        for width, n_cell, s in zip(width_stages, n_cell_stages, stride_stages):
            
            for i in range(n_cell):
                self.features.append(torch.nn.ModuleList())
                if i == 0:
                    stride = s
                else:
                    stride = 1
                # conv
                if stride == 1 and input_channel == width:                 
                    self.features[-1].append(MBInvertedConvLayer_with_shortcut(input_channel, width, kernel_size=3, stride = stride, expand_ratio = 3)) 
                    self.features[-1].append(MBInvertedConvLayer_with_shortcut(input_channel, width, kernel_size=3, stride = stride, expand_ratio = 6)) 
                    self.features[-1].append(MBInvertedConvLayer_with_shortcut(input_channel, width, kernel_size=5, stride = stride, expand_ratio = 3))
                    self.features[-1].append(MBInvertedConvLayer_with_shortcut(input_channel, width, kernel_size=5, stride = stride, expand_ratio = 6)) 
                    self.features[-1].append(MBInvertedConvLayer_with_shortcut(input_channel, width, kernel_size=7, stride = stride, expand_ratio = 3)) 
                    self.features[-1].append(MBInvertedConvLayer_with_shortcut(input_channel, width, kernel_size=7, stride = stride, expand_ratio = 6)) 
                    self.features[-1].append(Zero_with_shortcut())
    
                else:
                    self.features[-1].append(MBInvertedConvLayer(input_channel, width, kernel_size=3, stride = stride, expand_ratio = 3)) 
                    self.features[-1].append(MBInvertedConvLayer(input_channel, width, kernel_size=3, stride = stride, expand_ratio = 6)) 
                    self.features[-1].append(MBInvertedConvLayer(input_channel, width, kernel_size=5, stride = stride, expand_ratio = 3))
                    self.features[-1].append(MBInvertedConvLayer(input_channel, width, kernel_size=5, stride = stride, expand_ratio = 6)) 
                    self.features[-1].append(MBInvertedConvLayer(input_channel, width, kernel_size=7, stride = stride, expand_ratio = 3)) 
                    self.features[-1].append(MBInvertedConvLayer(input_channel, width, kernel_size=7, stride = stride, expand_ratio = 6)) 
               
                input_channel = width
                
        last_channel = 1280
        self.feature_mix_layer = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(input_channel, last_channel, kernel_size=(1, 1), stride=(1, 1), bias=False)),
            ('bn', nn.BatchNorm2d(last_channel, affine=False, track_running_stats=False)),
            ('act', nn.ReLU6(inplace=True))
        ]))
        
        self.global_avg_pooling = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Linear(in_features=1280, out_features=1000, bias=True)
        
#         self.arch_cache = None
        
    def forward(self, x, arch):
        x = self.first_conv(x)
        x = self.first_block_conv(x)
#         print(self.arch_cache)
        for archs, arch_id in zip(self.features, arch):
            x = archs[arch_id](x)
        x = self.feature_mix_layer(x)
        x = self.global_avg_pooling(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x
                
