import torch
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
import torch._utils as utils
import torch.nn.functional as F
import torch.functional as f
import torch.nn as nn
class GatedSpatialConv2d(_ConvNd):

    def __init__(self,out_channels,in_channels,kernel_size=1,stride=1,padding=0,dilation=1,groups=1,bias=False):

        kernel_size = _pair(kernel_size)

        stride = _pair(stride)

        padding = _pair(padding)

        dilation = _pair(dilation)


        super(GatedSpatialConv2d,self).__init__(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,
                                                transposed=False,output_padding=_pair(0),groups=groups,bias=bias,padding_mode='zeros')
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=1)

        self._gate_conv = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels+1),
            nn.Conv2d(in_channels=in_channels+1,out_channels=in_channels+1,kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels+1,out_channels=1,kernel_size=1),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )

    def forward(self,input_features,gating_feature,H,W):

        gating_feature = gating_feature.contiguous().view(gating_feature.shape[0],H,W,-1)

        gating_feature = self.conv1(gating_feature.permute(0,3,2,1))

        alphas = self._gate_conv(torch.cat([input_features,gating_feature],dim=1))

        input_feature1 = (alphas+1) * input_features


        return F.conv2d(input=input_feature1,weight=self.weight,bias=self.bias,stride=self.stride,padding=self.padding,dilation=self.dilation,groups=self.groups)

    def reset_parameters(self):

        nn.init.xavier_normal_(self.weight)

        if self.bias is not None:

            nn.init.zeros_(self.bias)

