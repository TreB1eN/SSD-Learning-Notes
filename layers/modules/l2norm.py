import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.init as init

class L2Norm(nn.Module):
    def __init__(self,n_channels, scale):
        super(L2Norm,self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()
        """
        注意这里reset_parameters,self.weight是个parameter
        它的值是初始化成20，然后可以在training过程中学习的
        设置成20是因为如果按channel把每一个像素上的值归一化成1的话，
        那x的值就太小了，不利于训练
        设置成20是一个经验值，论文里说这样训练效果最流畅
        这篇论文：
        ParseNet：Looking wider to see better
        https://arxiv.org/abs/1506.04579
        """
    def reset_parameters(self):
        init.constant(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        #x /= norm
        x = torch.div(x,norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
