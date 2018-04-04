from __future__ import division
import torch
from math import sqrt as sqrt
from itertools import product as product

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    Note:
    This 'layer' has changed between versions of the original SSD
    paper, so we include both versions, but note v2 is the most tested and most
    recent version of the paper.

    """
    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        # self.type = cfg.name
        self.image_size = cfg['min_dim']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['aspect_ratios'])
        self.variance = cfg['variance'] or [0.1]
        self.feature_maps = cfg['feature_maps']
        self.min_sizes = cfg['min_sizes']
        self.max_sizes = cfg['max_sizes']
        self.steps = cfg['steps']
        self.aspect_ratios = cfg['aspect_ratios']
        self.clip = cfg['clip']
        self.version = cfg['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        # TODO merge these
        """
        it's not following the paper at all,
        looks like the scale of anchors is rather handcrafted than build by the equation in the paper...
        it is because of the paper said :
        . In practice, one can also design a distribution of default boxes to
        best fit a specific dataset. How to design the optimal tiling is an open question as well.
        """
        """
        这个函数跟论文对不起来，应该是更新过了
        我的理解是这样的：
        首先，feature_maps有[38, 19, 10, 5, 3, 1]
        min_sizes： [30, 60, 111, 162, 213, 264]
        max_sizes : [60, 111, 162, 213, 264, 315]
        注意到max_size正好是min_size[index+1]
        paper里Sk的计算是套公式算出来的，这里实现直接就用了最终数值
        这个数值哪来的不知道，感觉是个经验值
        还有一些细节在函数体里面谈
        """


        if self.version == 'v2':
            """
            大循环是针对每一层feature_map尺寸的循环，
            相当于对每一层feature_map有选择地从min_sizes,max_sizes里面
            挑选size,生成不同大小的anchor,所以每一层的anchor尺度都不一样
            """
            for k, f in enumerate(self.feature_maps):
                """
                itertool里面带的product函数：
                product(range(3), repeat=2)的作用是生成如下序列：
                [1,1]
                [2,1]
                [3,1]
                [1,2]
                [2,2]
                [3,2]
                [1,3]
                [2,3]
                [3,3]
                其实就是把feature_map上的每一个点都遍历一遍
                """
                for i, j in product(range(f), repeat=2):
                    """
                    steps : [8, 16, 32, 64, 100, 300]
                    f_k其实就约等于feature_map的大小，你可以用image_size/steps看看
                    cx,cy就是中心点的坐标
                    """
                    f_k = self.image_size / self.steps[k]
                    # unit center x,y

                    cx = (j + 0.5) / f_k
                    cy = (i + 0.5) / f_k

                    # aspect_ratio: 1
                    # rel size: min_size
                    s_k = self.min_sizes[k]/self.image_size
                    mean += [cx, cy, s_k, s_k]

                    # aspect_ratio: 1
                    # rel size: sqrt(s_k * s_(k+1))
                    """论文里有写这个"""
                    s_k_prime = sqrt(s_k * (self.max_sizes[k]/self.image_size))
                    mean += [cx, cy, s_k_prime, s_k_prime]

                    """
                    aspect_ratios : [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
                    乍看觉得很奇葩是不是 ？
                    注意它是列表的列表
                    其实的意思是，论文里说对每一层feature_map里的每一个点都生成6个不同尺寸的anchor
                    但这里操作的时候，对第一层和最后两层就没有去生成1:3,3:1的anchor了
                    其实也是比较好理解，当一个anchor本身尺度就特别小或者特别大的时候，
                    特别扁的长方形的anchor可能意义就不大了，毕竟后面还有回归
                    这个操作感觉是个经验值
                    """
                    # rest of aspect ratios
                    for ar in self.aspect_ratios[k]:
                        mean += [cx, cy, s_k*sqrt(ar), s_k/sqrt(ar)]
                        mean += [cx, cy, s_k/sqrt(ar), s_k*sqrt(ar)]

        else:
            # original version generation of prior (default) boxes
            for i, k in enumerate(self.feature_maps):
                step_x = step_y = self.image_size/k
                for h, w in product(range(k), repeat=2):
                    c_x = ((w+0.5) * step_x)
                    c_y = ((h+0.5) * step_y)
                    c_w = c_h = self.min_sizes[i] / 2
                    s_k = self.image_size  # 300
                    # aspect_ratio: 1,
                    # size: min_size
                    mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                             (c_x+c_w)/s_k, (c_y+c_h)/s_k]
                    if self.max_sizes[i] > 0:
                        # aspect_ratio: 1
                        # size: sqrt(min_size * max_size)/2
                        c_w = c_h = sqrt(self.min_sizes[i] *
                                         self.max_sizes[i])/2
                        mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                                 (c_x+c_w)/s_k, (c_y+c_h)/s_k]
                    # rest of prior boxes
                    for ar in self.aspect_ratios[i]:
                        if not (abs(ar-1) < 1e-6):
                            c_w = self.min_sizes[i] * sqrt(ar)/2
                            c_h = self.min_sizes[i] / sqrt(ar)/2
                            mean += [(c_x-c_w)/s_k, (c_y-c_h)/s_k,
                                     (c_x+c_w)/s_k, (c_y+c_h)/s_k]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        """
        还原回tensor,size : [8732, 4]
        38*38*4 + 19**2*6 + 10**2*6 + 5**2*6 + 3**2*4 + 1*4 = 8732
        """
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
