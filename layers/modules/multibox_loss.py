# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import v2 as cfg
from ..box_utils import match, log_sum_exp


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']

    def forward(self, predictions, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
                5位数，前四位坐标，最后一位是label
        """
        loc_data, conf_data, priors = predictions
        """
        loc_data : [batch_size, 8732, 4]
        conf_data : [batch_size, 8732, 21]
        priors : [8732, 4]
        """
        num = loc_data.size(0)
        # num : batch_size
        priors = priors[:loc_data.size(1), :]
        # priors ： [8732, 4]
        num_priors = (priors.size(0))
        # num_priors : 8732
        num_classes = self.num_classes
        # num_classes : 21(object 类型数目)

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        # loc_t : [batch_size, 8732, 4]
        conf_t = torch.LongTensor(num, num_priors)
        # conf_t : [batch_size, 8732]

        for idx in range(num):
            """遍历每一张图片"""
            truths = targets[idx][:, :-1].data
            """
            shape = [num_instance, 4]
            每一个ground truth类的bbox
            """
            labels = targets[idx][:, -1].data
            """
            labels = [num_instance,1]
            每一个ground truth的label
            """
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,loc_t, conf_t, idx)
            """
            遍历每一张图片，分别求得每一张图片的每一个anchor所匹配的gt box offset和gt classid
            loc_t : [batch_size, 8732, 4]
            conf_t : [batch_size, 8732]
            """

        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        """看起来这两个变量是要作为target输入loss计算的，当然不需要回传梯度了"""

        pos = conf_t > 0 # 0代表是background
        # [batch_size, 8732]
        num_pos = pos.sum(dim=1, keepdim=True)
        # [batch_size,1]
        """
        求出每张图片中包含positive object的框的数目
        （每张图有多少个框框住目标了）
        看起来会出现有些图片中目标数为0，进而positive anchor的数目也是0的情况
        """

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        # [8, 8732, 4] 复制四倍是为了接下来collect方便
        loc_p = loc_data[pos_idx].view(-1, 4)
        # 预测的所有图片的所有anchors的offset里面按positive的idx来collect
        loc_t = loc_t[pos_idx].view(-1, 4)
        # 实际生成的的所有图片的所有anchors的offset里面按positive的idx来collect
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        # 这里就可以计算offset 回归loss了

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # [batch_size*8732,num_classes]

        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        """ 
        这个计算和如下过程是等同的：
        torch.nn.functional.log_softmax(batch_conf,dim=1).gather(1,conf_t.view(-1,1))
        其实就是先对batch_conf做一番softmax(dim=1)
        然后取每一行(既每一个预测的anchor的21个类别的概率分布)真实类别对应的id的值
        或者说是classifier预测对的那个值取softmax
        论文里描述这一过程的段落是Hard Negative Mining
        loss_c的值在论文里被称为confidence loss
        与其随机的取negative的sample,不如取那些confidence loss大的
        论文说这样能加速和稳定训练
        直观的理解是如果只是随机的取negative samples，
        很有可能取到的是空空如也的背景，
        对machine来说太容易了，
        如果我取那些概率高的negative samples，
        意味着机器就要学着仔细分辨了
        可是这里是一刀切0.5以上positive,0.5一下negative.
        设置成 >0.7 positive, <0.3 negative会不会更好 ？
        """

        # Hard Negative Mining
        # loss_c[pos] = 0  # filter out pos boxes for now
        loss_c[pos.view(-1,1)] = 0  # 修改了一下下，这样不会报user warning
        loss_c = loss_c.view(num, -1) # shape : [8, 8732]
        _, loss_idx = loss_c.sort(1, descending=True) # 求出每一行按数值大小降序排列的序号
        _, idx_rank = loss_idx.sort(1) # 再对loss_idx求出每一行数值大小升序排列的序号
        num_pos = pos.long().sum(1, keepdim=True) # shape=[batch_size,1] 每一张图片positive框的个数
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1) 
        # num_neg是等会要取多少个negative框的个数，与positive数量成比例
        
        """
        shape=[batch_size,1] 每一张图片negative框的个数
        clamp的作用是截断，当positive框很多的时候，negative框数量也不可能超过所有框的总数
        """
        neg = idx_rank < num_neg.expand_as(idx_rank)
        """
        _, loss_idx = loss_c.sort(1, descending=True) # 求出每一行按数值大小降序排列的序号
        _, idx_rank = loss_idx.sort(1) # 再对loss_idx求出每一行数值大小升序排列的序号
        neg = idx_rank < num_neg.expand_as(idx_rank)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        这个矩阵操作很巧妙
        通过求两次sort(先对值sort,再对sort后的序号sort),
        idx_rank < 每一行都设置一个想取的数量N，
        这时就可以得到每一行里N个最大值所在位置的掩码
        你可以构造一个矩阵试一试，看看idx_rank中每一行等于1的位置是不是正好也是loss_c中该行最大值的位置？
        如果是我，很可能就要用for循环了
        """

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        """
        gt是greater than的意思。。。。
        (pos_idx+neg_idx).gt(0) 相当于把所有大于0的值标出来
        """
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N
        return loss_l, loss_c
