import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma = 2, eps = 1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


class CMFL(nn.Module):
    r"""
    This criterion is a implemenation of Focal Loss, which is proposed in
    Focal Loss for Dense Object Detection.
                Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5), putting more focus on hard, misclassi?ed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are instead summed for each minibatch.
        """
    # criterion_CMFL = CMFL(device=device, class_num=2, gamma=2)
    def __init__(self, alpha=1, gamma=2, binary=False, multiplier=2, sg=False):
        super(CMFL, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.binary = binary
        self.multiplier = multiplier
        self.sg = sg

    def forward(self, inputs_p, inputs_q, targets):
        probs_p = F.cross_entropy(inputs_p, targets, reduction='none')
        probs_q = F.cross_entropy(inputs_q, targets, reduction='none')
        pt_a = torch.exp(-probs_p)
        pt_b = torch.exp(-probs_q)
        eps = 0.000000001
        if self.sg:
            d_pt_a = pt_a.detach()
            d_pt_b = pt_b.detach()
            w_p = ((d_pt_b + eps) * (self.multiplier * pt_a * d_pt_b)) / (pt_a + d_pt_b + eps)
            w_q = ((d_pt_a + eps) * (self.multiplier * d_pt_a * pt_b)) / (d_pt_a + pt_b + eps)
        else:
            w_p = ((pt_b + eps) * (self.multiplier * pt_a * pt_b)) / (pt_a + pt_b + eps)
            w_q = ((pt_a + eps) * (self.multiplier * pt_a * pt_b)) / (pt_a + pt_b + eps)

        if self.binary:
            w_p = w_p * (1 - targets)
            w_q = w_q * (1 - targets)

        f_loss_p = self.alpha * (1 - w_p) ** self.gamma * probs_p
        f_loss_q = self.alpha * (1 - w_q) ** self.gamma * probs_q

        loss = 0.5 * torch.mean(f_loss_p) + 0.5 * torch.mean(f_loss_q)

        return torch.mean(f_loss_p), torch.mean(f_loss_q), loss


class ICMFL(nn.Module):
    r"""
    This criterion is a implemenation of Focal Loss, which is proposed in
    Focal Loss for Dense Object Detection.
                Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classi?ed examples (p > .5), putting more focus on hard, misclassi?ed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are instead summed for each minibatch.
        """
    # criterion_CMFL = CMFL(device=device, class_num=2, gamma=2)
    def __init__(self, alpha=1, gamma=2, binary=False, multiplier=3, sg=False):
        super(ICMFL, self).__init__()
        self.alpha = alpha
        # self.gamma = gamma
        self.gamma = 0.5
        self.binary = binary
        self.multiplier = multiplier
        self.sg = sg

    def forward(self, inputs_p1, inputs_p2,inputs_p3, targets):
        probs_p1 = F.cross_entropy(inputs_p1, targets, reduction='none')
        probs_p2 = F.cross_entropy(inputs_p2, targets, reduction='none')
        probs_p3 = F.cross_entropy(inputs_p3, targets, reduction='none')

        pt_a = torch.exp(-probs_p1)
        pt_b = torch.exp(-probs_p2)
        pt_c = torch.exp(-probs_p3)

        eps = 0.000000001
        if self.sg:
            d_pt_a = pt_a.detach()
            d_pt_b = pt_b.detach()
            d_pt_c = pt_c.detach()
            w_p1 = (torch.min(d_pt_b, d_pt_c) + eps) * (self.multiplier * pt_a * d_pt_b * d_pt_c) / (pt_a * d_pt_b + pt_a * d_pt_c + d_pt_b * d_pt_c + eps)
            w_p2 = (torch.min(d_pt_a, d_pt_c) + eps) * (self.multiplier * d_pt_a * d_pt_b * pt_c) / (d_pt_a * d_pt_b + d_pt_a * pt_c + d_pt_b * pt_c + eps)
            w_p3 = (torch.min(d_pt_b, d_pt_a) + eps) * (self.multiplier * d_pt_a * pt_b * d_pt_c) / (d_pt_a * pt_b + d_pt_a * d_pt_c + pt_b * d_pt_c + eps)
        else:
            th = (self.multiplier * pt_a * pt_b * pt_c) / (pt_a * pt_b + pt_a * pt_c + pt_b * pt_c + eps)
            w_p1 = (torch.min(pt_b, pt_c) + eps) * th
            w_p2 = (torch.min(pt_a, pt_c) + eps) * th
            w_p3 = (torch.min(pt_b, pt_a) + eps) * th


        if self.binary:
            w_p1 = w_p1 * (1 - targets)
            w_p2 = w_p2 * (1 - targets)
            w_p3 = w_p3 * (1 - targets)

        # thre = 0.999
        # zero = torch.zeros_like(pt_a)
        # w_p1 = torch.where(pt_a>=thre,0,1) * w_p1 + torch.where(pt_a<thre, zero, pt_a)
        # w_p2 = torch.where(pt_b>=thre,0,1) * w_p2 + torch.where(pt_b<thre, zero, pt_b)
        # w_p3 = torch.where(pt_c>=thre,0,1) * w_p3 + torch.where(pt_c<thre, zero, pt_c)
        # w_p1 = torch.where(pt_a>=thre,0,1) * w_p1
        # w_p2 = torch.where(pt_b>=thre,0,1) * w_p2
        # w_p3 = torch.where(pt_c>=thre,0,1) * w_p3

        f_loss_p1 = self.alpha * (1 - w_p1) ** self.gamma * probs_p1
        f_loss_p2 = self.alpha * (1 - w_p2) ** self.gamma * probs_p2
        f_loss_p3 = self.alpha * (1 - w_p3) ** self.gamma * probs_p3        

        loss = torch.mean(f_loss_p1) + torch.mean(f_loss_p2) + torch.mean(f_loss_p3)

        return torch.mean(f_loss_p1), torch.mean(f_loss_p2), torch.mean(f_loss_p3), loss


class PixWiseBCELoss(nn.Module):
    def __init__(self, beta=0.5):
        super(PixWiseBCELoss).__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.beta = beta

    def forward(self, net_mask, net_label, target_mask, target_label):
        pixel_loss = self.criterion(net_mask, target_mask)
        binary_loss = self.criterion(net_label, target_label)
        loss = pixel_loss * self.beta + binary_loss * (1 - self.beta)
        return loss


class PixWise_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, net_mask, target_mask):
        pixel_loss = self.criterion(net_mask, target_mask)
        return pixel_loss


class Moat_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, targets):
        mask_pos = targets > 0
        mask_neg = targets == 0
        color_p = outputs[mask_pos, 1].cpu().data.numpy()
        color_n = outputs[mask_neg, 0].cpu().data.numpy()
        color_dis_p = 1 - color_p  # protocol 4 : 1;
        color_dis_p[color_dis_p < 0] = 0
        color_dis_n = 1 - color_n  # protocol 4 : 1;
        color_dis_n[color_dis_n < 0] = 0
        color_dis_n = color_dis_n.mean()
        color_dis_p = color_dis_p.mean()

        return color_dis_n + color_dis_p


def softmax_cross_entropy_criterion(logit, truth, is_average=True):
    reduction = 'mean' if is_average else 'sum'
    loss = F.cross_entropy(logit, truth, reduction=reduction)
    return loss


def bce_criterion(logit, truth, is_average=True):
    reduction = 'mean' if is_average else 'sum'
    loss = F.binary_cross_entropy_with_logits(logit, truth, reduction=reduction)
    return loss


def init_loss(criterion_name='BCE', device=None, class_num=2):
    if criterion_name == 'BCE':
        loss = F.cross_entropy
    elif criterion_name == 'CCE':
        loss = F.binary_cross_entropy_with_logits
    elif criterion_name == 'focal_loss':
        loss = FocalLoss()
    elif criterion_name == 'moat_loss':
        loss = Moat_Loss()
    elif criterion_name == 'PixWiseBCELoss':
        loss = PixWiseBCELoss()
    elif criterion_name == 'PixWise_Loss':
        loss = PixWise_Loss()
    elif criterion_name == 'CMFL':
        loss = CMFL()
    else:
        raise Exception('This loss function is not implemented yet.')

    return loss


def get_criterion(device=None, class_num=2, label_smoothing=0.0):
    criterion = {}
    if label_smoothing > 0:
        from functools import partial
        criterion['BCE'] = partial(F.cross_entropy, label_smoothing=label_smoothing)
    else:
        criterion['BCE'] = F.cross_entropy
    criterion['CCE'] = F.binary_cross_entropy_with_logits
    criterion['FL'] = FocalLoss()
    criterion['CMFL'] = CMFL()
    criterion['ICMFL'] = ICMFL()
    criterion['PWL'] = PixWise_Loss()
    criterion['ML'] = Moat_Loss()

    return criterion


if __name__ == '__main__':
    loss = init_loss('BCE')
    print(loss)