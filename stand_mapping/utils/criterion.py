import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
from kornia.losses.focal import FocalLoss, focal_loss
from .metrics import masked_dice_coef


class DiscriminativeLoss(_Loss):
    """
    The code is from
    https://github.com/nyoki-mtl/pytorch-discriminative-loss
    This is the implementation of following paper:
    https://arxiv.org/pdf/1802.05591.pdf
    This implementation is based on following code:
    https://github.com/Wizaron/instance-segmentation-pytorch
    """
    def __init__(self, delta_var=0.5, delta_dist=1.5,
                 norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=True, reduction='mean'):
        super(DiscriminativeLoss, self).__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target, n_clusters):
        assert not target.requires_grad
        return self._discriminative_loss(input, target, n_clusters)

    def _discriminative_loss(self, input, target, n_clusters):
        bs, n_features, height, width = input.size()
        max_n_clusters = target.size(1)

        input = input.contiguous().view(bs, n_features, height * width)
        target = target.contiguous().view(bs, max_n_clusters, height * width)

        c_means = self._cluster_means(input, target, n_clusters)
        l_var = self._variance_term(input, target, c_means, n_clusters)
        l_dist = self._distance_term(c_means, n_clusters)
        l_reg = self._regularization_term(c_means, n_clusters)

        loss = self.alpha * l_var + self.beta * l_dist + self.gamma * l_reg

        return loss

    def _cluster_means(self, input, target, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features,
                                          max_n_clusters, n_loc)
        # bs, 1, max_n_clusters, n_loc
        target = target.unsqueeze(1)
        # bs, n_features, max_n_clusters, n_loc
        input = input * target

        means = []
        for i in range(bs):
            # n_features, n_clusters, n_loc
            input_sample = input[i, :, :n_clusters[i]]
            # 1, n_clusters, n_loc,
            target_sample = target[i, :, :n_clusters[i]]
            # n_features, n_cluster
            mean_sample = input_sample.sum(2) / target_sample.sum(2)

            # padding
            n_pad_clusters = max_n_clusters - n_clusters[i]
            assert n_pad_clusters >= 0
            if n_pad_clusters > 0:
                pad_sample = torch.zeros(n_features, n_pad_clusters)
                if self.usegpu:
                    pad_sample = pad_sample.cuda()
                mean_sample = torch.cat((mean_sample, pad_sample), dim=1)
            means.append(mean_sample)

        # bs, n_features, max_n_clusters
        means = torch.stack(means)

        return means

    def _variance_term(self, input, target, c_means, n_clusters):
        bs, n_features, n_loc = input.size()
        max_n_clusters = target.size(1)

        # bs, n_features, max_n_clusters, n_loc
        c_means = c_means.unsqueeze(3).expand(bs, n_features,
                                              max_n_clusters, n_loc)
        # bs, n_features, max_n_clusters, n_loc
        input = input.unsqueeze(2).expand(bs, n_features,
                                          max_n_clusters, n_loc)
        # bs, max_n_clusters, n_loc
        var = (torch.clamp(torch.norm((input - c_means), self.norm, 1)
               - self.delta_var, min=0) ** 2) * target

        var_term = 0
        for i in range(bs):
            # n_clusters, n_loc
            var_sample = var[i, :n_clusters[i]]
            # n_clusters, n_loc
            target_sample = target[i, :n_clusters[i]]

            # n_clusters
            c_var = var_sample.sum(1) / target_sample.sum(1)
            var_term += c_var.sum() / n_clusters[i]
        var_term /= bs

        return var_term

    def _distance_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        dist_term = 0
        for i in range(bs):
            if n_clusters[i] <= 1:
                continue

            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]

            # n_features, n_clusters, n_clusters
            means_a = (mean_sample
                       .unsqueeze(2)
                       .expand(n_features, n_clusters[i], n_clusters[i]))
            means_b = means_a.permute(0, 2, 1)
            diff = means_a - means_b

            margin = 2 * self.delta_dist * (1.0 - torch.eye(n_clusters[i]))
            if self.usegpu:
                margin = margin.cuda()
            c_dist = torch.sum(
                         torch.clamp(margin - torch.norm(
                             diff, self.norm, 0), min=0) ** 2)
            dist_term += c_dist / (2 * n_clusters[i] * (n_clusters[i] - 1))
        dist_term /= bs

        return dist_term

    def _regularization_term(self, c_means, n_clusters):
        bs, n_features, max_n_clusters = c_means.size()

        reg_term = 0
        for i in range(bs):
            # n_features, n_clusters
            mean_sample = c_means[i, :, :n_clusters[i]]
            reg_term += torch.mean(torch.norm(mean_sample, self.norm, 0))
        reg_term /= bs

        return reg_term


class MaskedFocalLoss(FocalLoss):
    """Focal Loss extended to accept a nodata mask.
    """

    def __init__(self, alpha, gamma=2, reduction='none', eps=1e-8):
        super().__init__(alpha=alpha, gamma=gamma, reduction=reduction)
        self.reduction = reduction
        self.eps = eps

    def __call__(self, input, target, nodata=None):
        loss = focal_loss(input, target[:,0,:,:],
                          alpha=self.alpha, gamma=self.gamma,
                          reduction='none', eps=self.eps).unsqueeze(1)

        if nodata is None:
            nodata = torch.ones(loss.shape) == 0

        if self.reduction == 'mean':
            loss = loss[~nodata].mean()
        elif self.reduction == 'sum':
            loss = loss[~nodata].sum()
        elif self.reduction == 'none':
            loss = loss * ~nodata

        return loss


class MaskedCrossEntropyLoss(_WeightedLoss):
    """Cross Entropy Loss extended to accept a nodata mask. Nodata is expected
    to be a boolean mask where nodata values are indicated by True and valid
    data values are indicated by False. Weights are per-class. Loss is set
    to zero for nodata pixels.
    """

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def __call__(self, input, target, nodata=None):
        loss = F.cross_entropy(input, target[:,0,:,:],
                               reduction='none',
                               weight=self.weight).unsqueeze(1)

        if nodata is None:
            nodata = torch.ones(loss.shape) == 0

        if self.reduction == 'mean':
            if self.weight is not None:
                loss = loss[~nodata].sum() / self.weight[target[~nodata]].sum()
            else:
                loss = loss[~nodata].mean()
        elif self.reduction == 'sum':
            loss = loss[~nodata].sum()
        elif self.reduction == 'none':
            loss = loss * ~nodata

        return loss


class MaskedDiceLoss(_Loss):
    def __init__(self, reduction='mean', eps=1e-8):
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def __call__(self, input, target, nodata=None):
        score = masked_dice_coef(input, target, nodata=nodata, eps=self.eps)
        loss = 1 - score

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()

        return loss
