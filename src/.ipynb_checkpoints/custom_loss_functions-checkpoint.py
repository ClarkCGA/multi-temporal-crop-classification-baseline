import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedCrossEntropyLoss(nn.Module):
    """
    Balanced cross entropy loss by weighting of inverse class ratio.

    Args:
        ignore_index (int): Class index to ignore.
        reduction (str): Reduction method to apply to loss.
                         Options: 'mean', 'sum', 'none'.
        weight_scheme (str): Strategy to weight samples. Options:
                      "icr" -- inverse class ratio
                      "mcf" -- median class frequency

    Returns:
        Loss tensor according to the specified reduction.
    """

    def __init__(self, ignore_index=-100, reduction="mean", weight_scheme="icr"):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert weight_scheme in ["icr", "mcf"], "'weight_scheme' is not recognized."
        self.weight_scheme = weight_scheme

    def forward(self, predict, target):
        """
        Args:
            predict (torch.Tensor): Predicted output tensor.
            target (torch.Tensor): Target tensor.
        """

        class_counts = torch.bincount(target.view(-1), minlength=predict.shape[1])
        # get class weights
        if self.weight_scheme == "icr":
            class_weights = 1.0 / torch.sqrt(class_counts.float())
        else:
            median_frequency = torch.median(class_counts.float())
            class_weights = median_frequency / class_counts.float()

        # set weight of ignore index to 0
        if self.ignore_index >= 0 and self.ignore_index < len(class_weights):
            class_weights[self.ignore_index] = 0

        # normalize weights
        class_weights /= torch.sum(class_weights)

        # apply class weights to loss function
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, ignore_index=self.ignore_index,
                                      reduction=self.reduction)

        return loss_fn(predict, target)

class OhemCrossEntropyLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) Cross Entropy Loss for Semantic Segmentation
    Params:
        ignore_index (int): Class index to ignore
        reduction (str): Reduction method to apply to loss, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
        ohem_ratio (float): Ratio of hard examples to use in the loss function

    Returns:
        Loss tensor according to arg reduction
    """
    def __init__(self, ignore_index=-100, reduction='mean', ohem_ratio=0.25):
        super(OhemCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ohem_ratio = ohem_ratio

    def forward(self, predict, target):
        # calculate pixel-wise cross entropy loss
        loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        pixel_losses = loss_fn(predict, target)

        # apply online hard example mining
        num_hard = int(self.ohem_ratio * pixel_losses.numel())
        _, top_indices = pixel_losses.flatten().topk(num_hard)
        ohem_losses = pixel_losses.flatten()[top_indices]

        # apply reduction to ohem losses
        if self.reduction == 'mean':
            loss = ohem_losses.mean()
        elif self.reduction == 'sum':
            loss = ohem_losses.sum()
        else:
            loss = ohem_losses

        return loss


class BinaryDiceLoss(nn.Module):
    '''
        Dice loss of binary class
        Params:
            smooth (float): A float number to smooth loss, and avoid NaN error, default: 1
            p (int): Denominator value: \sum{x^p} + \sum{y^p}, default: 2. Used
                     to control the sensitivity of the loss.
            predict (torch.tensor): Predicted tensor of shape [N, *]
            target (torch.tensor): Target tensor of same shape with predict
        Returns:
            Loss tensor
    '''
    def __init__(self, smooth=1, p=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):

        assert predict.shape == target.shape, "predict & target shape do not match"
        assert predict.shape == target.shape, "predict & target shapes do not match"
        assert predict.dtype == target.dtype, "predict & target data types do not match"

        predict = predict.contiguous().view(-1)
        target = target.contiguous().view(-1)

        num = 2 * (predict * target).sum() + self.smooth
        den = (predict.pow(self.p) + target.pow(self.p)).sum() + self.smooth
        loss = 1 - num / den

        return loss


class DiceLoss(nn.Module):
    r"""
    Dice loss

    Arguments:
        weight (torch.tensor): Weight array of shape [num_classes,]
        ignore_index (int): Class index to ignore
        predict (torch.tensor): Predicted tensor of shape [N, C, *]
        target (torch.tensor): Target tensor either in shape [N,*] or of same shape with predict
        other args pass to BinaryDiceLoss
    Returns:
        same as BinaryDiceLoss
    """

    def __init__(self, weight=None, ignore_index=-100, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        nclass = predict.shape[1]
        if predict.shape == target.shape:
            pass
        elif len(predict.shape) == 4:
            target = F.one_hot(target, num_classes=nclass).permute(0, 3, 1, 2).contiguous()
        else:
            assert 'Predict tensor shape of {} is not assceptable.'.format(predict.shape)

        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        weight = torch.Tensor([1. / nclass] * nclass).cuda() if self.weight is None else self.weight
        predict = F.softmax(predict, dim=1)

        for i in range(nclass):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])

                assert weight.shape[0] == nclass, \
                    'Expected weight tensor with shape [{}], but got[{}]'.format(nclass, weight.shape[0])
                dice_loss *= weight[i]
                total_loss += dice_loss

        return total_loss


class BalancedDiceLoss(nn.Module):
    """
    Dice Loss weighted by inverse of label frequency

    Arguments:
        ignore_index (int): Class index to ignore
        **kwargs: Additional arguments passed to BinaryDiceLoss

    Returns:
        Loss tensor
    """

    def __init__(self, ignore_index=-100, **kwargs):
        super(BalancedDiceLoss, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        class_weights = self.calculate_class_weights(target)

        loss_weight = torch.ones(predict.shape[1], device=predict.device) * 0.00001
        for i, weight in enumerate(class_weights):
            loss_weight[i] = weight

        loss = DiceLoss(weight=loss_weight, ignore_index=self.ignore_index, **self.kwargs)

        return loss(predict, target)

    def calculate_class_weights(self, target):
        unique, unique_counts = torch.unique(target[target != self.ignore_index], return_counts=True)
        class_ratios = unique_counts.float() / torch.numel(target)
        class_weights = 1.0 / class_ratios
        class_weights /= torch.sum(1. / class_weights)

        return class_weights


class DiceCELoss(nn.Module):
    """
    Combination of dice loss and cross entropy loss through summation

    Arguments:
        loss_weight (Tensor, optional): A manual rescaling weight given to each class.
                                        If provided, should be a Tensor of size C
        dice_weight (float): Weight on dice loss for the summation, while the weight
                             on cross-entropy loss is (1 - dice_weight)
        dice_smooth (float, optional): A float number to smooth dice loss and avoid NaN error.
                                       Default: 1
        dice_p (int, optional): Denominator value: \sum{x^p} + \sum{y^p}. Default: 1
        ignore_index (int, optional): Class index to ignore. Default: None

    Returns:
        Loss tensor
    """

    def __init__(self, loss_weight=None, dice_weight=0.5, dice_smooth=1,
                 dice_p=1, ignore_index=-100):

        super(DiceCELoss, self).__init__()
        self.loss_weight = loss_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.dice_p = dice_p
        self.ignore_index = ignore_index

        self.dice_loss = DiceLoss(weight=self.loss_weight, ignore_index=self.ignore_index,
                                  smooth=self.dice_smooth, p=self.dice_p)
        self.ce_loss = nn.CrossEntropyLoss(weight=self.loss_weight, ignore_index=self.ignore_index)

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size do not match"

        loss = self.dice_weight * self.dice_loss(predict, target) + (1 - self.dice_weight) * self.ce_loss(predict, target)

        return loss


class BalancedDiceCELoss(nn.Module):
    r"""
    Dice Cross Entropy weighted by inverse of label frequency
    Arguments:
        ignore_index (int): Class index to ignore
        predict (torch.tensor): Predicted tensor of shape [N, C, *]
        target (torch.tensor): Target tensor either in shape [N,*] or of same shape with predict
        other args pass to DiceCELoss, excluding loss_weight
    Returns:
        Same as DiceCELoss
    """

    def __init__(self, ignore_index=-100, **kwargs):
        super(BalancedDiceCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.kwargs = kwargs

    def forward(self, predict, target):
        # get class weights
        class_weights = self.calculate_class_weights(target)
        loss_weight = torch.ones(predict.shape[1], device=predict.device) * 0.00001

        for i, weight in enumerate(class_weights):
            loss_weight[i] = weight

        loss = DiceCELoss(loss_weight=loss_weight, **self.kwargs)

        return loss(predict, target)

    def calculate_class_weights(self, target):
        unique, unique_counts = torch.unique(target[target != self.ignore_index], return_counts=True)
        class_ratios = unique_counts.float() / torch.numel(target)
        class_weights = 1.0 / class_ratios
        class_weights /= torch.sum(1. / class_weights)

        return class_weights

