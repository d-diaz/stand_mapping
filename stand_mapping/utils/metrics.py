import numpy as np
import torch
import torch.nn.functional as F


def batchify(targets, predictions, nodata, score_func,
             aggregate=True, *args, **kwargs):
    """
    Applies a scoring function to each sample image in a batch.

    Parameters
    ----------
    targets : array-like, shape (batch_size, height, width)
      observed, ground-truth target images
    predictions : array-like, shape (batch_size, height, width)
      predicted images
    nodata : array-like, shape (batch_size, height, width)
      nodata masks indicating areas where differences between targets and
      predictions will be ignored
    score_func : callable
      scoring function that will be called on each sample, should expect
      targets, predictions and nodata as arguments, with optional args and
      kwargs to follow.
    aggregate : bool
      whether to return the average of each metrics across the batch (default),
      or to return each of the metrics for each of the samples in the batch
    """
    results = []
    for targ, pred, msk in zip(targets, predictions, nodata):
        # unpack the batch
        # X, Y, Z = batch
        results.append(score_func(targ, pred, msk, *args, **kwargs))

    if aggregate:
        results = tuple(np.array(results).mean(axis=0))

    return results


def masked_accuracy(input, target, nodata=None, reduction='mean'):
    """Calculates classification accuracy for a batch of images.

    Parameters
    ----------
    input : tensor, shape (B, 1, H, W)
      batch of images with predicted classes
    target : tensor, shape (B, 1, H, W)
      batch of images with target classes
    nodata : tensor, shape (B, 1, H, W), optional
      batch of boolean images indicating areas to be excluded from scoring

    Returns
    -------
    score : tensor, shape (B,)
      average accuracy among valid (not nodata) pixels for each of B images
    """
    correct = (input == target)
    support = torch.ones(target.shape)

    if nodata is not None:
        if nodata.dtype != torch.bool:
            nodata = nodata > 0
        correct *= ~nodata
        support *= ~nodata

    score = correct.sum(dim=(1, 2, 3)) / support.sum(dim=(1, 2, 3))

    if reduction == 'mean':
        score = score.mean()
    elif reduction == 'sum':
        score = score.sum()

    return score


def masked_precision(input, target, nodata=None):
    """Calculates classification precision for a batch of images.

    Parameters
    ----------
    input : tensor, shape (B, 1, H, W)
      batch of images with predicted classes
    target : tensor, shape (B, 1, H, W)
      batch of images with target classes
    nodata : tensor, shape (B, 1, H, W), optional
      batch of boolean images indicating areas to be excluded from scoring
    """
    correct = (input == target)
    support = torch.ones(target.shape)

    if nodata is not None:
        if nodata.dtype != torch.bool:
            nodata = nodata > 0
        correct *= ~nodata
        support *= ~nodata

    score = correct.sum(dim=(1, 2, 3)) / support.sum(dim=(1, 2, 3))
    return score


def masked_classification_stats(input, target, nodata=None, num_classes=5):
    """Calculates rates of true and false positives and negatives with
    optional nodata mask.

    Parameters
    input : tensor, shape (B, 1, H, W)
      batch of images with predicted classes
    target : tensor, shape (B, 1, H, W)
      batch of images with target classes
    nodata : tensor, shape (B, 1, H, W), optional
      batch of boolean images indicating areas to be excluded from scoring

    Returns
    -------
    stats : 5-tuple of tensors, each shape (B, N)
      ratio of true positives, true negatives, false positives, and false
      negatives, and support for each of N classes in each of B images in batch
    """
    # convert input and target with shape (B,N,H,W)
    B, C, H, W = input.shape
    hard_pred = torch.argmax(F.softmax(input, dim=1), axis=1)
    input_onehot = F.one_hot(hard_pred, num_classes=num_classes).permute(0,3,1,2)
    targ_onehot = F.one_hot(target[:,0,:,:].clip(0,), num_classes=num_classes).permute(0,3,1,2)
    valid_pixels = H*W

    tp = (input_onehot == targ_onehot) * targ_onehot
    tn = (input_onehot == targ_onehot) * (targ_onehot == 0)
    fp = (input_onehot > targ_onehot)
    fn = (input_onehot < targ_onehot)

    if nodata is not None:
        if nodata.dtype != torch.bool:
            nodata = nodata > 0
        tp *= ~nodata
        tn *= ~nodata
        fp *= ~nodata
        fn *= ~nodata
        valid_pixels = (~nodata).sum(dim=(1, 2, 3)).unsqueeze(-1)

    tp = tp.sum(dim=(2, 3)) / valid_pixels
    tn = tn.sum(dim=(2, 3)) / valid_pixels
    fp = fp.sum(dim=(2, 3)) / valid_pixels
    fn = fn.sum(dim=(2, 3)) / valid_pixels
    support = targ_onehot.sum(dim=(2, 3))

    return (torch.nan_to_num(tp),  # replaces NaNs with zero
            torch.nan_to_num(tn),  # usually where support is 0
            torch.nan_to_num(fp),
            torch.nan_to_num(fn),
            torch.nan_to_num(support))


def masked_dice_coef(input, target, nodata=None, num_classes=5, eps=1e-8):
    """Calculates the Sorensen-Dice Coefficient with the option of including a
    nodata mask.

    Parameters
    ----------
    input : tensor, shape (B, N, H, W)
      logits (unnormalized predictions) for a batch, will be converted to class
      probabilities using softmax.
    target : tensor, shape (B, 1, H, W)
      batch of semantic segmentation target labels.
    nodata : tensor, shape (B, 1, H, W), optional
      boolean or binary tensor where values of True or 1 indicate areas that
      should be excluded from scoring (e.g., where no label was annotated)
    eps : float
      a small value added to denominator of Dice Coefficient for numerical
      stability (prevents divide by zero)

    Returns
    -------
    score : tensor, shape (B,)
      Dice Coefficient for each image in batch
    """
    # compute softmax over the classes dimension
    pred = torch.argmax(F.softmax(input, dim=1), axis=1)

    # convert target to one-hot, then scatter to shape (B,N,H,W)
    one_hot = F.one_hot(target[:,0,:,:].clip(0,), num_classes=num_classes).permute(0,3,1,2)

    if nodata is not None:
        if nodata.dtype != nodata.bool:
            nodata = nodata > 0  # cast to bool
        soft *= ~nodata
        one_hot *= ~nodata

    inter = torch.sum(soft * one_hot, dim=(1, 2, 3))
    card = torch.sum(soft + one_hot, dim=(1, 2, 3))

    score = 2 * inter / (card + eps)

    return score
