import torch
import torchvision


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Inverse of normalization transform. Takes a normalized image and returns
    the pixel values in the input domain.

    Parameters
    ----------
    mean : sequence
      sequence of means for each channel
    std : sequence
      sequence of standard deviations for each channel
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def distance_weight(x, m=0.5, s=5, vmin=None, vmax=None, eps=1e-23):
    """Assigns a weight in the range of [0,1] to a distance measure using a
    logistic function.

    Parameters
    ----------
    x : tensor
      distances that will be transformed into weights
    m : float
      value between 0.0 and 1.0 indicating the inflection point of the weight
      function, which will occur at the m proportion of the range of `x` (or
      the range set using `vmin` and/or `vmax`). For example, at the default
      value of 0.5, the inflection point will be half-way between the min and
      max values of `x`.
    s : numeric
      shape parameter for the logistic, larger values create a steeper curve.
      negative values will flip the weight function such that smaller values of
      `x` are closer to 0 and larger values of `x` are closer to 1.
    eps : numeric
      a smaller number added to `x` after it has been rescaled to [0,1] for
      numerical stability (helps avoid errors or warnings due to dividing by
      zero).
    """
    if vmin is not None:
        x = torch.clip(x, vmin, None)
    else:
        vmin = x.min()
    if vmax is not None:
        x = torch.clip(x, None, vmax)
    else:
        vmax = x.max()
    if vmin is None and vmax is None:
        vmin, vmax = x.min(), x.max()

    x = (x - vmin) / (vmax - vmin)
    w = 1 / (1 + ((x * (1 - m)) / (m * (1 - x) + eps) + eps)**(s))

    return w
