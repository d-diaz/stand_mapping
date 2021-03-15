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
