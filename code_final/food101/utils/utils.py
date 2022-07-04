

import sys


import numpy as np
import torch
import torch.nn as nn


def min_max_tensor(tensor):
    """
        Normalize tensor
    Args:
        tensor: tensor
    Returns:
        tensor: tensor normalized
    """
    if tensor.max() == tensor.min():
        tensor = torch.zeros(tensor.shape)
    else:
        tensor -= tensor.min()
        tensor /= tensor.max() - tensor.min()
    return tensor


def use_gpu(func):
    """
    Generator for copy pytorch tensors and models to gpu if it exist
    """
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    def wrapper_func(*args, **kwargs):
        args = list(args)
        for i, _ in enumerate(args):
            if isinstance(args[i], (torch.Tensor, nn.Module)):
                try:
                    args[i] = args[i].to(dev)
                except Exception:
                    args[i] = tuple([args[i][j].to(dev) for j in range(len(args[i]))])
        saliency = func(*args, **kwargs)
        return saliency

    return wrapper_func


def print_trainning_steps(count, train_length):
    """
        Print  training state, ==>..|
    Args:
        count: int, actual training step
        train_length: int max training step
    Returns:
        None
    """
    sys.stdout.write(
        "\r"
        + "=" * int(count / train_length * 50)
        + ">"
        + "." * int((train_length - count) / train_length * 50)
        + "|"
        + " * {} %".format(int(count / train_length * 100))
    )
    sys.stdout.flush()
    if count == train_length:
        sys.stdout.write("\n")
        sys.stdout.flush()
