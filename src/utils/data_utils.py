import numpy as np
import torch


def igibson_collate(samples):
    imgs = torch.stack([img for img, _ in samples])
    targets = []
    for _, target in samples:
        targets.append(target)
    return imgs, targets
