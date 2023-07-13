import torch.nn.functional as F
import torch.nn as nn

def loss_function(y_hat, target):
    return F.binary_cross_entropy(y_hat, target)