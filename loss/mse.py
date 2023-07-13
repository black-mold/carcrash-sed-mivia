import torch.nn.functional as F
import torch.nn as nn

def loss_function(y_hat, target):
    return F.mse_loss(y_hat, target)