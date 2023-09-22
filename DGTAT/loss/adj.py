import torch.nn as nn
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss

def adj_l1_losses(pred, true):
    # pred: (idx, val) ; true: (idx, val)
    pred_idx, pred_val = pred[0], pred[1]
    true_idx, true_val = true[0], true[1]

    if cfg.model.loss_fun == 'adj_l1':
        l1_loss = nn.L1Loss()
        loss = l1_loss(pred, true)
        return loss, pred
    elif cfg.model.loss_fun == 'adj_smoothl1':
        l1_loss = nn.SmoothL1Loss()
        loss = l1_loss(pred, true)
        return loss, pred
