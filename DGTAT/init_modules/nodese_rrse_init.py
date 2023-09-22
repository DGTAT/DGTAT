import torch
import torch.nn.functional as F
import pickle
import torch.utils.data
import time
import os
import numpy as np
import csv
import dgl
from scipy import sparse as sp
import numpy as np
import networkx as nx

nodese_rrse_init.rwse._dim = 5

def compute_rrse(g, rwse_dim):
    """Initializing node structral encoding with RWPE"""
    
    n = g.number_of_nodes()
    A = g.adjacency_matrix(scipy_fmt="csr")
    Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
    RW = A * Dinv  
    M = RW
    nb_pos_enc = rwse_dim
    SE = [torch.from_numpy(M.diagonal()).float()]
    M_power = M
    
    for _ in range(nb_pos_enc-1):
        M_power = M_power * M
        SE.append(torch.from_numpy(M_power.diagonal()).float())
        
    SE = torch.stack(SE,dim=-1)
    g.ndata['pos_enc'] = SE    
    
    return g

class RRSENodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim, out_dim, use_bias=False, batchnorm=False, layernorm=False, se_name="rrse"):
        super().__init__()
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.name = se_name
        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)
            
    def forward(self, batch):
        rrwp = batch[f"{self.name}"]
        rrwp = self.fc(rrse)

        if self.batchnorm:
            rrwp = self.bn(rrse)

        if self.layernorm:
            rrwp = self.ln(rrse)
        if "x" in batch:
            batch.x = batch.x + rrse
        else:
            batch.x = rrse
            
        return batch