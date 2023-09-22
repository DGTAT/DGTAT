import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import torch
import pickle

class positional_attn_layer(MessagePassing):
    def __init__(self, in_channels: int,node_num: int, 
                 **kwargs):
        super(positional_attn_layer, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.node_num = node_num

    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index
    
    def normalize_features(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def positional_attn(jaccard_index, jaccard_all, g):
        """compute the positional_attn"""
        for i in range(len(jaccard_index)):
            for j in range(len(jaccard_index))
                intersection = set(jaccard_index[i]).intersection(set(jaccard_index[j]))
                union = set(jaccard_index[i]).union(set(jaccard_index[j]))
                intersection = list(intersection)
                union = list(union)
                intersection_jaccard_alli = []
                intersection_jaccard_allj = []
                union_jaccard_alli = []
                union_jaccard_allj = []
                g[i][j] = 0
                if len(intersection) == 0:
                    g[i][j] = 0
                    continue
                else:
                    for k in range(len(intersection)):
                        intersection_jaccard_alli.append(jaccard_all[i][jaccard_index[i].tolist().index(intersection[k])])
                        intersection_jaccard_allj.append(jaccard_all[j][jaccard_index[j].tolist().index(intersection[k])])
                    union_rest = set(union).difference(set(intersection))
                    union_rest = list(union_rest)
                    if i==j and len(union_rest) == 0:
                        g[i][j] = 0.5
                        break
                    else:
                        for k in range(len(union_rest)):
                            if union_rest[k] in jaccard_index[i]:
                                union_jaccard_alli.append(jaccard_all[i][jaccard_index[i].tolist().index(union_rest[k])])#i-j
                            else:
                                union_jaccard_allj.append(jaccard_all[j][jaccard_index[j].tolist().index(union_rest[k])])#j-i
                    k_max = np.maximum(intersection_jaccard_alli,intersection_jaccard_allj)
                    k_min = np.minimum(intersection_jaccard_alli,intersection_jaccard_allj)
                    k_max = k_max.tolist()
                    k_min = k_min.tolist()
                    union_jaccard_allj = k_max + union_jaccard_allj + union_jaccard_alli 
                    union_num = np.sum(np.array(union_jaccard_allj), axis=0)
                    inter_num = np.sum(np.array(k_min), axis=0)
                    g[i][j] = inter_num / union_num
        return g
    
    def forward(self, jaccard_index, jaccard_all, Adj):
        return self.positional_attn(jaccard_index, jaccard_all, torch.zeros_like(Adj))
