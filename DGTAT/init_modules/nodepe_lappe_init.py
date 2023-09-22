from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from numpy.linalg import eigvals
from torch_geometric.utils import (get_laplacian, to_scipy_sparse_matrix, to_undirected, to_dense_adj, scatter)

nodepe_lappe_init.eigen.laplacian_norm = 'sym'
nodepe_lappe_init.eigen.eigvec_norm = 'L2'
"""The lowest k evals"""
nodepe_lappe_init.eigen.max_freqs = 5

def concat_LapPE(data: Data, value: Any,) -> Data:
    if 'x' in data:
        x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
        data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
    else:
        data.x = value
        
    return data

def compute_LapPE(data):
    """compute LapPE for the graph. """
    if hasattr(data, 'num_nodes'):
        N = data.num_nodes 
    else:
        N = data.x.shape[0]    
    
    evals, evects = None, None   
    
    L = to_scipy_sparse_matrix(
        *get_laplacian(undir_edge_index, normalization=laplacian_norm_type,
                        num_nodes=N)
    )
    evals, evects = np.linalg.eigh(L.toarray())
    
    max_freqs = LapPE_init.eigen.max_freqs
    eigvec_norm = LapPE_init.eigen.eigvec_norm
    
    data.EigVals, data.EigVecs = LapPE_decomp(
        evals=evals, evects=evects,
        max_freqs=max_freqs,
        eigvec_norm=eigvec_norm)
    
    data = concat_LapPE(data, EigVecs)
    
    return data    

def LapPE_decomp(evals, evects, max_freqs, eigvec_norm='L2'):
    N = len(evals) 
    idx = evals.argsort()[:max_freqs]
    evals, evects = evals[idx], np.real(evects[:, idx])
    evals = torch.from_numpy(np.real(evals)).clamp_min(0)
    evects = torch.from_numpy(evects).float()
    evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
    
    if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
    else:
        EigVecs = evects
        
    if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
    else:
        EigVals = evals.unsqueeze(0)
        
    EigVals = EigVals.repeat(N, 1).unsqueeze(2)
    
    return EigVals, EigVecs

