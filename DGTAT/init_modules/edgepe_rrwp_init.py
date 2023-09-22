from typing import Union, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import BaseTransform
from torch_scatter import scatter, scatter_add, scatter_max
from torch_geometric.graphgym.config import cfg
from torch_geometric.utils import (
    get_laplacian,
    get_self_loop_attr,
    to_scipy_sparse_matrix,
)
import torch_sparse
from torch_sparse import SparseTensor


def concat_rrwp(data: Data, value: Any,
                  attr_name: Optional[str] = None) -> Data:
    if attr_name is None:
        if 'x' in data:
            x = data.x.view(-1, 1) if data.x.dim() == 1 else data.x
            data.x = torch.cat([x, value.to(x.device, x.dtype)], dim=-1)
        else:
            data.x = value
    else:
        data[attr_name] = value
    return data


def compute_rrwp(data,
                  walk_length=8,
                  attr_name_abs="rrwp",
                  attr_name_rel="rrwp",
                  add_identity=True,
                  spd=False,
                  **kwargs
                  ):
    device=data.edge_index.device
    ind_vec = torch.eye(walk_length, dtype=torch.float, device=device)
    num_nodes = data.num_nodes
    edge_index, edge_weight = data.edge_index, data.edge_weight
    adj = SparseTensor.from_edge_index(edge_index, edge_weight,
                                       sparse_sizes=(num_nodes, num_nodes),
                                       )
    deg = adj.sum(dim=1)
    deg_inv = 1.0 / adj.sum(dim=1)
    deg_inv[deg_inv == float('inf')] = 0
    adj = adj * deg_inv.view(-1, 1)
    adj = adj.to_dense()
    pe_list = []
    i = 0
    if add_identity:
        pe_list.append(torch.eye(num_nodes, dtype=torch.float))
        i = i + 1
    out = adj
    pe_list.append(adj)
    if walk_length > 2:
        for j in range(i + 1, walk_length):
            out = out @ adj
            pe_list.append(out)
    pe = torch.stack(pe_list, dim=-1) 
    abs_pe = pe.diagonal().transpose(0, 1) 
    rel_pe = SparseTensor.from_dense(pe, has_value=True)
    rel_pe_row, rel_pe_col, rel_pe_val = rel_pe.coo()
    rel_pe_idx = torch.stack([rel_pe_row, rel_pe_col], dim=0)
    if spd:
        spd_idx = walk_length - torch.arange(walk_length)
        val = (rel_pe_val > 0).type(torch.float) * spd_idx.unsqueeze(0)
        val = torch.argmax(val, dim=-1)
        rel_pe_val = F.one_hot(val, walk_length).type(torch.float)
        abs_pe = torch.zeros_like(abs_pe)
    data = concat_rrwp(data, abs_pe, attr_name=attr_name_abs)
    data = concat_rrwp(data, rel_pe_idx, attr_name=f"{attr_name_rel}_index")
    data = concat_rrwp(data, rel_pe_val, attr_name=f"{attr_name_rel}_val")
    data.log_deg = torch.log(deg + 1)
    data.deg = deg.type(torch.long)
    return data

class RRWPEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, out_dim, batchnorm=False, layernorm=False, use_bias=False,
                 pad_to_full_graph=True, fill_value=0.,
                 add_node_attr_as_self_loop=False,
                 overwrite_old_attr=False):
        super().__init__()
        # note: batchnorm/layernorm might ruin some properties of pe on providing shortest-path distance info
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.add_node_attr_as_self_loop = add_node_attr_as_self_loop
        self.overwrite_old_attr=overwrite_old_attr # remove the old edge-attr
        self.batchnorm = batchnorm
        self.layernorm = layernorm
        self.fc = nn.Linear(emb_dim, out_dim, bias=use_bias)
        torch.nn.init.xavier_uniform_(self.fc.weight)
        self.pad_to_full_graph = pad_to_full_graph
        self.fill_value = 0.
        padding = torch.ones(1, out_dim, dtype=torch.float) * fill_value
        self.register_buffer("padding", padding)
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)
        if self.layernorm:
            self.ln = nn.LayerNorm(out_dim)

    def forward(self, batch):
        rrwp_idx = batch.rrwp_index
        rrwp_val = batch.rrwp_val
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        rrwp_val = self.fc(rrwp_val)

        if edge_attr is None:
            edge_attr = edge_index.new_zeros(edge_index.size(1), rrwp_val.size(1))
        if self.overwrite_old_attr:
            out_idx, out_val = rrwp_idx, rrwp_val
        else:
            edge_index, edge_attr = add_remaining_self_loops(edge_index, edge_attr, num_nodes=batch.num_nodes, fill_value=0.)
            out_idx, out_val = torch_sparse.coalesce(
                torch.cat([edge_index, rrwp_idx], dim=1),
                torch.cat([edge_attr, rrwp_val], dim=0),
                batch.num_nodes, batch.num_nodes,
                op="add"
            )

        if self.pad_to_full_graph:
            edge_index_full = full_edge_index(out_idx, batch=batch.batch)
            edge_attr_pad = self.padding.repeat(edge_index_full.size(1), 1)
            # zero padding to fully-connected graphs
            out_idx = torch.cat([out_idx, edge_index_full], dim=1)
            out_val = torch.cat([out_val, edge_attr_pad], dim=0)
            out_idx, out_val = torch_sparse.coalesce(
               out_idx, out_val, batch.num_nodes, batch.num_nodes,
               op="add"
            )

        if self.batchnorm:
            out_val = self.bn(out_val)

        if self.layernorm:
            out_val = self.ln(out_val)


        batch.edge_index, batch.edge_attr = out_idx, out_val
        return batch
    def __repr__(self):
        return f"{self.__class__.__name__}" \
               f"(pad_to_full_graph={self.pad_to_full_graph}," \
               f"fill_value={self.fill_value}," \
               f"{self.fc.__repr__()})"
