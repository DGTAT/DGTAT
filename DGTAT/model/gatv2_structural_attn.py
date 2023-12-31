from typing import Union, Tuple, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import (
    remove_self_loops,
    add_self_loops,
    softmax,
)
from torch_geometric.nn.inits import glorot, zeros


class structural_attn_layer(MessagePassing):
    _alpha: OptTensor
    def __init__(self, in_channels: int,
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,
                 bias: bool = True,share_weights: bool = False,
                 **kwargs):
        super(structural_attn_layer, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_weights = share_weights
        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
        
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)
        
        self.att = Parameter(torch.Tensor(1, heads, out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, s: Union[Tensor, PairTensor], edge_index: torch.ones_like(Adj),
                size: Size = None, return_attention_weights: bool = None):
        H, C = self.heads, self.out_channels
        s_l: OptTensor = None
        s_r: OptTensor = None
        if isinstance(s, Tensor):
            assert s.dim() == 2
            s_l = self.lin_l(s).view(-1, H, C)
            if self.share_weights:
                s_r = s_l
            else:
                s_r = self.lin_r(s).view(-1, H, C)
        else:
            s_l, s_r = s[0], s[1]
            assert s[0].dim() == 2
            s_l = self.lin_l(s_l).view(-1, H, C)
            if s_r is not None:
                s_r = self.lin_r(s_r).view(-1, H, C)
        assert s_l is not None
        assert s_r is not None
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                num_nodes = s_l.size(0)
                if s_r is not None:
                    num_nodes = min(num_nodes, s_r.size(0))
                if size is not None:
                    num_nodes = min(size[0], size[1])
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            elif isinstance(edge_index, SparseTensor):
                edge_index = set_diag(edge_index)
        out = self.propagate(edge_index, s=(s_l, s_r), size=size)
        alpha = self._alpha
        self._alpha = None
        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        if self.bias is not None:
            out += self.bias
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out
        
    def message(self, s_j: Tensor, s_i: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        s = s_i + s_j
        s = F.leaky_relu(s, self.negative_slope)
        alpha = (s * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return s_j * alpha.unsqueeze(-1)
    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)