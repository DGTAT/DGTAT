from typing import Union, Tuple, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
import gatv2_structural_attn.py
import jaccard_positional_attn.py


class dgtat_layer(MessagePassing):
    
    def __init__(self, ae_in_channels: int, ae_out_channels: int, 
                 se_in_channels: int, se_out_channels: int,
                 pe_channels: int, edge_channels: int,
                 heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.,bias: bool = True,
                 **kwargs):
        
        super(dgtat_layer, self).__init__(node_dim=0, **kwargs)
        self.ae_in_channels = ae_in_channels
        self.ae_out_channels = ae_out_channels
        self.se_in_channels = se_in_channels
        self.se_out_channels = se_out_channels
        self.pe_channels = pe_channels
        self.edge_channels = edge_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.lin_l = Linear(ae_in_channels, heads * ae_out_channels, bias=bias)
        self.structralattn = structural_attn_layer(se_in_channels, heads * se_out_channels, bias=bias)
        self.positionalattn = positional_attn_layer(pe_channels)
        self.att = Parameter(torch.Tensor(1, heads, ae_out_channels))   
        self.a = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.b = nn.Parameter(torch.zeros(size=(1, 1)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.W_A = nn.Parameter(torch.zeros(size=(1, ae_out_channels)))
        nn.init.xavier_uniform_(self.W_A.data, gain=1.414)
        self.W_EW = nn.Parameter(torch.zeros(size=(edge_channels, ae_out_channels)))
        nn.init.xavier_uniform_(self.W_EW.data, gain=1.414)   
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * ae_out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(ae_out_channels))
        else:
            self.register_parameter('bias', None)
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.structralattn.weight)
        glorot(self.positionalattn)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_feature: edge_input,
                size: Size = None, return_attention_weights: bool = None):
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
                
        S = structural_attn_layer(s,torch.ones_like(Adj))
        P = positional_attn_layer(jaccard_index, jaccard_all, torch.zeros_like(Adj))
        
        edge_out = SQRT(relu(dot(self.propagate(edge_index, x=(x_l, x_r), size=size),torch.matmul(W_EW, edge_feature)))+torch.matmul(W_Eb, edge_feature,S)).view(edge_input.shape[0],edge_channels)
        a_n = a*S + b*p + torch.matmul(W_A,edge_out)
        
        Sp = dot(S,P)
        Sp = Sample_K_mask(Sp)
        
        a_s = a*S + b*p + SQRT(relu(self.propagate(edge_index, x=(x_l, x_r), size=size)))
        a_all = a_n + a_s
        
        x_out = torch.matmul(a_all, x)
        p_out = relu(P + relu(torch.matmul(W_P, a_all)))

        if self.bias is not None:
            x_out += self.bias
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return x_out, p_out

    def message(self, x_j: Tensor, x_i: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)