import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from dgl.nn.pytorch.conv import TAGConv







class DrugGraphConv(nn.Module):
    def __init__(self, num_drug_features=74, drug_out_dim=37) -> None:
        super().__init__()

        self.drug_out_dim = drug_out_dim

        self.drug_graph_conv = nn.ModuleList()
        self.drug_graph_conv.append(TAGConv(num_drug_features, 70 ,2))
        self.drug_graph_conv.append(TAGConv(70, 65,2))
        self.drug_graph_conv.append(TAGConv(65, 60,2))
        self.drug_graph_conv.append(TAGConv(60, 55,2))
        self.drug_graph_conv.append(TAGConv(55, drug_out_dim,2))


        _gate_nn = nn.Linear(drug_out_dim, 1)
        self.pooling_drug = GlobalAttentionPooling(_gate_nn)

    def forward(self, g):
        ndata = g.ndata['h']

        for module in self.drug_graph_conv:
            g = g.to(ndata.device)
            ndata = F.relu(module(g,ndata))
        out = self.pooling_drug(g, ndata)
        out = out.view(-1, 2, self.drug_out_dim)
        return out


'''
The implementation of part of the code of "MultiHeadAttention" and "scaled_dot_product" is quoted from
https://github.com/yazdanimehdi/AttentionSiteDTI/blob/main/layers.py
'''
def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        if return_attention:
            return o, attention
        else:
            return o



