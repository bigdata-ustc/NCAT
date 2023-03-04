import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
import torch.optim as optim
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from util import tensor_to_numpy
import logger

class NCAT(nn.Module):
    def __init__(self, n_question, d_model, n_blocks,
                 kq_same, dropout, policy_fc_dim=512, n_heads=1, d_ff=2048,  l2=1e-5, separate_qa=None, pad=0):
        super().__init__()
        """
        Input:
            d_model: question emb and dimension of attention block 
            final_fc_dim: dimension of final fully connected net before prediction
            n_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
        """
        self.device = torch.device('cuda')
        self.pad = pad
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.separate_qa = separate_qa
        embed_l = d_model
        self.q_embed_0 = nn.Embedding(self.n_question, embed_l) # 两个通道是否用相同embedding
        self.q_embed_1 = nn.Embedding(self.n_question, embed_l)
        self.contradiction = MultiHeadedAttention_con(n_heads, d_model)
        self.con_ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_heads, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.self_atten_0 = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_blocks)
        self.self_atten_1 = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), n_blocks)
        
        self.policy_layer = nn.Sequential(
            nn.Linear(d_model * 4, policy_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(policy_fc_dim, n_question)
        )

        for name, param in self.named_parameters():
            # print(name)
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    def forward(self, p_0_rec, p_1_rec, p_0_target, p_1_target):
        # embedding layer
        bs = len(p_0_rec)
        item_emb_0 = self.q_embed_0(p_0_rec) # C
        item_emb_1 = self.q_embed_1(p_1_rec) # K
        src_mask_0 = mask(p_0_rec, p_0_target + 1).unsqueeze(-2)
        src_mask_1 = mask(p_1_rec, p_1_target + 1).unsqueeze(-2)
        item_per_0 = self.self_atten_0(item_emb_0, src_mask_0) # bs len emb_dim
        # print(item_per_0.shape)
        item_per_1 = self.self_atten_1(item_emb_1, src_mask_1)
        # contradiction learning
        input_01, input_10 = self.contradiction(item_emb_0, item_emb_1, item_per_1, item_per_0)
        # print('???', input_01.shape)
        input_01, input_10 = input_01.mean(-2), input_10.mean(-2)
        
        # cat 
        input_0 = item_per_0[torch.arange(bs), p_0_target]
        input_1 = item_per_1[torch.arange(bs), p_1_target] # bs dim
        input_emb = torch.cat([input_0, input_1, input_01, input_10], dim=-1)
        # policy layer
        output_value = self.policy_layer(input_emb) # bs n_item

        return output_value

    def predict(self, data):
        self.eval()
        # print(data)
        with torch.no_grad():
            p_0_rec, p_1_rec, p_0_target, p_1_target = \
                                    data['p_0_rec'], data['p_1_rec'],data['p_0_t'],data['p_1_t']
            
            p_0_rec, p_1_rec, p_0_target, p_1_target = \
                                    torch.LongTensor(p_0_rec).to(self.device), \
                                    torch.LongTensor(p_1_rec).to(self.device), torch.LongTensor(p_0_target).to(self.device), \
                                    torch.LongTensor(p_1_target).to(self.device)
            policy = self.forward(p_0_rec, p_1_rec, p_0_target, p_1_target)
            policy = tensor_to_numpy(policy)
        return policy
    
    def optimize_model(self, data, lr):
        self.train()
        p_0_rec, p_1_rec, p_0_target, p_1_target, target, goal = \
                                data['p_0_rec'], data['p_1_rec'],data['p_0_t'],data['p_1_t'], data['iid'], data['goal']
        
        p_0_rec, p_1_rec, p_0_target, p_1_target, target, goal = \
                                torch.LongTensor(p_0_rec).to(self.device), \
                                torch.LongTensor(p_1_rec).to(self.device), torch.LongTensor(p_0_target).to(self.device), \
                                torch.LongTensor(p_1_target).to(self.device), torch.LongTensor(target).to(self.device), \
                                torch.FloatTensor(goal).to(self.device)
        op = optim.Adam(self.parameters(), lr=lr)
        op.zero_grad()
        policy = self.forward(p_0_rec, p_1_rec, p_0_target, p_1_target)
        pre_value = policy[torch.arange(len(p_0_rec)), target]
        loss_func = torch.nn.MSELoss(reduction='mean')  
        loss = loss_func(pre_value, goal)
        loss.backward()
        op.step()
        # loss = tensor_to_numpy(loss)
        return tensor_to_numpy(loss)

    @classmethod
    def create_model(cls, config):
        logger.info("CREATE MODEL", config.model)
        model = cls(config.item_num, config.latent_factor, config.num_blocks,
                 True, config.dropout_rate, policy_fc_dim=512, n_heads=config.num_heads, d_ff=2048,  l2=1e-5, separate_qa=None, pad=0)
        return model.to(torch.device('cuda'))


def mask(src, s_len):
    if type(src) == torch.Tensor:
        mask = torch.zeros_like(src)
        for i in range(len(src)):
            mask[i, :s_len[i]] = 1

    return mask


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    output = torch.matmul(p_attn, value)
    return scores, output, p_attn





class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch. 
        _, x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class MultiHeadedAttention_con(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention_con, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 5)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value1, value2, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key))]

        value1 = value1.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value2 = value2.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        # 2) Apply attention on all the projected vectors in batch. 
        
        # print(query.shape, key.shape, value1.shape)
        scores, x1, attn_s = attention(query, key, value1, mask=mask, 
                                 dropout=self.dropout)

        x2 = torch.matmul(attn_s.transpose(-1, -2), value2)
        
        # 3) "Concat" using a view and apply a final linear. 
        x1 = x1.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        
        x2 = x2.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x1), self.linears[-1](x2), 

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
