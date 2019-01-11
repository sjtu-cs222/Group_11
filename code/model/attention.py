import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


import json
with open("config.json", 'r') as load_f:
    config = json.load(load_f)




class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1, gpu = True):
        super().__init__()
        self.gpu = gpu
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, gpu = True):
        super().__init__()

        self.gpu = gpu
        self.n_head = config['n_head']
        self.d_k = d_k
        self.d_v = d_v


        self.w_qs = nn.Linear(d_model, int(n_head * d_k))
        self.w_ks = nn.Linear(d_model, int(n_head * d_k))
        self.w_vs = nn.Linear(d_model, int(n_head * d_v))
        nn.init.normal(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        #self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(int(n_head * d_v), d_model)
        nn.init.xavier_normal(self.fc.weight)

        self.dropout = nn.Dropout(dropout)
        # if self.gpu:
        #     self.w_qs = self.w_qs.cuda()
        #     self.w_ks = self.w_ks.cuda()
        #     self.w_vs = self.w_vs.cuda()


    def forward(self, q, k, v):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head


        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()


        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn = self.attention(q, k, v)
        # if self.gpu:
        #     output = output.cuda()

        output = output.view(n_head, sz_b, len_q, d_v)
        #output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, int(len_q*n_head*d_k))

        if config['combination'] == 0:
            fc = nn.Linear(len_q*d_k*n_head, d_k*n_head)
        elif config['combination'] == 1:
            fc = nn.Linear(len_q * d_k * n_head, 100)
        if self.gpu:
            fc = fc.cuda()
        output = self.dropout(fc(output))

        #output = self.layer_norm(output + residual)


        return output, attn

if __name__ == '__main__':
    # gpu = torch.cuda.is_available(5, 50, 10, 10, gpu=gpu)
    # ATT = MultiHeadAttention(5, 50, 10, 10, gpu=gpu)
    # data = torch.randn(32, 10, 6)
    # att = ATT(3, 128, 10)
    # if gpu:
    #     data =
    #     att =
    #
    # output = lstm(data)
    # print(output, output.shape)
    test = torch.randn(32, 10, 6)
