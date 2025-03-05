import copy
import math
from copy import deepcopy

import torch
from torch import nn


if torch.cuda.is_available():
    device = torch.device('cuda')

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_embed, h):
        super(MultiHeadAttentionLayer, self).__init__()
        self.d_embed = d_embed
        self.h = h
        self.d_model = d_embed * h
        self.q_fc = nn.Linear(d_embed, self.d_model).to(device)  # (d_embed, d_model)
        self.k_fc = nn.Linear(self.d_embed, self.d_model).to(device)  # (d_embed, d_model)
        self.v_fc = nn.Linear(self.d_embed, self.d_model).to(device)  # (d_embed, d_model)
        self.out_fc = nn.Linear(self.d_model, d_embed).to(device)  # (d_model, d_embed)

    def forward(self, *args, query, key, value, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        n_batch = query.size(0)
        softmax = nn.Softmax(dim=-1)

        def transform(x, fc):  # (n_batch, seq_len, d_embed)
            out = fc(x)  # (n_batch, seq_len, d_model)
            out = out.contiguous().view(n_batch, -1, self.h, self.d_embed)  # (n_batch, seq_len, h, d_k)
            out = out.contiguous().transpose(1, 2)  # (n_batch, h, seq_len, d_k)
            return out

        def calculate_attention(query, key, value, mask, n_xor=0, mask_xor=None):
            # query, key, value: (n_batch, h, seq_len, d_k)
            # mask: (n_batch, 1, seq_len, seq_len)

            # n_xor==0: query and key have same number of zero
            # n_xor==1: key have more zero than query -> query win
            # n_xor==2: query have more zero than key -> key win

            ### mask_ 추가하기

            mask = mask.unsqueeze(dim=1).repeat(1, self.h, 1, 1)
            d_k = key.shape[-1]
            attention_score = torch.matmul(query, key.contiguous().transpose(-2,
                                                                             -1))  # Q x K^T, (n_batch, h, seq_len, seq_len)
            attention_score = attention_score / math.sqrt(d_k)

            if mask is not None:
                attention_score = attention_score.masked_fill(mask == 0, -1e9)
            attention_prob = softmax(attention_score)  # (n_batch, h, seq_len, seq_len)

            if mask is not None:
                attention_prob = attention_prob.masked_fill(mask == 0, 0)  # set zero to original zero index

            out = torch.matmul(attention_prob, value)  # (n_batch, h, seq_len, d_k)

            #             if n_xor==1:
            #                 print('n_xor=1')
            #                 for i in range(self.h):
            #                     out[:,j,:,:][mask_xor] = query[:,j,:,:][mask_xor].contiguous()
            #             elif n_xor==2:
            #                 print('n_xor=2')
            #                 for i in range(self.h):
            #                     out[:,j,:,:][mask_xor] = key[:,j,:,:][mask_xor].contiguous()

            return out

        temp = torch.ne(torch.sum(torch.ne(query, 0), dim=-1), 0)
        mask_q = torch.stack([deepcopy(temp) for i in range(query.shape[-1])], axis=-1)
        temp = torch.ne(torch.sum(torch.ne(key, 0), dim=-1), 0)
        mask_k = torch.stack([deepcopy(temp) for i in range(key.shape[-1])], axis=-1)

        #         mask_k = torch.ne(torch.sum(torch.ne(key,0),dim=-1))
        #         mask_q = torch.zeros_like(query) # recognize zero vector
        #         mask_q[~torch.logical_not(query)] = 1
        #         mask_k = torch.zeros_like(key) # recognize zero vector
        #         mask_k[~torch.logical_not(key)] = 1

        mask_xor = torch.logical_xor(mask_q, mask_k)
        if (mask_xor.sum() > 0):
            n_q = mask_q.sum()
            n_k = mask_k.sum()
            if n_q > n_k:
                n_xor = 1
            elif n_q < n_k:
                n_xor = 2
            else:
                n_xor = 0

        n_xor = 0
        query = transform(query.to(device), self.q_fc)  # (n_batch, h, seq_len, d_k)
        key = transform(key.to(device), self.k_fc)  # (n_batch, h, seq_len, d_k)
        value = transform(value.to(device), self.v_fc)  # (n_batch, h, seq_len, d_k)

        out = calculate_attention(query, key, value, mask, n_xor, mask_xor)  # (n_batch, h, seq_len, d_k)
        out = out.transpose(1, 2)  # (n_batch, seq_len, h, d_k)
        out = out.contiguous().view(n_batch, -1, self.d_model)  # (n_batch, seq_len, d_model)
        out = self.out_fc(out)  # (n_batch, seq_len, d_embed)

        return out


# -

class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, d_k, d_ff=None):
        super(PositionWiseFeedForwardLayer, self).__init__()
        if d_ff is None: d_ff = d_k
        self.fc1 = nn.Linear(d_k, d_ff).to(device)  # (d_embed, d_ff)
        self.relu = nn.ReLU().to(device)
        self.fc2 = nn.Linear(d_ff, d_k).to(device)  # (d_ff, d_embed)

    def forward(self, x):
        out = x
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class ResidualConnectionLayer(nn.Module):

    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()

    def forward(self, x):
        lin = nn.Linear(x.shape[-1], x.shape[-1]).to(device)
        return lin(x) + x


# ### 3-2) Encoder

class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(2)]
        self.do = nn.Dropout(0.2).to(device)
        self.layer_norm = nn.LayerNorm(D_hidden).to(device)

    def forward(self, trg, src, src_mask=None):
        out = self.layer_norm(trg)
        src = self.layer_norm(src)
        out = self.self_attention(query=out, key=src, value=src, mask=src_mask)
        out = self.residuals[0](self.layer_norm(self.do(out)))
        out = self.position_ff(self.layer_norm(self.do(out)))
        out = self.residuals[1](self.layer_norm(self.do(out)))
        out = self.do(out)

        return out


class Encoder(nn.Module):

    def __init__(self, encoder_layer, n_layer):  # n_layer: Encoder Layer의 개수
        super(Encoder, self).__init__()
        self.layers = []
        for i in range(n_layer):
            self.layers.append(copy.deepcopy(encoder_layer))

    def forward(self, trg, src, src_mask):
        out = trg
        for layer in self.layers:
            out = layer(out, src, src_mask)
        return out


class Transformer(nn.Module):
    def __init__(self, encoder):
        super(Transformer, self).__init__()
        self.encoder = encoder

    def forward(self, query, key, bMask=False):
        def make_pad_mask(query, key, pad_idx=0):
            query = query[:, :, 0]
            key = key[:, :, 0]
            # query: (n_batch, query_seq_len)
            # key: (n_batch, key_seq_len)

            mask_q = torch.ne(query, 0).unsqueeze(dim=1).repeat(1, query.shape[1], 1)
            mask_k = torch.ne(key, 0).unsqueeze(dim=1).repeat(1, query.shape[1], 1)
            mask = (mask_q * mask_k).transpose(1, 2).contiguous()

            return mask

        def make_src_mask(trg, src):
            pad_mask = make_pad_mask(trg, src)
            return pad_mask

        src_mask = make_src_mask(query, key)

        if bMask:
            out = self.encoder(query, key, src_mask)
        else:
            out = self.encoder(query, key, None)

        return out


class CrossModalTransformer(nn.Module):

    def __init__(self, d_embed=64, h=8, n_layer=5):
        super(CrossModalTransformer, self).__init__()
        # declare
        self.attention = MultiHeadAttentionLayer(d_embed=64, h=8)
        self.positionff = PositionWiseFeedForwardLayer(d_k=64, d_ff=64)
        self.residual = ResidualConnectionLayer()
        self.encoder_layer = EncoderBlock(self.attention, self.positionff)
        self.encoder = Encoder(self.encoder_layer, n_layer)
        self.model = Transformer(self.encoder)

    def forward(self, query, key, bMask):
        out = self.model(query, key, bMask)
        return out


class Attention(nn.Module):
    def __init__(self, d_embed):
        super(Attention, self).__init__()
        self.d_embed = d_embed
        self.q_fc = nn.Linear(self.d_embed, self.d_embed).to(device)
        self.k_fc = nn.Linear(self.d_embed, self.d_embed).to(device)
        self.v_fc = nn.Linear(self.d_embed, self.d_embed).to(device)
        self.out_fc = nn.Linear(self.d_embed, self.d_embed).to(device)

    def forward(self, src, mask=None):
        # query, key, value: (n_batch, seq_len, d_embed)
        # mask: (n_batch, seq_len, seq_len)
        # return value: (n_batch, h, seq_len, d_k)
        softmax = nn.Softmax(dim=-1).to(device)
        query = src.clone().to(device)
        key = src.clone().to(device)
        value = src.clone().to(device)
        n_batch = query.size(0)

        def make_pad_mask(query, key, pad_idx=0):
            query = query[:, :, 0]
            key = key[:, :, 0]
            # query: (n_batch, query_seq_len)
            # key: (n_batch, key_seq_len)

            mask_q = torch.ne(query, 0).unsqueeze(dim=1).repeat(1, query.shape[1], 1)
            mask_k = torch.ne(key, 0).unsqueeze(dim=1).repeat(1, query.shape[1], 1)
            mask = (mask_q * mask_k).transpose(1, 2).contiguous()
            mask.requires_grad = False

            return mask

        def make_src_mask(trg, src):
            pad_mask = make_pad_mask(trg, src)
            return pad_mask

        def transform(x, fc):  # (n_batch, seq_len, d_embed)
            out = fc(x)  # (n_batch, seq_len, d_model)
            return out

        def calculate_attention(query, key, value, mask=None):
            d_k = key.shape[-1]
            attention_score = torch.matmul(query, key.transpose(-2, -1))  # Q x K^T, (n_batch, seq_len, seq_len)
            attention_score = attention_score / math.sqrt(d_k)
            if mask is not None:
                attention_score = attention_score.masked_fill(mask == 0, -1e9)
            attention_prob = softmax(attention_score)  # (n_batch, seq_len, seq_len)
            if mask is not None:
                attention_prob = attention_prob.masked_fill(mask == 0, 0)  # set zero to original zero index
            out = torch.matmul(attention_prob, value)  # (n_batch, seq_len, d_k)
            return out

        query = transform(query, self.q_fc)  # (n_batch, seq_len, d_k)
        key = transform(key, self.k_fc)  # (n_batch, seq_len, d_k)
        value = transform(value, self.v_fc)  # (n_batch, seq_len, d_k)

        src_mask = make_src_mask(query, key)
        out = calculate_attention(query, key, value, src_mask)  # (n_batch, seq_len, d_k)
        out = self.out_fc(out)  # (n_batch, seq_len, d_embed)

        return out
