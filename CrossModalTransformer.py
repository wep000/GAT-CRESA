import copy
import math

import torch
from torch import nn
import torch.nn.functional as F

if torch.cuda.is_available():
    device = torch.cuda.is_available()

class MultiHeadAttentionLayer(nn.Module):
    """
    多头注意力计算
    """

    def __init__(self, head, d_model, dropout=0.1):
        """
        :param head: 头数
        :param d_model: 词向量的维度，必须是head的整数倍
        :param dropout: drop比率
        """
        super(MultiHeadAttentionLayer, self).__init__()
        assert (d_model % head == 0)  # 确保词向量维度是头数的整数倍
        self.d_k = d_model // head  # 被拆分为多头后的某一头词向量的维度
        self.head = head
        self.d_model = d_model

        """
        由于多头注意力机制是针对多组Q、K、V，因此有了下面这四行代码，具体作用是，
        针对未来每一次输入的Q、K、V，都给予参数进行构建
        其中linear_out是针对多头汇总时给予的参数
        """
        self.linear_query = nn.Linear(d_model, d_model)  # 进行一个普通的全连接层变化，但不修改维度
        self.linear_key = nn.Linear(d_model, d_model)
        self.linear_value = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_softmax = None  # attn_softmax是能量分数, 即句子中某一个词与所有词的相关性分数， softmax(QK^T)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            """
            多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
            再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第二维（head维）添加一维，与后面的self_attention计算维度一样
            具体点将，就是：
            因为mask的作用是未来传入self_attention这个函数的时候，作为masked_fill需要mask哪些信息的依据
            针对多head的数据，Q、K、V的形状维度中，只有head是通过view计算出来的，是多余的，为了保证mask和
            view变换之后的Q、K、V的形状一直，mask就得在head这个维度添加一个维度出来，进而做到对正确信息的mask
            """
            mask = mask.unsqueeze(1)

        n_batch = query.size(0)  # batch_size大小，假设query的维度是：[10, 32, 512]，其中10是batch_size的大小

        """
        下列三行代码都在做类似的事情，对Q、K、V三个矩阵做处理
        其中view函数是对Linear层的输出做一个形状的重构，其中-1是自适应（自主计算）
        从这种重构中，可以看出，虽然增加了头数，但是数据的总维度是没有变化的，也就是说多头是对数据内部进行了一次拆分
        transopose(1,2)是对前形状的两个维度(索引从0开始)做一个交换，例如(2,3,4,5)会变成(2,4,3,5)
        因此通过transpose可以让view的第二维度参数变成n_head
        假设Linear成的输出维度是：[10, 32, 512]，其中10是batch_size的大小
        注：这里解释了为什么d_model // head == d_k，如若不是，则view函数做形状重构的时候会出现异常
        """
        query = self.linear_query(query).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 32, 64]，head=8
        key = self.linear_key(key).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]
        value = self.linear_value(value).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)  # [b, 8, 28, 64]

        # x是通过自注意力机制计算出来的值， self.attn_softmax是相似概率分布
        x, self.attn_softmax = self_attention(query, key, value, dropout=self.dropout, mask=mask)

        """
        下面的代码是汇总各个头的信息，拼接后形成一个新的x
        其中self.head * self.d_k，可以看出x的形状是按照head数拼接成了一个大矩阵，然后输入到linear_out层添加参数
        contiguous()是重新开辟一块内存后存储x，然后才可以使用.view方法，否则直接使用.view方法会报错
        """
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
        return self.linear_out(x)

def self_attention(query, key, value, dropout=None, mask=None):
    """
    自注意力计算
    :param query: Q
    :param key: K
    :param value: V
    :param dropout: drop比率
    :param mask: 是否mask
    :return: 经自注意力机制计算后的值
    """
    d_k = query.size(-1)  # 防止softmax未来求梯度消失时的d_k
    # Q,K相似度计算公式：\frac{Q^TK}{\sqrt{d_k}}
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # Q,K相似度计算
    # 判断是否要mask，注：mask的操作在QK之后，softmax之前
    if mask is not None:
        """
        scores.masked_fill默认是按照传入的mask中为1的元素所在的索引，
        在scores中相同的的索引处替换为value，替换值为-1e9，即-(10^9)
        """
        # mask.cuda()
        # 进行mask操作，由于参数mask==0，因此替换上述mask中为0的元素所在的索引
        scores = scores.masked_fill(mask == 0, -1e9)

    self_attn_softmax = F.softmax(scores, dim=-1)  # 进行softmax
    # 判断是否要对相似概率分布进行dropout操作
    if dropout is not None:
        self_attn_softmax = dropout(self_attn_softmax)

    # 注意：返回经自注意力计算后的值，以及进行softmax后的相似度（即相似概率分布）
    return torch.matmul(self_attn_softmax, value), self_attn_softmax



class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, d_k, d_ff=None):
        super(PositionWiseFeedForwardLayer, self).__init__()
        if d_ff is None: d_ff = d_k
        self.fc1 = nn.Linear(d_k,d_ff).to(device)   # (d_embed, d_ff)
        self.relu = nn.ReLU().to(device)
        self.fc2 = nn.Linear(d_ff,d_k).to(device) # (d_ff, d_embed)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class ResidualConnectionLayer(nn.Module):

    def __init__(self):
        super(ResidualConnectionLayer, self).__init__()

    def forward(self, x):
        lin = nn.Linear(x.shape[-1],x.shape[-1]).to(device)
        return lin(x) + x


class EncoderBlock(nn.Module):

    def __init__(self, self_attention, position_ff):
        super(EncoderBlock, self).__init__()
        self.self_attention = self_attention  #多头注意力层
        self.position_ff = position_ff
        self.residuals = [ResidualConnectionLayer() for _ in range(2)]
        self.do = nn.Dropout(0.2).to(device)
        self.layer_norm = nn.LayerNorm(hidden_dim).to(device)

    def forward(self, trg, src, src_mask=None):
        '''trg为query,src为key'''
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

    def forward(self, query, key, bMask=None):
        def make_pad_mask(query, key, pad_idx=0):
            query = query[:, :, 0]
            key = key[:, :, 0]
            #这里去掉第三个维度，只需要seq_len来做掩码
            # query: (n_batch, query_seq_len)
            # key: (n_batch, key_seq_len)

            mask_q = torch.ne(query, 0).unsqueeze(dim=1).repeat(1, query.shape[1], 1)
            '''#torch.ne(query, 0)检查query中非0元素返回和query形状相同的boolean Tensor,若为0则为FALSE，非0则为true，
            repeat(1, query.shape[1], 1)表示第一个维度重复一次，第二个维度重复query.shape[1]次，第三个维度也重复一次，最后得到的
            mask_q是(n_batch,query_seq_len,query_seq_len)形状，元素全为boolean型的tensor'''
            mask_k = torch.ne(key, 0).unsqueeze(dim=1).repeat(1, query.shape[1], 1)
            mask = (mask_q * mask_k).transpose(1, 2).contiguous()

            return mask

        def make_src_mask(trg, src):
            pad_mask = make_pad_mask(trg, src)
            return pad_mask

        src_mask = make_src_mask(query, key)

        out = self.encoder(query, key, bMask)

        return out

class CrossModalTransformer(nn.Module):

    def __init__(self, d_embed=64, h=8, n_layer=5):
        super(CrossModalTransformer, self).__init__()
    # declare
        self.attention = MultiHeadAttentionLayer(d_embed = 64, h = 8)
        self.positionff = PositionWiseFeedForwardLayer(d_k=64, d_ff=64)
        self.residual = ResidualConnectionLayer()
        self.encoder_layer = EncoderBlock(self.attention, self.positionff)
        self.encoder = Encoder(self.encoder_layer,n_layer)
        self.model = Transformer(self.encoder)

    def forward(self, query, key, bMask):
        out = self.model(query, key, bMask)
        return out