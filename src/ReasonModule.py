import torch
from torch import nn
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


class ReasonModule(nn.Module):
    def __init__(self, in_channels=200, processing_steps=0, num_layers=1):
        """
        Reasoning Module
        """
        super(ReasonModule, self).__init__()

        self.in_channels = in_channels
        self.out_channels = 2 * in_channels
        self.processing_steps = processing_steps
        self.num_layers = num_layers
        if processing_steps > 0:
            self.lstm = nn.LSTM(self.out_channels, self.in_channels, num_layers)  # 400,200,1
            self.lstm.reset_parameters()

    def forward(self, x, batch, q_star):
        if self.processing_steps <= 0: return q_star

        batch_size = batch.max().item() + 1
        h = (x.new_zeros((self.num_layers, batch_size, self.in_channels)),  #生成一个元组h，里面有两个tensor元素，分别是(1,batch_size,200)形状的
             x.new_zeros((self.num_layers, batch_size, self.in_channels)))
        for i in range(self.processing_steps):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.in_channels)
            e = (x * q[batch]).sum(dim=-1, keepdim=True)                 #formulate 6 in paper
            a = softmax(e, batch, num_nodes=batch_size)                  #formulate 7 in paper
            r = scatter_add(a * x, batch, dim=0, dim_size=batch_size)    #formulate 8 in paper
            q_star = torch.cat([q, r], dim=-1)                           #new q
        return q_star

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)