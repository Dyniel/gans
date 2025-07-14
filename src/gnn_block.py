import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

class GraphBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphBlock, self).__init__()
        self.conv = SAGEConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)
