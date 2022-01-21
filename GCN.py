import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv, FiLMConv, GATConv, SAGPooling
from torch_geometric.nn.glob import global_max_pool, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(num_node_features, 32, num_relations)
        #self.conv1 = FiLMConv(num_node_features, 32, num_relations) #FiLMConv looks like a better choice than RGCNConv but it bugs on GPU
        self.conv2 = RGCNConv(32, 16, num_relations)
        #self.conv2 = FiLMConv(32, 16, num_relations)
        self.dense = nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, (data.edge_attr).flatten()

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, 0.5)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch) #note: looks like mean pooling is working better than max pooling
        x = self.dense(x)

        return x
        #return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GATConv(num_node_features, 32, heads=4, add_self_loops=False, edge_dim=1)
        self.dense1 = nn.Linear(32*4, 32)
        self.dense2 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, (data.edge_attr).flatten()

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        #print(data.num_nodes, x.shape) #num nodes, (h_size * N_heads)
        x = F.dropout(x, 0.5)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = global_max_pool(x, data.batch)
        x = self.dense2(x)
        #print(x)

        return x
