
import torch
import torch_geometric
import torch.nn.functional as F


class GCNModel(torch.nn.Module):
    """
    Class to define the graphical convolutional network model

    Attributes

    """

    def __init__(self, y_ind, n_nodes, edge_index, edge_weights, dim_input, dim_output, nhid, kernel_size=2, dropout=0.):
        super().__init__()

        n_y = len(y_ind)

        # TODO: you do not need to initialize self.y_ind, try without it
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.kernel_size = kernel_size
        if self.kernel_size > 0:
            self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(1, self.kernel_size))
            self.gcn1 = torch_geometric.nn.GCNConv(dim_input - self.kernel_size + 1, nhid[0])
        else:
            self.gcn1 = torch_geometric.nn.GCNConv(dim_input, nhid[0])
        self.gcn2 = torch_geometric.nn.GCNConv(nhid[0], nhid[1])
        # self.gcn3 = torch_geometric.nn.GCNConv(nhid[1], nhid[2])
        # self.linear = torch.nn.Linear(nhid[2] * n_y, dim_output * n_y)
        self.linear = torch.nn.Linear(nhid[1] * n_nodes, dim_output * n_y)
        self.dropout = dropout
        self.output_shape = [n_y, dim_output]

    def forward(self, x):

        # TODO: get rid of Conv2d and dropout if useless
        # Conv2d
        if self.kernel_size > 0:
            x = torch.unsqueeze(x, dim=0)
            x = self.conv1(x)
            x = F.relu(x)
            x = x[0, :, :]

        # GCNConv1
        x = self.gcn1(x, self.edge_index, self.edge_weights)
        x = F.relu(x)
        # print(x.shape)

        # Dropout
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        # GCNConv2
        x = self.gcn2(x, self.edge_index, self.edge_weights)
        x = F.relu(x)
        # print(x.shape)

        # # GCNConv3
        # x = self.gcn3(x, self.edge_index, self.edge_weights)
        # x = F.relu(x)
        # # print(x.shape)

        # Linear
        x = self.linear(x.view(-1))
        # print(x.shape)

        return x.reshape(self.output_shape)