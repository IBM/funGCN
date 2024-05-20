
import torch
import torch_geometric
import torch.nn.functional as F


class GCNModel(torch.nn.Module):
    """
    Class to define the Graphical Convolutional Network (GCN) model.

    This class defines a GCN model that takes in node features and edge information as input,
    and outputs a predicted value for each node.

    Attributes:
        y_ind (list): List of indices of output variables
        n_nodes (int): Number of nodes in the graph
        edge_index (torch.tensor): Edge index tensor, shape (2, num_edges)
        edge_weights (torch.tensor): Edge weight tensor, shape (num_edges,)
        dim_input (int): Dimensionality of input node features
        dim_output (int): Dimensionality of output node features
        nhid (list): List of hidden dimensions for GCN layers
        kernel_size (int): Kernel size for Conv2d layer (default=0)
        dropout (float): Dropout rate (default=0.)

    """

    def __init__(self, y_ind, n_nodes, edge_index, edge_weights, dim_input, dim_output, nhid, kernel_size=0, dropout=0.):
        super().__init__()

        n_y = len(y_ind)

        # TODO: you do not need to initialize self.y_ind, try without it
        self.edge_index = edge_index
        self.edge_weights = edge_weights
        self.kernel_size = kernel_size
        if self.kernel_size > 0:
            # Conv2d layer to process node features
            self.conv1 = torch.nn.Conv2d(1, 1, kernel_size=(1, self.kernel_size))
            # GCN layer to process graph structure
            self.gcn1 = torch_geometric.nn.GCNConv(dim_input - self.kernel_size + 1, nhid[0])
        else:
            # GCN layer to process graph structure
            self.gcn1 = torch_geometric.nn.GCNConv(dim_input, nhid[0])
        # GCN layer to process graph structure
        self.gcn2 = torch_geometric.nn.GCNConv(nhid[0], nhid[1])
        # # GCN layer to process graph structure
        # self.gcn3 = torch_geometric.nn.GCNConv(nhid[1], nhid[2])
        # Linear layer to output predicted values
        self.linear = torch.nn.Linear(nhid[1] * n_nodes, dim_output * n_y)
        self.dropout = dropout
        self.output_shape = [n_y, dim_output]

    def forward(self, x):
        """
        Forward pass through the GCN model.

        Inputs:
            x (torch.tensor): Node feature tensor, shape (n_nodes, dim_input)

        Outputs:
            x (torch.tensor): Predicted value tensor, shape (n_y, dim_output)
        """

        # TODO: get rid of Conv2d and dropout if useless
        # Conv2d layer
        if self.kernel_size > 0:
            x = torch.unsqueeze(x, dim=0)
            x = self.conv1(x)
            x = F.relu(x)
            x = x[0, :, :]

        # GCNConv1 layer
        x = self.gcn1(x, self.edge_index, self.edge_weights)
        x = F.relu(x)

        # Dropout layer
        if self.dropout > 0:
            x = F.dropout(x, self.dropout, training=self.training)

        # GCNConv2 layer
        x = self.gcn2(x, self.edge_index, self.edge_weights)
        x = F.relu(x)

        # # GCNConv3 layer
        # x = self.gcn3(x, self.edge_index, self.edge_weights)
        # x = F.relu(x)

        # Linear layer
        x = self.linear(x.view(-1))
        return x.reshape(self.output_shape)