"""recommender-2: A Flower / PyTorch app."""
# flower-superlink --insecure
# flower-server-app server_app:app --superlink 127.0.0.1:9091
# flower-client-app client_app:app --insecure --superlink 127.0.0.1:9092
# flower-client-app client_app1:app --insecure --superlink 127.0.0.1:9092

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import unittest
# https://flower.ai/docs/datasets/how-to-use-with-local-data.html
from datasets import Dataset
from flwr_datasets.partitioner import IidPartitioner
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch_geometric.data import Data
from collections import OrderedDict
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
datapath = "/home/yang/Documents/GitHub/recommender/data/ratings_Electronics (1).csv"


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels * 2, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Apply the final linear layer on the concatenated edge features
        edge_pred = self.fc(torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=1))
        return edge_pred.squeeze()

def load_data(num_partitions: int, partition_id: int, split_ratio=0.2):
    url = datapath
    df = pd.read_csv(url)
    df.rename(columns={'AKM1MP6P0OYPR': 'userId', '0132793040': 'productId', '5.0': 'Rating', '1365811200': 'timestamp'}, inplace=True)
    df = df.head(5000)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['userId'] = user_encoder.fit_transform(df['userId'])
    df['productId'] = item_encoder.fit_transform(df['productId'])

    train_df, test_df = train_test_split(df, test_size=split_ratio, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=split_ratio, random_state=42)

    # training edge matrix using only the training data
    train_edge_index = torch.tensor([train_df['userId'].values, train_df['productId'].values], dtype=torch.long)
    train_edge_attr = torch.tensor(train_df['Rating'].values, dtype=torch.float)
    # val edge matrix using only the val data
    val_edge_index = torch.tensor([val_df['userId'].values, val_df['productId'].values], dtype=torch.long)
    val_edge_attr = torch.tensor(val_df['Rating'].values, dtype=torch.float)
    # node matrix using all the data
    num_users = df['userId'].nunique()
    num_items = df['productId'].nunique()
    num_nodes = num_users + num_items
    node_features = torch.eye(num_nodes)
    train_data = Data(edge_index=train_edge_index, edge_attr=train_edge_attr, x=node_features)
    val_data = Data(edge_index=val_edge_index, edge_attr=val_edge_attr, x=node_features)

    # Randomly partition nodes into clients
    # node_indices = np.random.permutation(num_nodes)
    # client_partitions = np.array_split(node_indices, num_partitions)

    # client_data = []

    # for client_nodes in client_partitions:
    #     # Create a mask for the nodes that belong to this client
    #     mask = torch.zeros(num_nodes, dtype=torch.bool)
    #     mask[client_nodes] = True

    #     # Filter the edges to include only those where both nodes are in the client's partition
    #     edge_mask = mask[train_data.edge_index[0]] & mask[train_data.edge_index[1]]

    #     # Create the subgraph
    #     subgraph = Data(x=train_data.x[mask],
    #                     edge_index=train_data.edge_index[:, edge_mask],
    #                     edge_attr=train_data.edge_attr[edge_mask]
    #     )
        
    #     client_data.append(subgraph)

    # print(client_data)

    # partitioner = IidPartitioner(num_partitions)
    # partitioner.dataset = data
    # partition = partitioner.load_partition(partition_id)
    
    trainloader = DataLoader([train_data], batch_size=1, shuffle=True)
    valloader = DataLoader([val_data], batch_size=1, shuffle=False)
    
    return trainloader, valloader

a, b = load_data(1, 3)


def train(net, trainloader, valloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            optimizer.zero_grad()
            out = net(batch)
            loss = criterion(out, batch.edge_attr.view(-1, 1))
            loss.backward()
            optimizer.step()

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.MSELoss()
    correct, loss = 1, 0.0
    total_loss = 0.0
    with torch.no_grad():
            for batch in testloader:
                out = net(batch)
                loss = criterion(out, batch.edge_attr.view(-1, 1))
                total_loss += loss.item()
    accuracy = correct / len(testloader.dataset)
    return total_loss, accuracy


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
