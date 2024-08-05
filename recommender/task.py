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


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Need feature projection or padding
# Each client has a different number of node_features
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
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
    
data = None  # Cache FederatedDataset
partitioner = None # Cache partitioner

def load_data(partition_id: int, num_partitions: int):
    """Load partition Amazon Reviews data."""
    global data
    global partitioner
    if data is None:
        data = pd.read_csv("/home/yang/Documents/GitHub/recommender/data/ratings_Electronics (1).csv")
        data = data.head(5000) # crop data for dev
        data.rename(columns = {'AKM1MP6P0OYPR':'userId', '0132793040':'productId', '5.0':'Rating', '1365811200':'timestamp'}, inplace = True)
        # Clean the data
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)

        # Encode user IDs and item IDs
        user_encoder = LabelEncoder()
        item_encoder = LabelEncoder()

        data['userId'] = user_encoder.fit_transform(data['userId'])
        data['productId'] = item_encoder.fit_transform(data['productId'])
        dataset = Dataset.from_pandas(data)

        partitioner = IidPartitioner(num_partitions)
        partitioner.dataset = dataset

    partition = partitioner.load_partition(partition_id)

    # Print the size of the partition
    # print(f"Partition {partition_id} size: {len(partition)}")

    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    # train data processing for GCN
    train_geometric_data_object, _ = get_edges(partition_train_test["train"])
    train_feature_engineered_data, train_node_features = get_nodes(partition, train_geometric_data_object)
    trainloader = DataLoader([train_feature_engineered_data], batch_size=1, shuffle=True)
    # test data processing for GCN
    test_geometric_data_object, test_edge_attributes = get_edges(partition_train_test["test"])
    test_feature_engineered_data, test_node_features = get_nodes(partition, test_geometric_data_object)
    # testloader = DataLoader(partition_train_test["test"], batch_size=32) # not sure
    return trainloader, test_geometric_data_object, test_edge_attributes
# , test_feature_engineered_data, test_edge_attributes
    
def get_edges(train_dataset):
    train_data_df = train_dataset.to_pandas()
    # Create edge index from user-item interactions
    # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. 
    # Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. 
    # (Triggered internally at ../torch/csrc/utils/tensor_new.cpp:275.)
    edge_index = torch.tensor([train_data_df['userId'].values, train_data_df['productId'].values], dtype=torch.long)

    # # Convert the columns to numpy arrays
    # user_ids = train_data_df['userId'].values
    # product_ids = train_data_df['productId'].values

    # # Stack the arrays to create a 2D numpy array
    # edge_index_array = np.vstack((user_ids, product_ids))

    # # Convert the numpy array to a tensor
    # edge_index = torch.tensor(edge_index_array, dtype=torch.long)

    # Create edge attributes (ratings)
    edge_attr = torch.tensor(train_data_df['Rating'].values, dtype=torch.float)

    # Create the PyTorch Geometric data object
    data = Data(edge_index=edge_index, edge_attr=edge_attr)

    # print(data)

    return data, edge_attr

def get_nodes(partition, data, target_size=606):
    dataset_df = partition.to_pandas()
    num_users = dataset_df['userId'].nunique()
    # print(f"Number of users: {num_users}")
    num_items = dataset_df['productId'].nunique()
    # print(f"Number of items: {num_items}")
    num_nodes = num_users + num_items

    # Create node features
    node_features = torch.tensor(np.eye(num_nodes), dtype=torch.float)
    
    # Pad node features to the target size
    if num_nodes < target_size:
        padding = np.zeros((target_size - num_nodes, num_nodes), dtype=np.float32)
        node_features = np.vstack((node_features, padding))
    elif num_nodes > target_size:
        raise ValueError(f"Number of nodes ({num_nodes}) exceeds the target size ({target_size}).")

    node_features = torch.tensor(node_features, dtype=torch.float)


    # Add node features to the PyTorch Geometric data object
    data.x = node_features

    print(data)

    return data, node_features

def train(net, trainloader, valloader, epochs, device, val_edge_attr):
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

    val_rmse, val_mae = test(net, valloader, val_edge_attr)

    results = {
        "train_loss": loss.item(),
        "train_accuracy": 0.0,
        "val_rmse": val_rmse,
        "val_mae": val_mae,
    }
    return results

def test(net, testloader, test_edge_attr):
    """Validate the model on the test set."""
    net.eval()
    with torch.no_grad():
        out = net(testloader)
        test_rmse = mean_squared_error(test_edge_attr.numpy(), out.numpy(), squared=False)
        test_mae = mean_absolute_error(test_edge_attr.numpy(), out.numpy())
    return test_rmse, test_mae

def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# class TestLoadData(unittest.TestCase):
#     def test_load_data(self):
#         for i in range(10):
#             try:
#                 train_data, test_data, test_edge = load_data(i, 10)
#                 print(f"Partition {i} loaded successfully with node features size.")
#             except Exception as e:
#                 print(f"Error loading partition {i}: {e}")

# if __name__ == "__main__":
#     unittest.main()
