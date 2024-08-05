"""recommender: A Flower / PyTorch app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from recommender.task import (
    GCN,
    DEVICE,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, val_edge_attr):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.val_edge_attr = val_edge_attr

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            self.val_edge_attr,
            DEVICE,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.val_edge_attr)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, testdataobject, valloader, val_edge_attributes = load_data(partition_id, num_partitions)
    net = GCN(in_channels=5227, hidden_channels=16, out_channels=1).to(DEVICE)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, val_edge_attributes).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
