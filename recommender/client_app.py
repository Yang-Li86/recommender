"""recommender-2: A Flower / PyTorch app."""

import sys
sys.path.append('/home/yang/Documents/GitHub/recommender')

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from logging import INFO, DEBUG
from flwr.common.logger import log, configure

from datetime import datetime

from recommender.task import (
    Net,
    DEVICE,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)

configure(identifier="client0", filename="client0_log.txt")


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs

    def fit(self, parameters, config):
        log(INFO, f"Got parameters at time {datetime.now()}")
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            DEVICE,
        )
        log(INFO, f"Sent parameters at time {datetime.now()}")
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net(in_channels=5227, hidden_channels=16, out_channels=1).to(DEVICE)
    # partition_id = context.node_config["partition-id"]
    partition_id = context.node_config.get("partition-id", 0)
    # num_partitions = context.node_config["num-partitions"]
    num_partitions = context.node_config.get("num-partitions", 3)
    trainloader, valloader = load_data()
    # local_epochs = context.run_config["local-epochs"]
    local_epochs = context.run_config.get("local-epochs", 1)

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
