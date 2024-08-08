"""recommender-2: A Flower / PyTorch app."""

import sys
sys.path.append('/home/yang/Documents/GitHub/recommender')

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg, FedAdam

from recommender.task import Net, get_weights


# Initialize model parameters
ndarrays = get_weights(Net(in_channels=5227, hidden_channels=16, out_channels=1))
parameters = ndarrays_to_parameters(ndarrays)

def server_fn(context: Context):
    # Read from config
    # num_rounds = context.run_config["num-server-rounds"]
    num_rounds = context.run_config.get("num-server-rounds", 10)

    # Define strategy
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=3,
        min_fit_clients=3,
        initial_parameters=parameters,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn)
