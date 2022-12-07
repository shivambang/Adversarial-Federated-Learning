import warnings
from collections import OrderedDict

import numpy as np
from FLCosine import FLCosine
from FLEuclid import FLEuclid
from FLMedian import FLMedian
from flwr.server.strategy import FedAvg
import flwr as fl
from flwr.common import Metrics
from model_cnn import LeNet
import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, ToTensor
from client import FlowerClient
from model_train import DEVICE, get_parameters, set_parameters, test
from typing import List, Tuple

import argparse

parser = argparse.ArgumentParser(description='Federated Learning')
parser.add_argument('-N', '--num_cli', type=int, default=2, help='num of clients')
parser.add_argument('-A', '--num_adv', type=int, default=0, help='num of adversaries')
parser.add_argument('-D', '--defense', default='none', choices=['none', 'median', 'euclid', 'cosine'], help='type of defense')

args = parser.parse_args()
assert args.num_adv <= args.num_cli
NUM_CLIENTS = args.num_cli

def load_datasets(num_clients, poison=False):
    transform = Compose(
      [ToTensor(), Normalize(mean=0.5, std=0.5)]
    )
    target_transform = (lambda x: (x+2)%10) if poison else None
    train_data = MNIST("./data", train=True, download=True, transform=transform, 
        target_transform=target_transform)
    test_data = MNIST("./data", train=False, download=True, transform=transform)

    partition_size = len(train_data) // num_clients
    lengths = [partition_size] * num_clients
    partitions = random_split(train_data, lengths, torch.Generator().manual_seed(42))

    trainloaders = []
    valloaders = []
    for data in partitions:
        len_val = len(data) // 10
        len_train = len(data) - len_val
        lengths = [len_train, len_val]
        data_train, data_val = random_split(data, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(data_train, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(data_val, batch_size=32))
    testloader = DataLoader(test_data, batch_size=32)
    return trainloaders, valloaders, testloader

trainloaders, valloaders, testloader = load_datasets(NUM_CLIENTS)
trainpoison, valpoison, testpoison = load_datasets(NUM_CLIENTS, poison=True)
def client_fn(cid) -> FlowerClient:
    net = LeNet().to(DEVICE)
    if int(cid) < args.num_adv:
        trainloader = trainpoison[int(cid)]
        valloader = valpoison[int(cid)]
    else:
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
    return FlowerClient(cid, net, trainloader, valloader)


def evaluate(server_round, parameters, config):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LeNet()
    set_parameters(model, parameters)
    model.to(device)

    data = MNIST("./data", train=False, download=True, transform = 
        Compose([ToTensor(), Normalize(mean=0.5, std=0.5)]))
    testloader = torch.utils.data.DataLoader(data, batch_size=32)
    loss, accuracy = test(model, testloader)

    return loss, {"accuracy": accuracy}

strategy = FedAvg(evaluate_fn=evaluate)
if args.defense == 'mean':
    strategy = FedAvg(evaluate_fn=evaluate)
elif args.defense == 'median':
    strategy = FLMedian(evaluate_fn=evaluate)
elif args.defense == 'euclid':
    strategy = FLEuclid(evaluate_fn=evaluate, initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(LeNet())),)
elif args.defense == 'cosine':
    strategy = FLCosine(evaluate_fn=evaluate, initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(LeNet())),)


fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)