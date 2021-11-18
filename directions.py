from pca import get_pca_directions
import argparse
import torch
from models import create_model

parser = argparse.ArgumentParser(
    description="This is a program to calculate the main directions (using pca)"
)

parser.add_argument(
    "--models-list",
    type=str,
    default="models_list.txt",
    help="this is a file that contains the list of models",
)

parser.add_argument(
    "--architecture", type=str, default="vgg9", help="specify the architecture"
)

parser.add_argument(
    "--dataset",
    type=str,
    default="CIFAR10",
    help="specify the dataset that was used for training",
)

args = parser.parse_args()
models_list = []

with open(args.models_list, "r") as f:
    for line in f.readlines():
        model = create_model(args)
        model.load_state_dict(torch.load(line[:-1]))
        models_list.append([w_ for w_ in model.parameters()])
    f.close()

get_pca_directions(models_list)
