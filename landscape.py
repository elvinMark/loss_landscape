import torch
import torch.nn as nn
import argparse
import os
from models import create_model
from datasets import create_dataloaders
from utils import get_landscape, create_random_directions, get_params_ref
import matplotlib.pyplot as plt
import pickle
from pca import load_pca_directions

parser = argparse.ArgumentParser(
    description="train a model and visualize the loss landscape"
)

parser.add_argument(
    "--architecture",
    type=str,
    default="mlp",
    choices=["mlp", "cnn", "vgg9"],
    help="specify the network architecture that was used",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="MNIST",
    choices=["MNIST", "CIFAR10"],
    help="specify the dataset to be used",
)
parser.add_argument(
    "--batch-size", type=int, default=128, help="specify the batch size"
)
parser.add_argument(
    "--optim",
    type=str,
    default="sgd",
    choices=["sgd", "sam"],
    help="specify the optimizer",
)
parser.add_argument(
    "--experiment",
    type=str,
    default="experiment",
    help="specify the name of the experiment",
)
parser.add_argument(
    "--path",
    type=str,
    default="./log",
    help="specify the path in which the models were saved",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=50,
    help="specify the number of epochs used during training",
)

parser.add_argument(
    "--range",
    type=float,
    nargs="+",
    default=(-0.5, 0.5),
    help="specify the range of the x and y",
)

parser.add_argument("--gpu", type=int, default=0, help="specify which gpu to use")
parser.add_argument(
    "--pca",
    type=str,
    default=None,
    help="specify the directory where the pca directions are stored",
)

args = parser.parse_args()
args.path = os.path.join(args.path, args.experiment + "_last")
dev = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
model = create_model(args).to(dev)
model.load_state_dict(torch.load(args.path))
crit = nn.CrossEntropyLoss()
train_dl, test_dl = create_dataloaders(args)
base_w = get_params_ref(model)

if args.pca is None:
    print("Creating random directions")
    alpha, beta = create_random_directions(base_w, dev)
else:
    print("Loading the pca directions")
    alpha, beta = load_pca_directions(dev)

X, Y, Z = get_landscape(
    model,
    test_dl,
    crit,
    dev,
    base_w,
    alpha,
    beta,
    x_range=args.range,
    y_range=args.range,
)

data_ = {
    "dataset": args.dataset,
    "architecture": args.architecture,
    "X": X,
    "Y": Y,
    "Z": Z,
}

with open(f"data_{args.batch_size}_{args.optim}", "wb") as f:
    pickle.dump(data_, f)
    f.close()

plt.figure(1)
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(X, Y, Z)
plt.savefig(f"./results/{args.experiment}_surface.png")

plt.figure(2)
plt.contour(X, Y, Z)
plt.savefig(f"./results/{args.experiment}_contour.png")
