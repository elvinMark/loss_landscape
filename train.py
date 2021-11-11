import torch
import torch.nn as nn
import argparse
from models import create_model
from datasets import create_dataloaders
from optimizers import create_optimizer
from schedulers import create_lr_scheduler
from utils import train
import os

parser = argparse.ArgumentParser(
    description="train a model and visualize the loss landscape"
)

parser.add_argument(
    "--architecture",
    type=str,
    default="mlp",
    choices=["mlp", "cnn", "vgg9"],
    help="specify the network architecture to be used",
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
    "--epochs",
    type=int,
    default=50,
    help="specify the number of epochs to be used during training",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.01,
    help="specify the base learning rate to be used during training",
)
parser.add_argument(
    "--optim",
    type=str,
    default="sgd",
    help="specify the optimizer to be used in the training",
)
parser.add_argument(
    "--sched", type=str, default="step", help="specify the scheduler to be used"
)
parser.add_argument(
    "--step-size",
    type=int,
    default=10,
    help="specify the step size to be used in the step lr scheduler",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.2,
    help="specify the gamma factor used in the step lr scheduler",
)
parser.add_argument(
    "--T_max",
    type=int,
    default=50,
    help="specify the T_max factor to be used in the cosine lr scheduler",
)
parser.add_argument(
    "--eta_min",
    type=float,
    default=0.0,
    help="specify the eta_min factor to be used in the cosine lr scheduler",
)
parser.add_argument(
    "--milestones",
    type=int,
    nargs="+",
    default=[20, 40],
    help="specify the milestones to be used in the multistep lr scheduler",
)
parser.add_argument(
    "--project",
    type=str,
    default="project",
    help="specify the name of the wandb project",
)
parser.add_argument(
    "--experiment",
    type=str,
    default="experiment",
    help="specify the name of the experiment",
)
parser.add_argument(
    "--checkpoint",
    type=int,
    default=-1,
    help="specify how frequent to save the model while training. -1 indicates that it just saves the last trained model",
)
parser.add_argument(
    "--path",
    type=str,
    default="./log",
    help="specify the path in which the models are going to be saved",
)

args = parser.parse_args()
args.T_max = args.epochs
args.path = os.path.join(args.path, args.experiment)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = create_model(args).to(dev)

train_dl, test_dl = create_dataloaders(args)
crit = nn.CrossEntropyLoss()
optim = create_optimizer(model, args)
sched = create_lr_scheduler(optim, args)


train(model, train_dl, test_dl, crit, optim, sched, dev, args)
