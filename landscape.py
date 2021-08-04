import torch
import torch.nn as nn
import argparse
import os
from models import create_model
from datasets import create_dataloaders
from utils import get_pca_model
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="train a model and visualize the loss landscape")

parser.add_argument("--architecture",type=str,default="mlp",choices=["mlp","cnn"],help="specify the network architecture that was used")
parser.add_argument("--dataset",type=str,default="MNIST",help="specify the dataset to be used")
parser.add_argument("--batch-size",type=int,default=128,help="specify the batch size")
parser.add_argument("--experiment",type=str,default="experiment",help="specify the name of the experiment")
parser.add_argument("--path",type=str,default="./log",help="specify the path in which the models were saved")
parser.add_argument("--epochs",type=int,default=50,help="specify the number of epochs used during training")
parser.add_argument("--checkpoint",type=int,default=1,help="specify how frequent the model was saved during training. value should be greater than 0")

args = parser.parse_args()
args.path = os.path.join(args.path,args.experiment)

if args.checkpoint < 1:
    assert("wrong value for checkpoint")

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_dl, test_dl = create_dataloaders(args)
crit = nn.CrossEntropyLoss()

reduced_w, pca = get_pca_model(args,dev)
plt.scatter(reduced_w[:,0],reduced_w[:,1])
plt.savefig("./results/test.png")

