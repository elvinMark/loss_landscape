import torch
import torch.nn as nn
import argparse
import os
from models import create_model
from datasets import create_dataloaders

parser = argparse.ArgumentParser(description="train a model and visualize the loss landscape")

parser.add_argument("--architecture",type=str,default="mlp",choices=["mlp","cnn"],help="specify the network architecture to be used")
parser.add_argument("--dataset",type=str,default="MNIST",help="specify the dataset to be used")
parser.add_argument("--batch-size",type=int,default=128,help="specify the batch size")
parser.add_argument("--epochs",type=int,default=50,help="specify the number of epochs to be used during training")
parser.add_argument("--project",type=str,default="project",help="specify the name of the wandb project")
parser.add_argument("--experiment",type=str,default="experiment",help="specify the name of the experiment")
parser.add_argument("--checkpoint",type=int,default=-1,help="specify how frequent to save the model while training. -1 indicates that it just saves the last trained model")
parser.add_argument("--path",type=str,default="./log",help="specify the path in which the models are going to be saved")

args = parser.parse_args()
args.path = os.path.join(args.path,args.experiment)

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = create_model(args).to(dev)
train_dl, test_dl = create_dataloaders(args)
crit = nn.CrossEntropyLoss()

model.load_state_dict(args.path)




