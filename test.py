import torch
import torch.nn as nn
import argparse
import os
from models import create_model
from datasets import create_dataloaders
from utils import validate

parser = argparse.ArgumentParser(description="train a model and visualize the loss landscape")

parser.add_argument("--architecture",type=str,default="mlp",choices=["mlp","cnn"],help="specify the network architecture that was used")
parser.add_argument("--dataset",type=str,default="MNIST",help="specify the dataset that was used")
parser.add_argument("--batch-size",type=int,default=128,help="specify the batch size")
parser.add_argument("--experiment",type=str,default="experiment",help="specify the name of the experiment")
parser.add_argument("--path",type=str,default="./log",help="specify the path in which the models were saved")

args = parser.parse_args()
args.path = os.path.join(args.path,args.experiment)  + "_last"

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = create_model(args).to(dev)
train_dl, test_dl = create_dataloaders(args)
crit = nn.CrossEntropyLoss()

model.load_state_dict(torch.load(args.path))

print(validate(model,test_dl,crit,dev))



