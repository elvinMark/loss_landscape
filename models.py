import torch
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self,new_shape):
        super().__init__()
        self.new_shape = new_shape

    def forward(self,x):
        return x.view(self.new_shape)

def create_model(args):
    if args.architecture == "mlp":
        if args.dataset == "MNIST":
            return nn.Sequential(
                Reshape((-1,784)),
                nn.Linear(784,256),
                nn.ReLU(inplace=True),
                nn.Linear(256,10)
            )
        else:
            return None
    elif args.architecture == "cnn":
        if args.dataset == "MNIST":
            return nn.Sequential(
                nn.Conv2d(1,8,4,stride=2,padding=1,bias=False), # in: 1x28x28 -> out: 8x14x14
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True),
                nn.Conv2d(8,16,4,stride=2,padding=2,bias=False), #in:8x14x14 -> out:16x8x8 
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16,32,4,stride=2,bias=False), #in: 16x8x8 -> out: 32x3x3
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                Reshape((-1,288)),
                nn.Linear(288,64),
                nn.ReLU(inplace=True),
                nn.Linear(64,10)
            )
        else:
            return None
    else:
        return None
