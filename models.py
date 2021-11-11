import torch
import torch.nn as nn


class Reshape(nn.Module):
    def __init__(self, new_shape):
        super().__init__()
        self.new_shape = new_shape

    def forward(self, x):
        return x.view(self.new_shape)


def create_fc(layers=[784, 256, 10]):
    """
    Create fully connected neural network
    """
    hidden_layers_ = []
    for l1, l2 in zip(layers[:-2], layers[1:-1]):
        hidden_layers_.append(nn.Linear(l1, l2))
        hidden_layers_.append(nn.ReLU(inplace=True))
    return nn.Sequential(
        Reshape((-1, layers[0])), *hidden_layers_, nn.Linear(layers[-2], layers[-1])
    )


def create_conv_block(
    in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=False
):
    """
    Create Conv + ReLU + BN block
    """
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        ),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


def create_model(args):
    if args.architecture == "mlp":
        if args.dataset == "MNIST":
            return create_fc()
        else:
            return None
    elif args.architecture == "cnn":
        if args.dataset == "MNIST":
            return nn.Sequential(
                *[
                    create_conv_block(1, 8, kernel_size=4, stride=2, padding=1),
                    create_conv_block(8, 16, kernel_size=4, stride=2, padding=2),
                    create_conv_block(16, 32, kernel_size=4, stride=2),
                    create_fc([288, 64, 10]),
                ]
            )
        else:
            return None
    elif args.architecture == "vgg9":
        if args.dataset == "CIFAR10":
            return nn.Sequential(
                *[
                    create_conv_block(3, 64, padding=1),
                    nn.MaxPool2d(kernel_size=2),
                    create_conv_block(64, 128, padding=1),
                    nn.MaxPool2d(kernel_size=2),
                    create_conv_block(128, 256, padding=1),
                    create_conv_block(256, 256, padding=1),
                    nn.MaxPool2d(kernel_size=2),
                    create_conv_block(256, 512, padding=1),
                    create_conv_block(512, 512, padding=1),
                    nn.MaxPool2d(kernel_size=2),
                    create_conv_block(512, 512, padding=1),
                    create_conv_block(512, 512, padding=1),
                    nn.MaxPool2d(kernel_size=2),
                    create_fc(layers=[512, 512, 10]),
                ]
            )
        else:
            return None

    else:
        return None
