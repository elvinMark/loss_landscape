import torch
from sam import SAM


def create_optimizer(model, args):
    if args.optim == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.lr)

    elif args.optim == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr)

    elif args.optim == "sam":
        return SAM(model.parameters(), lr=args.lr)
    else:
        return None
