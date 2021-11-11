import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from models import create_model
import math


class SmallIterator:
    def __init__(self, x, y, small_bs=128):
        self.x = x
        self.y = y
        self.sbs = small_bs
        self.curr = 0
        self.total_length = len(x)
        self.length = math.ceil(len(x) / small_bs)

    def __len__(self):
        return self.length

    def __iter__(self):
        self.curr = 0
        return self

    def __next__(self):
        if self.curr < self.total_length:
            tmp_x = self.x[self.curr : self.curr + self.sbs]
            tmp_y = self.y[self.curr : self.curr + self.sbs]
            self.curr += self.sbs
            return tmp_x, tmp_y
        else:
            raise StopIteration


def get_parameters(model):
    w = torch.tensor([])
    for param in model.parameters():
        w = torch.cat((w, param.clone().detach().view(-1).cpu()))
    return w


def modify_parameters(model, w):
    idx = 0
    for param in model.parameters():
        tot_elem = torch.prod(torch.tensor(param.shape))
        param.data = w[idx : idx + tot_elem].reshape(param.shape)
        idx += tot_elem


def calculate_loss(model, test_dl, crit, dev):
    test_loss = 0.0

    for x, y in test_dl:
        x = x.to(dev)
        y = y.to(dev)
        o = model(x)
        l = crit(o, y)
        test_loss += l

    return test_loss


def get_pca_model(args, dev):
    w = []
    for epoch in range(args.epochs):
        if epoch % args.checkpoint == 0:
            model = create_model(args).to(dev)
            model.load_state_dict(torch.load(args.path + f"_{epoch}"))
            w_ = get_parameters(model)
            w.append(w_.detach().numpy())
    w = np.array(w)
    pca = PCA(n_components=2)
    reduced_w = pca.fit_transform(w)
    return reduced_w, pca


def get_landscape(base_model, test_dl, crit, pca, dev):
    base_w = get_parameters(base_model)
    w_dir1 = torch.rand_like(base_w)
    w_dir2 = torch.rand_like(base_w)


def train(model, train_dl, test_dl, crit, optim, sched, dev, args):
    try:
        import wandb

        wandb.init(project=args.project, name=args.experiment)
        logger = wandb.log
    except:
        logger = print

    best_acc = 0.0

    for epoch in range(args.epochs):
        train_loss = 0.0
        total = 0.0
        correct = 0.0

        model.train()
        for idx, (x, y) in enumerate(train_dl):
            x = x.to(dev)
            y = y.to(dev)
            optim.zero_grad()
            si = SmallIterator(x, y)
            si_length = len(si)
            for x_, y_ in si:
                o = model(x_)
                l = crit(o, y_)
                l = l / si_length
                l.backward()
                train_loss += l
                top1 = torch.argmax(o, axis=1)
                correct += torch.sum(top1 == y_)
            optim.step()
            total += len(y)

        train_loss = train_loss / len(train_dl)
        train_acc = 100.0 * correct / total

        model.eval()
        test_loss, test_acc = validate(model, test_dl, crit, dev)

        sched.step()

        best_acc = max(best_acc, test_acc)

        logger(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )

        if (
            args.checkpoint != -1
            and args.checkpoint != 0
            and epoch % args.checkpoint == 0
        ):
            torch.save(model.state_dict(), args.path + f"_{epoch}")
            torch.save(test_loss, args.path + f"_loss_{epoch}")

    logger({"best_test_acc": best_acc})

    if args.checkpoint == -1:
        torch.save(model.state_dict(), args.path + "_last")


def validate(model, test_dl, crit, dev):
    test_loss = 0.0
    total = 0.0
    correct = 0.0

    for x, y in test_dl:
        x = x.to(dev)
        y = y.to(dev)
        si = SmallIterator(x, y)
        si_length = len(si)
        for x_, y_ in si:
            with torch.no_grad():
                o = model(x_)
                l = crit(o, y_)
            l = l / si_length
            test_loss += l
            top1 = torch.argmax(o, axis=1)
            correct += torch.sum(top1 == y_)
        total += len(y)

    test_acc = 100 * correct / total
    test_loss = test_loss / len(test_dl)
    return test_loss, test_acc
