import torch
import torch.nn as nn
import numpy as np
from sklearn.decomposition import PCA
from models import create_model

def get_parameters(model):
    w = torch.tensor([])
    for param in model.parameters():
        w = torch.cat((w,param.clone().detach().view(-1).cpu()))
    return w

def calculate_loss(model,test_dl,crit,dev):
    test_loss = 0.
    
    for x,y in test_dl:
        x = x.to(dev)
        y = y.to(dev)
        o = model(x)
        l = crit(o,y)
        test_loss += l
        
    return test_loss

def get_pca_model(args,dev):
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

def get_landscape(base_model,test_dl,crit,dev):
    # TODO
    pass

def train(model,train_dl,test_dl,crit,optim,sched,dev,args):
    try:
        import wandb
        wandb.init(
            project = args.project,
            name = args.experiment
        )
        logger = wandb.log
    except:
        logger = print

    best_acc = 0.
    
    for epoch in range(args.epochs):
        train_loss = 0.
        total = 0.
        correct = 0.
        
        model.train()
        for idx, (x,y) in enumerate(train_dl):
            x = x.to(dev)
            y = y.to(dev)
            optim.zero_grad()
            o = model(x)
            l = crit(o,y)
            l.backward()
            optim.step()
            top1 = torch.argmax(o,axis=1)
            correct += torch.sum(top1 == y)
            total += len(y)
            train_loss += l

        train_acc = 100.0* correct / total

        model.eval()
        test_loss, test_acc = validate(model,test_dl,crit,dev)
        
        sched.step()

        best_acc = max(best_acc, test_acc)
        
        logger({
            "epoch": epoch,
            "train_loss" : train_loss,
            "train_acc" : train_acc,
            "test_loss" : test_loss,
            "test_acc" : test_acc
        })

        if args.checkpoint!= -1 and args.checkpoint!=0 and epoch % args.checkpoint == 0:
            torch.save(model.state_dict(),args.path + f"_{epoch}")

    logger({
        "best_test_acc" : best_acc
    })

    if args.checkpoint == -1:
        torch.save(model.state_dict(),args.path + "_last")

def validate(model,test_dl,crit,dev):
    test_loss = 0.
    total = 0.
    correct = 0.
    
    for x,y in test_dl:
        x = x.to(dev)
        y = y.to(dev)
        o = model(x)
        l = crit(o,y)
        test_loss += l
        top1 = torch.argmax(o,axis=1)
        correct += torch.sum(top1 == y)
        total += len(y)

    test_acc = 100 * correct / total
    return test_loss, test_acc
