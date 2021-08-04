import torch
import torch.nn as nn

def get_parameters(model):
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

        train_acc = 100.0* correct / total
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

    logger({
        "best_test_acc" : best_acc
    })

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

