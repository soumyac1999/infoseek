import os

import numpy as np
import torch
from tqdm import tqdm


def train_epoch(net, trainloader, criterion, optimizer, device):
    train_loss = 0.0
    net.train()

    for i, train_data in enumerate(trainloader):
        task_loss = 0.0
        for task_id, task_data in enumerate(train_data):
            batch_data = [t.to(device) for t in task_data]

            (
                stage1inp,
                stage1tgt,
                stage2mask,
                stage2inp,
                stage2tgt,
                stage3mask,
                stage3inp,
                stage3tgt,
                stage4mask,
                stage4inp,
                stage4tgt,
                uid,
            ) = batch_data

            optimizer.zero_grad()

            outputs = net(task_id, stage1inp, stage2inp, stage3inp, stage4inp, uid)

            loss = criterion(batch_data, outputs)
            loss.backward()
            optimizer.step()

            task_loss += loss.item()
        train_loss += task_loss / (task_id + 1)

    return net, train_loss / (i + 1)


def eval_net(net, dataloader, criterion, device):
    eval_loss = 0.0
    net.eval()
    for i, train_data in enumerate(dataloader):
        task_loss = 0.0
        for task_id, task_data in enumerate(train_data):
            batch_data = [t.to(device) for t in task_data]

            (
                stage1inp,
                stage1tgt,
                stage2mask,
                stage2inp,
                stage2tgt,
                stage3mask,
                stage3inp,
                stage3tgt,
                stage4mask,
                stage4inp,
                stage4tgt,
                uid,
            ) = batch_data

            outputs = net(task_id, stage1inp, stage2inp, stage3inp, stage4inp, uid)

            loss = criterion(batch_data, outputs)

            task_loss += loss.item()
        eval_loss += task_loss / (task_id + 1)

    return net, eval_loss / (i + 1)


def train(net, epochs, trainloader, valloader, criterion, optimizer, device="cuda:0"):
    os.makedirs("models", exist_ok=True)
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        net, train_loss = train_epoch(net, trainloader, criterion, optimizer, device)
        net, eval_loss = eval_net(net, valloader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(eval_loss)

        torch.save(net.state_dict(), f"models/{epoch}.pth")

        tqdm.write(
            f"Epoch {epoch}: Train Loss = {train_loss:.6f} Val Loss = {eval_loss:.6f}"
        )

    best_ep = epochs - np.argmin(val_losses[::-1]) - 1
    net.load_state_dict(torch.load(f"models/{best_ep}.pth"))
    return net, train_losses, val_losses
