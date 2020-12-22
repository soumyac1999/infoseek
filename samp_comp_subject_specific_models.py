import logging
import os
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from cardenv import CardEnv
from datasets import TaskSpecificDataset
from get_dataframe import get_dataframe
from loss import Loss
from networks import SimSubjModel as SimModel
from networks import SubjModel as Model
from sim_utils import simulate_trial
from utils import split_df_by_uid

device = "cpu"

os.makedirs("models_samp", exist_ok=True)
logging.basicConfig(filename="data.txt", level=logging.INFO)

file = "data/TGBE_cardturn_data_54145"
try:
    df = pickle.load(open(file + ".pkl", "rb"))
except:
    df = get_dataframe(file)
    pickle.dump(df, open(file + ".pkl", "wb"))

# Remap uid
d = {k: i + 1 for i, k in enumerate(np.unique(df[df["trialtype"] == 4]["uid"]))}
ld = len(d)
num_subjects = ld
for i, h in enumerate(
    np.array(list(set(df["uid"]) - set(df[df["trialtype"] == 4]["uid"])))
):
    d[h] = i + 1 + ld
df["uid"] = df["uid"].apply(lambda x: d[x])

# Train and Eval Utils
def train_epoch(net, trainloader, criterion, optimizer, device):
    train_loss = 0.0
    net.train()
    for i, train_data in enumerate(trainloader):
        batch_data = [t.to(device) for t in train_data]

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

        outputs = net(stage1inp, stage2inp, stage3inp, stage4inp, uid)

        loss = criterion(batch_data, outputs)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    return net, train_loss / (i + 1)


def eval_net(net, dataloader, criterion, device):
    eval_loss = 0.0
    net.eval()
    for i, train_data in enumerate(dataloader):
        batch_data = [t.to(device) for t in train_data]

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

        outputs = net(stage1inp, stage2inp, stage3inp, stage4inp, uid)

        loss = criterion(batch_data, outputs)

        eval_loss += loss.item()

    return net, eval_loss / (i + 1)


def train(net, epochs, trainloader, valloader, criterion, optimizer, device="cuda:0"):
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):
        net, train_loss = train_epoch(net, trainloader, criterion, optimizer, device)
        net, eval_loss = eval_net(net, valloader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(eval_loss)

        torch.save(net.state_dict(), f"models_samp/{epoch}.pth")

    best_ep = epochs - np.argmin(val_losses[::-1]) - 1
    net.load_state_dict(torch.load(f"models_samp/{best_ep}.pth"))
    return net, train_losses, val_losses


mult_big_df = df[df["trialtype"] == 4]
subjs = np.unique(mult_big_df["uid"])
np.random.shuffle(subjs)
test_subjs = subjs[:500]
other_subjs = subjs[500:]
testsubjs_df = pd.DataFrame(mult_big_df[mult_big_df["uid"].isin(test_subjs)])
d = {k: i + 1 for i, k in enumerate(np.unique(testsubjs_df["uid"]))}
testsubjs_df["uid"] = testsubjs_df["uid"].apply(lambda x: d[x])
testsubjs_train_df, testsubjs_val_df = split_df_by_uid(testsubjs_df, frac=0.6)

for num_subj in [100, 500, 1000, 2000, 5000, 10000]:
    othersubjs_df = pd.DataFrame(
        mult_big_df[mult_big_df["uid"].isin(other_subjs[:num_subj])]
    )
    d = {k: i + 1 + 500 for i, k in enumerate(np.unique(othersubjs_df["uid"]))}
    othersubjs_df["uid"] = othersubjs_df["uid"].apply(lambda x: d[x])
    othersubjs_df, _ = split_df_by_uid(othersubjs_df, frac=0.6)

    mult_big_train_syn = TaskSpecificDataset(
        pd.concat([testsubjs_train_df, othersubjs_df], ignore_index=True), 4
    )
    mult_big_val = TaskSpecificDataset(testsubjs_val_df, 4)

    # Get the corresponding dataloaders
    batch_size = 256
    mult_big_syn_trainloader = DataLoader(
        mult_big_train_syn,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    mult_big_valloader = DataLoader(
        mult_big_val, batch_size=20000, shuffle=True, num_workers=16, pin_memory=True
    )

    print(mult_big_train_syn.df["uid"].max())
    syn_net = Model(mult_big_train_syn.df["uid"].max()).to(device)

    criterion = Loss()

    lr = 0.003
    optimizer = torch.optim.Adam(syn_net.parameters(), lr=lr)
    epochs = 100

    syn_net, train_losses, val_losses = train(
        syn_net,
        epochs,
        mult_big_syn_trainloader,
        mult_big_valloader,
        criterion,
        optimizer,
        device,
    )

    sim_net = SimModel(syn_net)
    env = CardEnv(4)

    dataset = mult_big_val

    sim_net = sim_net.cpu()
    scores = np.zeros(len(np.unique(dataset.df["uid"])))
    steps = np.zeros(len(np.unique(dataset.df["uid"])))
    success = np.zeros(len(np.unique(dataset.df["uid"])))

    for i, uid in enumerate(np.unique(dataset.df["uid"])):
        uid_ = torch.tensor(uid, dtype=torch.long, device=device)
        score = 0
        step = 0
        success_ = 0
        for l, d in enumerate(
            np.array(
                dataset.df[dataset.df["uid"] == uid][
                    ["AA", "card1", "card2", "card3", "card4"]
                ]
            )
        ):
            aa, *cards = d
            score_, step_ = simulate_trial(sim_net, None, env, uid, aa, np.array(cards))
            score += score_
            step += step_
            success_ += 1 if score_ > 0 else 0
        scores[i] = score / (l + 1)
        steps[i] = step / (l + 1)
        success[i] = success_ / (l + 1)

    avg_score = dataset.df.groupby("uid")["score"].mean()
    avg_steps = dataset.df.groupby("uid")["numMoves"].mean()
    avg_success = dataset.df.groupby("uid")["success"].mean()

    logging.info(
        f"{num_subj}\t{min(val_losses)}\t{np.corrcoef(avg_score, scores)[0,1]}\t{np.corrcoef(avg_steps, steps)[0,1]}\t{np.corrcoef(avg_success, success)[0,1]}"
    )
    print(
        f"{num_subj}\t{min(val_losses)}\t{np.corrcoef(avg_score, scores)[0,1]}\t{np.corrcoef(avg_steps, steps)[0,1]}\t{np.corrcoef(avg_success, success)[0,1]}"
    )
