import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from baseline_model import get_baseline
from datasets import MultiTaskDataset, TaskSpecificDataset
from get_dataframe import get_dataframe
from loss import Loss
from networks import MultiTaskModel, SimModel
from nll_utils import baseline_stage_wise_NLL, stage_wise_NLL
from plot_utils import (
    approach_avoid_plots,
    approaching_postive_bias_plot1,
    approaching_postive_bias_plot2,
    framing_bias_plot_1a,
    framing_bias_plot_1b_behav,
    framing_bias_plot_1b_model,
    paper_plot_1b,
    plot_losses,
    rejecting_unsampled_bias_plot,
    sim_behav_corr,
    sub_emb_behav,
    sub_emb_demographics,
)
from sim_utils import simulate
from train_eval_utils import train
from utils import split_df_by_uid


def main(args):
    # Try loading cached df
    if args.approach_avoid:
        file = "data/TGBE_cardturn_data_40686_robbparam"
    else:
        file = "data/TGBE_cardturn_data_54145"

    try:
        df = pickle.load(open(file + ".pkl", "rb"))
    except:
        df = get_dataframe(file)
        pickle.dump(df, open(file + ".pkl", "wb"))

    # Remap uid
    uids = set(df[df["trialtype"] == args.tasks[0]]["uid"])
    for i in args.tasks[1:]:
        uids = uids.intersection(set(df[df["trialtype"] == i]["uid"]))
    uids = list(sorted(uids))
    d = {k: i + 1 for i, k in enumerate(uids)}
    ld = len(d)
    for i, h in enumerate(np.array(list(set(df["uid"]) - set(uids)))):
        d[h] = i + 1 + ld
    df["uid"] = df["uid"].apply(lambda x: d[x])

    uids = [d[u] for u in uids]

    # Split into train-val sets
    mult_train_df, mult_val_df = split_df_by_uid(
        df[df["uid"].isin(uids)], frac=args.split_frac
    )

    # Check to avoid CUDA args.device side assert trigger for subject embedding
    assert max(mult_train_df["uid"].max(), mult_val_df["uid"].max()) == len(uids)

    # Get datasets
    mult_train = MultiTaskDataset(mult_train_df, args.tasks)
    mult_val = MultiTaskDataset(mult_val_df, args.tasks)

    # Get the corresponding dataloaders
    mult_trainloader = DataLoader(
        mult_train,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    mult_valloader = DataLoader(
        mult_val, batch_size=2000, shuffle=False, num_workers=4, pin_memory=True
    )

    net = MultiTaskModel(len(uids), len(args.tasks)).to(args.device)
    criterion = Loss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    net, train_losses, val_losses = train(
        net,
        args.epochs,
        mult_trainloader,
        mult_valloader,
        criterion,
        optimizer,
        args.device,
    )

    # Get baseline and Stage-wise NLL values
    probsAA, probsAB, valsAA, valsAB, vals2 = get_baseline()
    if args.calc_nll:
        stage_wise_NLL(net, mult_val_df, args.device)
        baseline_stage_wise_NLL(valsAA, valsAB, vals2, mult_val_df)

    # Loss Plots
    if not args.paper_plots_only:
        plot_losses(train_losses, val_losses).show()

    mult_big_dataset = TaskSpecificDataset(mult_val_df, 4)
    mult_small_dataset = TaskSpecificDataset(mult_val_df, 5)

    if not args.approach_avoid:
        fig, guess_probs_AA, guess_probs_AB = framing_bias_plot_1a(
            net, [mult_big_dataset, mult_small_dataset], args.device
        )
        if not args.paper_plots_only:
            fig.show()

            framing_bias_plot_1b_model(guess_probs_AA, guess_probs_AB).show()

        big = mult_big_dataset.df.groupby(["AA", "card1val"]).mean()["guess@1"]
        small = mult_small_dataset.df.groupby(["AA", "card1val"]).mean()["guess@1"]
        if not args.paper_plots_only:
            framing_bias_plot_1b_behav(big, small).show()

        # Paper figure 1B
        paper_plot_1b(big, small, guess_probs_AB, probsAB).show()

        # Paper figure 5
        rejecting_unsampled_bias_plot(net, mult_big_dataset, 0, args.device).show()
        if not args.paper_plots_only:
            rejecting_unsampled_bias_plot(
                net, mult_small_dataset, 1, args.device
            ).show()

        # Paper figure 4
        fig_big, ret_big = approaching_postive_bias_plot1(
            net, mult_big_dataset, 0, args.device
        )
        fig_small, ret_small = approaching_postive_bias_plot1(
            net, mult_small_dataset, 1, args.device
        )
        fig_big.show()
        fig_small.show()

        if not args.paper_plots_only:
            approaching_postive_bias_plot2(ret_big, ret_small).show()

        input("Wait for plots to generate. Press any key to continue")

    # Sub Emb Analysis
    fdf = pd.concat([mult_big_dataset.df, mult_small_dataset.df], ignore_index=True)

    feature_df = pd.DataFrame()
    feature_df["uid"] = np.unique(fdf["uid"])
    feature_df["subemb1"], feature_df["subemb2"] = (
        net.sub_emb.embed.weight.detach().cpu().numpy()[np.unique(fdf["uid"]) - 1].T
    )
    feature_df["avg_score"] = np.array(fdf.groupby("uid")["score"].mean())
    feature_df["avg_moves"] = np.array(fdf.groupby("uid")["numMoves"].mean())
    feature_df["avg_dectime"] = np.array(fdf.groupby("uid")["decisiontime"].mean())
    feature_df["avg_guesstime"] = np.array(fdf.groupby("uid")["guesstime"].mean())
    feature_df["success"] = np.array(fdf.groupby("uid")["success"].mean())
    feature_df["education"] = np.array(fdf.groupby("uid")["education"].min())
    feature_df["age"] = np.array(fdf.groupby("uid")["age"].min())
    feature_df["gender"] = np.array(fdf.groupby("uid")["gender"].min())

    if args.approach_avoid:
        feature_df["approach"] = np.array(fdf.groupby("uid")["robbp5"].min())
        feature_df["avoid"] = np.array(fdf.groupby("uid")["robbp6"].min())
        feature_df["approach-avoid"] = feature_df["approach"] - feature_df["avoid"]
        f = ["subemb1", "subemb2", "approach", "avoid", "approach-avoid"]
    else:
        f = [
            "subemb1",
            "subemb2",
            "avg_score",
            "avg_moves",
            "avg_dectime",
            "avg_guesstime",
            "success",
        ]

    for tm in ["avg_dectime", "avg_guesstime"]:
        feature_df[tm] = feature_df[tm] - feature_df[tm].min()
        feature_df[tm] = feature_df[tm].replace({0: 1})
        feature_df[tm] = np.log(feature_df[tm])

    if args.approach_avoid:
        # Paper Figure 9
        approach_avoid_plots(feature_df).show()
        input("Wait for plots to generate. Press any key to continue")
    else:
        # Paper Figure 6
        sim_behav_corr(*simulate(SimModel(net), fdf, args.tasks)).show()

        # Paper Figure 7
        sub_emb_behav(feature_df).show()
        sub_emb_demographics(feature_df).show()

        input("Wait for plots to generate. Press any key to continue")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--approach_avoid', default=False, action='store_true', 
        help="Use the data with Approach Avoid Parameters")
    parser.add_argument('--tasks', default=[4,5], nargs='+', type=int,
        help="IDs of tasks to be used")
    parser.add_argument('--split_frac', default=0.6, type=float,
        help="Train-val split fraction")
    parser.add_argument("--train_batch_size", default=256, type=int,
        help="Training batch size")
    parser.add_argument('--device', default="cpu", type=str)
    parser.add_argument('--lr', default=0.003, type=float, help="Learning rate")
    parser.add_argument('--epochs', default=30, type=int, help="Number of training epochs")
    parser.add_argument('--calc_nll', default=False, action='store_true',
        help="Calculate negative log likelihoods for the model and the baseline")
    parser.add_argument('--all_plots', default=False, action='store_true',
        help="Show all plots. Default: Only show the plots which are there in the paper")

    args = parser.parse_args()
    args.paper_plots_only = not args.all_plots

    main(args)
