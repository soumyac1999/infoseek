import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.stats import sem

from utils import equal_points_bin


def plot_losses(train_losses, val_losses):
    fig = plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.legend(["Train Loss", "Val Loss"])
    return fig


def framing_bias_plot_1a(net, datasets, device):
    """
    Prob of guessing in Stage 1
    """
    net.eval()

    eval_data = np.zeros((10, 2))
    eval_data[:, 0] = (np.arange(10) + 1.0 - 5.5) / 4.5

    fig = plt.figure(figsize=(4 * 2, 4))

    guess_probs_AA = []
    guess_probs_AB = []

    for i in range(2):
        dataset = datasets[i]

        # AA trial
        eval_data[:, 1] = 1.0

        guess_prob_AA = 0
        for uid in np.unique(dataset.uid):
            uid = torch.tensor(uid, dtype=torch.long).to(device)
            sub_emb = net.sub_emb(uid)
            outputs = net.parallel_nets[i].stage1net(
                torch.tensor(eval_data, dtype=torch.float32).to(device), sub_emb
            )[1][0]
            guess_prob_AA += torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        guess_prob_AA /= len(np.unique(dataset.uid))

        guess_probs_AA.append(guess_prob_AA)

        # AB trial
        eval_data[:, 1] = 0.0

        guess_prob_AB = 0
        for uid in np.unique(dataset.uid):
            uid = torch.tensor(uid, dtype=torch.long).to(device)
            sub_emb = net.sub_emb(uid)
            outputs = net.parallel_nets[i].stage1net(
                torch.tensor(eval_data, dtype=torch.float32).to(device), sub_emb
            )[1][0]
            guess_prob_AB += torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
        guess_prob_AB /= len(np.unique(dataset.uid))

        guess_probs_AB.append(guess_prob_AB)

        stage1guessdf = (
            dataset.df.groupby(["AA", "card1val", "uid"])["guess@1"]
            .mean()
            .groupby(["AA", "card1val"])
            .mean()
        )
        prob_AA = stage1guessdf[1]
        prob_AB = stage1guessdf[0]

        plt.subplot(1, 2, i + 1)
        plt.plot(np.arange(10) + 1, guess_prob_AA, "b+-")
        plt.plot(np.arange(10) + 1, guess_prob_AB, "g+-")
        plt.plot(np.arange(10) + 1, prob_AA, "b+--")
        plt.plot(np.arange(10) + 1, prob_AB, "g+--")

        plt.ylim([0, 1])
        if i == 0:
            plt.title("Multiply Big")
        elif i == 1:
            plt.title("Multiply Small")
        plt.xlabel("First Card Value")
        plt.ylabel("Probability of guessing")
        plt.legend(
            ["Model AA", "Model AB", "Behav AA", "Behav AB"]
        )  # , 'Hunt AA', 'Hunt AB'])
        plt.tight_layout()

    return fig, guess_probs_AA, guess_probs_AB


def framing_bias_plot_1b_model(guess_probs_AA, guess_probs_AB):
    fig = plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.plot(np.arange(10) + 1, guess_probs_AA[0], "b+-")
    plt.plot(np.arange(10) + 1, guess_probs_AA[1], "bo-")
    plt.legend(["Big", "Small"])
    plt.title("AA trial")
    plt.ylim([0, 1])
    plt.ylabel("Prob. of guessing")
    plt.xlabel("Card 1 Value")
    plt.subplot(1, 3, 2)
    plt.plot(np.arange(10) + 1, guess_probs_AB[0], "g+-")
    plt.plot(np.arange(10) + 1, guess_probs_AB[1], "go-")
    plt.legend(["Big", "Small"])
    plt.title("AB trial")
    plt.ylim([0, 1])
    plt.ylabel("Prob. of guessing")
    plt.xlabel("Card 1 Value")
    plt.subplot(1, 3, 3)
    plt.plot(np.arange(10) + 1, guess_probs_AA[0] - guess_probs_AA[1], "bs-")
    plt.plot(np.arange(10) + 1, guess_probs_AB[0] - guess_probs_AB[1], "g^-")
    plt.title("Big - Small")
    plt.legend(["AA", "AB"])
    plt.ylim([-0.2, 0.2])
    plt.ylabel("Prob. of guessing")
    plt.xlabel("Card 1 Value")
    plt.suptitle("Model")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def framing_bias_plot_1b_behav(big, small):
    fig = plt.figure(figsize=(9, 3))
    plt.subplot(1, 3, 1)
    plt.plot(big[1], "b+-")
    plt.plot(small[1], "bo-")
    plt.legend(["Big", "Small"])
    plt.title("AA trial")
    plt.ylim([0, 1])
    plt.ylabel("Prob. of guessing")
    plt.xlabel("Card 1 Value")
    plt.subplot(1, 3, 2)
    plt.plot(big[0], "g+-")
    plt.plot(small[0], "go-")
    plt.legend(["Big", "Small"])
    plt.title("AB trial")
    plt.ylim([0, 1])
    plt.ylabel("Prob. of guessing")
    plt.xlabel("Card 1 Value")
    plt.subplot(1, 3, 3)
    plt.plot(big[1] - small[1], "bs-")
    plt.plot(big[0] - small[0], "g^-")
    plt.title("Big - Small")
    plt.legend(["AA", "AB"])
    plt.ylim([-0.2, 0.2])
    plt.ylabel("Prob. of guessing")
    plt.xlabel("Card 1 Value")
    plt.suptitle("Behav")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def paper_plot_1b(big, small, guess_probs_AB, probsAB):
    fig = plt.figure(figsize=(4, 4))
    plt.plot(np.arange(10) + 1, np.zeros(10), "k-.")
    plt.plot(np.arange(10) + 1, np.array(big[0] - small[0]), "blue", marker="o")
    plt.plot(
        np.arange(10) + 1,
        guess_probs_AB[0] - guess_probs_AB[1],
        "tab:green",
        linestyle="--",
        marker="^",
    )
    plt.plot(
        np.arange(10) + 1,
        probsAB[2, :][::-1] - probsAB[2, :],
        "red",
        linestyle="--",
        marker="x",
    )
    plt.legend(
        ["Optimal (No bias)", "Human(max-min)", "Model(max-min)", "Baseline(max-min)"],
        loc="lower right",
    )
    plt.ylabel("Rel. Prob. of guessing")
    plt.xlabel("Card 1 Value")
    plt.ylim([-0.2, 0.2])
    plt.tight_layout()
    return fig


def rejecting_unsampled_bias_plot(net, dataset, task_id, device):
    """
    Prob of guessing row A Stage1 guess
    """
    net.eval()

    stage1guess_row_df = (
        dataset.df[dataset.df["numMoves"] == 1]
        .groupby(["AA", "card1val", "uid"])["rowAchosen"]
        .mean()
        .groupby(["AA", "card1val"])
        .mean()
    )

    prob_AA = stage1guess_row_df[1]
    prob_AB = stage1guess_row_df[0]

    eval_data = np.zeros((10, 2))
    eval_data[:, 0] = (np.arange(10) + 1.0 - 5.5) / 4.5

    ## AA trial
    eval_data[:, 1] = 1.0

    guess_prob_mult_big_AA = 0
    for uid in np.unique(dataset.uid):
        uid = torch.tensor(uid, dtype=torch.long).to(device)
        sub_emb = net.sub_emb(uid)
        outputs = net.parallel_nets[task_id].stage1net(
            torch.tensor(eval_data, dtype=torch.float32).to(device), sub_emb
        )[1][1]
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        guess_prob_mult_big_AA += probs[:, 1]
    guess_prob_mult_big_AA /= len(np.unique(dataset.uid))

    ## AB trial
    eval_data[:, 1] = 0.0

    guess_prob_mult_big_AB = 0
    for uid in np.unique(dataset.uid):
        uid = torch.tensor(uid, dtype=torch.long).to(device)
        sub_emb = net.sub_emb(uid)
        outputs = net.parallel_nets[task_id].stage1net(
            torch.tensor(eval_data, dtype=torch.float32).to(device), sub_emb
        )[1][1]
        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        guess_prob_mult_big_AB += probs[:, 1]
    guess_prob_mult_big_AB /= len(np.unique(dataset.uid))

    fig = plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(10) + 1, guess_prob_mult_big_AA, "b+-")
    plt.plot(np.arange(10) + 1, guess_prob_mult_big_AB, "g+-")
    plt.plot(np.arange(10) + 1, prob_AA, "b+--")
    plt.plot(np.arange(10) + 1, prob_AB, "g+--")
    # plt.plot(np.arange(10)+1, probsAA[0]/(1-probsAA[2]), 'b+:')
    # plt.plot(np.arange(10)+1, probsAB[0]/(1-probsAB[2]), 'g+:')

    plt.ylim([0, 1])
    if task_id == 0:
        plt.title("MaxProd")
    elif task_id == 1:
        plt.title("MinProd")
    plt.xlabel("First Card Value")
    plt.ylabel("Probability of choosing row A")
    plt.legend(
        ["Model Same", "Model Diff", "Behav Same", "Behav Diff"]
    )  # , 'Hunt AA', 'Hunt AB'])

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(10) + 1, guess_prob_mult_big_AA - guess_prob_mult_big_AB, "c+-")
    plt.plot(np.arange(10) + 1, prob_AA - prob_AB, "c+--")
    # plt.plot(np.arange(10)+1, probsAA[0]/(1-probsAA[2])-probsAB[0]/(1-probsAB[2]), 'c+:')
    plt.ylim([-0.1, 0.1])
    if task_id == 0:
        plt.title("MaxProd")
    elif task_id == 1:
        plt.title("MinProd")
    plt.xlabel("First Card Value")
    plt.ylabel("Relative probability of choosing row A")
    plt.legend(["Model(Same-Diff)", "Behav(Same-Diff)"])  # , 'Hunt'])
    plt.tight_layout()
    return fig


def approaching_postive_bias_plot1(net, dataset, task_id, device):
    net = net.eval()

    probs = 0

    ret = []

    for uid in np.unique(dataset.uid):
        inp1 = np.stack([(np.arange(100) // 10 + 1 - 5.5) / 4.5, np.zeros(100)]).T
        inp2 = np.stack([(np.arange(100) % 10 + 1 - 5.5) / 4.5]).T
        inp1 = torch.tensor(inp1, dtype=torch.float).to(device)
        inp2 = torch.tensor(inp2, dtype=torch.float).to(device)
        dummy = torch.empty(100, 1).to(device)
        uid = torch.tensor(uid, dtype=torch.long).to(device)
        probs += (
            torch.softmax(net(task_id, inp1, inp2, dummy, dummy, uid)[1][2], axis=1)[
                :, 1
            ]
            .detach()
            .cpu()
            .numpy()
        )

    prob_A = probs.reshape(10, 10) / len(np.unique(dataset.uid))
    ret.append(prob_A)

    fig = plt.figure(figsize=(7.5, 3.5))
    plt.subplot(121)
    plt.imshow(prob_A, cmap="jet", vmin=0.3, vmax=0.7, origin="lower")
    plt.title("Prob. of samp. row A (Model)")
    plt.xlabel("Second Card Value (Row B)")
    plt.ylabel("First Card Value (Row A)")
    plt.colorbar()

    df = dataset.df

    t = (
        df[(df["numMoves"] >= 3) & (df["AA"] != 1)]
        .groupby(["card1val", "card2val", "uid"])["sampleA@2"]
        .mean()
        .groupby(["card1val", "card2val"])
        .mean()
    )
    prob_A = t.to_numpy().reshape(10, 10)
    ret.append(prob_A)

    plt.subplot(122)
    plt.imshow(prob_A, cmap="jet", vmin=0.3, vmax=0.7, origin="lower")
    plt.title("Prob. of samp. row A (Human)")
    plt.xlabel("Second Card Value (Row B)")
    plt.ylabel("First Card Value (Row A)")
    plt.colorbar()

    if task_id == 0:
        plt.suptitle("MaxProd")
    elif task_id == 1:
        plt.suptitle("MinProd")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig, ret


def approaching_postive_bias_plot2(ret_big, ret_small):
    fig = plt.figure(figsize=(7, 3.5))
    plt.subplot(121)
    plt.imshow(
        ret_big[0] - ret_small[0], cmap="hot", vmin=-0.3, vmax=0.3, origin="lower"
    )
    plt.title("Big - Small (Model)")
    plt.xlabel("Second Card Value (Row B)")
    plt.ylabel("First Card Value (Row A)")
    plt.colorbar()

    plt.subplot(122)
    plt.imshow(
        ret_big[1] - ret_small[1], cmap="hot", vmin=-0.3, vmax=0.3, origin="lower"
    )
    plt.title("Big - Small (Human)")
    plt.xlabel("Second Card Value (Row B)")
    plt.ylabel("First Card Value (Row A)")
    plt.colorbar()
    plt.tight_layout()
    return fig


def approach_avoid_plots(feature_df):
    fig = plt.figure(figsize=(3 * 3, 4))
    for i, t in enumerate(
        [
            ("approach", "Approach"),
            ("avoid", "Avoid"),
            ("approach-avoid", "Approach-Avoid"),
        ]
    ):
        p = feature_df[t[0]]
        med = np.median(p)
        low_u = list(feature_df["uid"][p < med])
        high_u = list(feature_df["uid"][p >= med])

        low_p1 = feature_df["subemb1"][feature_df["uid"].isin(low_u)]
        high_p1 = feature_df["subemb1"][feature_df["uid"].isin(high_u)]
        low_p2 = feature_df["subemb2"][feature_df["uid"].isin(low_u)]
        high_p2 = feature_df["subemb2"][feature_df["uid"].isin(high_u)]

        plt.subplot(1, 3, i + 1)
        bincenters = np.array([0, 0.31])
        width = 0.3
        y = [-np.mean(low_p1), -np.mean(high_p1)]
        yerr = [sem(low_p1), sem(high_p1)]
        plt.bar(bincenters, y, width=width, color="r", yerr=yerr)

        y = [np.mean(low_p2), np.mean(high_p2)]
        yerr = [sem(low_p2), sem(high_p2)]
        plt.bar(bincenters + 1, y, width=width, color="b", yerr=yerr)

        plt.legend(["subemb1", "subemb2"])
        plt.ylabel("Subj Emb")
        plt.xticks(list(bincenters) + list(bincenters + 1), ["low", "high"] * 2)
        plt.ylim([0, 0.35])
        plt.xlabel(f"{t[1]} Type")
    plt.tight_layout()
    return fig


def sim_behav_corr(scores, steps, success, avg_score, avg_steps, avg_success):
    fig = plt.figure(figsize=(12, 4))
    plt.subplot(131)
    sns.regplot(avg_steps, steps, marker=".")
    plt.xlabel("Human Steps")
    plt.ylabel("Simulated Steps")
    plt.legend([f"r = {np.corrcoef(avg_steps, steps)[0,1]:.4f}"])
    plt.subplot(132)
    sns.regplot(avg_success, success, marker=".")
    plt.xlabel("Human Accuracy")
    plt.ylabel("Simulated Accuracy")
    plt.legend([f"r = {np.corrcoef(avg_success, success)[0,1]:.4f}"])
    plt.subplot(133)
    sns.regplot(avg_score, scores, marker=".")
    plt.xlabel("Human Scores")
    plt.ylabel("Simulated Scores")
    plt.legend([f"r = {np.corrcoef(avg_score, scores)[0,1]:.4f}"])
    plt.tight_layout()
    return fig


def sub_emb_demographics(feature_df):
    xs = ["education", "age", "gender"]
    xticks = [
        ([0, 1, 2, 3], ["GSCE", "ALev", "Grad", "PhD"]),
        ([1, 2, 3, 4, 5, 6], ["18-25", "25-29", "30-39", "40-49", "50-59", "60-69"]),
        ([0, 1], ["Male", "Female"]),
    ]

    fig = plt.figure(figsize=(3.25 * len(xs), 6))
    for i, x in enumerate(xs):
        plt.subplot(2, len(xs), i + 1)
        feat = feature_df.groupby(x).agg([np.mean, np.std, sem])[["subemb1", "subemb2"]]
        if x == "age":
            feat = feat[:-1]
        plt.errorbar(feat.index, feat["subemb1"]["mean"], yerr=feat["subemb1"]["sem"])
        plt.xlabel(x)
        plt.xticks(*xticks[i])
        plt.ylabel("subemb1")
        plt.subplot(2, len(xs), len(xs) + i + 1)
        plt.errorbar(feat.index, feat["subemb2"]["mean"], yerr=feat["subemb2"]["sem"])
        plt.xlabel(x)
        plt.xticks(*xticks[i])
        plt.ylabel("subemb2")
    plt.tight_layout()
    return fig


def sub_emb_behav(feature_df):
    xs = [
        "avg_moves",
        "success",
        "avg_dectime",
        "avg_score",
        "avg_guesstime",
    ]

    fig = plt.figure(figsize=(3 * len(xs), 6))
    for i, x in enumerate(xs):
        plt.subplot(2, len(xs), i + 1)
        bins = equal_points_bin(feature_df[x], 8)
        feature_df[f"digi_{x}"] = np.digitize(feature_df[x], bins)
        feat = feature_df.groupby(f"digi_{x}").agg([np.mean, np.std, sem])[
            ["subemb1", "subemb2"]
        ]
        plt.errorbar(feat.index, feat["subemb1"]["mean"], yerr=feat["subemb1"]["sem"])
        plt.xlabel(x)
        plt.ylabel("subemb1")
        plt.xticks(
            ticks=np.arange(1, 1 + len(bins))[1::2],
            labels=[f"{t:.3f}" for t in bins][1::2],
        )
        plt.subplot(2, len(xs), len(xs) + i + 1)
        plt.errorbar(feat.index, feat["subemb2"]["mean"], yerr=feat["subemb2"]["sem"])
        plt.xlabel(x)
        plt.ylabel("subemb2")
        plt.xticks(
            ticks=np.arange(1, 1 + len(bins))[1::2],
            labels=[f"{t:.3f}" for t in bins][1::2],
        )
    plt.tight_layout()
    return fig
