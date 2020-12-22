import numpy as np
import pandas as pd
from scipy.io import loadmat


def get_dataframe(file):
    """
    Arguments:
    file can be ['data/TGBE_cardturn_data_54145', 'data/TGBE_cardturn_data_40686_robbparam']

    Returns:
    Processed dataframe for further experiments
    """
    data = loadmat(file)

    df = pd.DataFrame()

    nS = int(data["nS"][0][0])
    nTr = int(data["nTr"][0][0])
    # 0 = FIND THE BIGGEST, 1 = FIND THE SMALLEST, 2 = ADD BIG,
    # 3 = ADD SMALL, 4 = MULTIPLY BIG, 5 = MULTIPLY SMALL
    # Shape: (# Blocks, # Trials/block)
    df["trialtype"] = data["trtrialtype"].reshape(-1)

    # List of 4 card present in the trial
    # Shape: (# Blocks, # Trials/block, 4)
    cardlist = data["trcardlist"].transpose(0, 2, 1)
    cardlist = (cardlist - 1) % 10 + 1
    cardlist = cardlist.reshape(-1, 4)

    df["card1"] = cardlist[:, 0]
    df["card2"] = cardlist[:, 1]
    df["card3"] = cardlist[:, 2]
    df["card4"] = cardlist[:, 3]

    # 0 = top left sampled, 1 = top right
    # 2 = bottom left, 3 = bottom right
    # -1 = invalid move (for padding)
    movelist = data["trmovelist"]
    turntime = data["trturntime"]
    # Shape: (# Blocks, # Trials/block, 4)
    t_movelist = -1 * np.ones((nS, nTr, 4), int)
    t_turntime = np.zeros((nS, nTr, 4), float) + np.inf
    for i, block in enumerate(movelist):
        assert block.shape[0] == 1 and block[0].shape[0] == 1
        block = block[0][0]
        timeblock = turntime[i][0][0]
        for j, trial in enumerate(block):
            trial = trial[0]
            t_movelist[i][j][: len(trial)] = trial
            t_turntime[i][j][: len(timeblock[j][0])] = timeblock[j][0]
    movelist = t_movelist.reshape(-1, 4)
    turntime = t_turntime.reshape(-1, 4)

    # Location where the second card was made available
    # Location indices same as movelist
    # Shape: (# Blocks, # Trial/block)
    secondcardloc = data["trsecondcard"].reshape(-1)
    card1val = np.zeros(nS * nTr, int)
    card1loc = np.zeros(nS * nTr, int)
    card2val = np.zeros(nS * nTr, int)
    card2loc = np.zeros(nS * nTr, int)
    card3val = np.zeros(nS * nTr, int)
    card3loc = np.zeros(nS * nTr, int)
    card4val = np.zeros(nS * nTr, int)
    card4loc = np.zeros(nS * nTr, int)

    for i in range(nS * nTr):
        moves = movelist[i]
        card1val[i] = cardlist[i][moves[0]]
        card1loc[i] = moves[0]
        card2loc[i] = secondcardloc[i]
        if moves[1] != -1:
            card2val[i] = cardlist[i][moves[1]]
        else:
            card2val[i] = -1
        if moves[2] != -1:
            card3loc[i] = moves[2]
            card3val[i] = cardlist[i][moves[2]]
        else:
            card3loc[i] = -1
            card3val[i] = -1
        if moves[3] != -1:
            card4loc[i] = moves[3]
            card4val[i] = cardlist[i][moves[3]]
        else:
            card4loc[i] = -1
            card4val[i] = -1

    df["card1val"] = card1val
    df["card1loc"] = card1loc
    df["card2val"] = card2val
    df["card2loc"] = card2loc
    df["card3val"] = card3val
    df["card3loc"] = card3loc
    df["card4val"] = card4val
    df["card4loc"] = card4loc

    uid = data["uid"]
    uid = np.repeat(uid.reshape(-1, 1), 22, axis=1)
    df["uid"] = uid.reshape(-1)

    try:
        robbp5 = data["robbp6"][:, 4]
        robbp5 = np.repeat(robbp5.reshape(-1, 1), 22, axis=1)
        df["robbp5"] = robbp5.reshape(-1)

        robbp6 = data["robbp6"][:, 5]
        robbp6 = np.repeat(robbp6.reshape(-1, 1), 22, axis=1)
        df["robbp6"] = robbp6.reshape(-1)
    except KeyError:
        print("Approach Avoid Parameters not found")

    df["success"] = data["trsuccess"].reshape(-1)

    # 0 for top row, 1 for bottom row
    chosen_row = np.zeros_like(df["success"])
    for i in range(nS * nTr):
        trialtype = df["trialtype"][i]
        success = df["success"][i]
        cards = cardlist[i]
        if trialtype == 0:
            correct_row = np.argmax(cards) // 2
            chosen_row[i] = correct_row if success else 1 - correct_row
        elif trialtype == 1:
            correct_row = np.argmin(cards) // 2
            chosen_row[i] = correct_row if success else 1 - correct_row
        elif trialtype == 2:
            correct_row = np.argmax([cards[0:2].sum(), cards[2:4].sum()])
            chosen_row[i] = correct_row if success else 1 - correct_row
        elif trialtype == 3:
            correct_row = np.argmin([cards[0:2].sum(), cards[2:4].sum()])
            chosen_row[i] = correct_row if success else 1 - correct_row
        elif trialtype == 4:
            correct_row = np.argmax([cards[0:2].prod(), cards[2:4].prod()])
            chosen_row[i] = correct_row if success else 1 - correct_row
        elif trialtype == 5:
            correct_row = np.argmin([cards[0:2].prod(), cards[2:4].prod()])
            chosen_row[i] = correct_row if success else 1 - correct_row

    df["chosenRow"] = chosen_row
    df["rowAchosen"] = 1 * (df["chosenRow"] == df["card1loc"] // 2)
    df["AA"] = 1 * (df["card1loc"] // 2 == df["card2loc"] // 2)
    df["numMoves"] = data["trNumMove"].reshape(-1)
    df["score"] = data["trScore"].reshape(-1)
    df["decisiontime"] = data["trdecisiontime"].reshape(-1)

    df["guesstime"] = data["trguesstime"].reshape(-1)
    df["turntime1"] = turntime[:, 0]
    df["turntime2"] = turntime[:, 1]
    df["turntime3"] = turntime[:, 2]
    df["turntime4"] = turntime[:, 3]

    df["age"] = np.repeat(data["age"].reshape(-1, 1), 22, axis=1).reshape(-1)
    df["education"] = np.repeat(data["education"].reshape(-1, 1), 22, axis=1).reshape(
        -1
    )
    df["gender"] = np.repeat(data["gender"].reshape(-1, 1), 22, axis=1).reshape(-1)
    df["location"] = np.repeat(data["location"].reshape(-1, 1), 22, axis=1).reshape(-1)

    df["guess@1"] = df["numMoves"].apply(lambda x: 1 * (x == 1))
    df["guess@2"] = 1 * (df["numMoves"] == 2)
    df["stage2mask"] = 1 * (df["numMoves"] >= 2)
    df["sampleA@2"] = 1 * (df["card1loc"] // 2 == df["card3loc"] // 2)
    df["guess@3"] = 1 * (df["numMoves"] == 3)
    df["stage3mask"] = 1 * (df["numMoves"] >= 3)
    df["stage4mask"] = 1 * (df["numMoves"] >= 4)

    return df
