import numpy as np
import torch
from tqdm import tqdm

from cardenv import CardEnv


def simulate_trial(sim_net, task_id, env, uid, aa=None, cards=None):
    sim_net.end_trial()
    if cards.any() != None:
        obs = env.reset_with_cards(aa, cards)
    else:
        obs = env.reset()

    ret = 0

    # Stage1
    aa = obs[1]
    guess, guessrowA = sim_net(
        task_id,
        (
            torch.tensor([(obs[0] - 5.5) / 4.5, obs[1]], dtype=torch.float),
            torch.tensor(uid, dtype=torch.long),
        ),
        1,
    )

    if guess[0] > guess[1]:
        action = 0
    elif guessrowA[1] > guessrowA[0]:
        action = 1
    else:
        action = 2

    card, rew, done, info = env.step(action)
    ret += rew
    if done:
        return ret, 1

    # Stage2
    guess, guessrowA, sampleA2 = sim_net(
        task_id, torch.tensor([(card - 5.5) / 4.5], dtype=torch.float), 2
    )

    if guess[0] > guess[1]:
        # this means sample
        if aa == 1:
            action = 0
        elif sampleA2[1] > sampleA2[0]:
            action = 0
        else:
            action = 1
    else:
        if aa == 1:
            if guessrowA[1] > guessrowA[0]:
                action = 1
            else:
                action = 2
        else:
            if guessrowA[1] > guessrowA[0]:
                action = 2
            else:
                action = 3

    card, rew, done, info = env.step(action)
    ret += rew
    if done:
        return ret, 2

    # Stage3
    guess, guessrowA = sim_net(
        task_id, torch.tensor([(card - 5.5) / 4.5], dtype=torch.float), 3
    )

    if guess[0] > guess[1]:
        action = 0
    elif guessrowA[1] > guessrowA[0]:
        action = 1
    else:
        action = 2

    card, rew, done, info = env.step(action)
    ret += rew
    if done:
        return ret, 3

    # Stage4
    guessrowA = sim_net(
        task_id, torch.tensor([(card - 5.5) / 4.5], dtype=torch.float), 4
    )[0]

    if guessrowA[1] > guessrowA[0]:
        action = 0
    else:
        action = 1

    card, rew, done, info = env.step(action)
    ret += rew
    return ret, 4


def simulate(sim_net, df_for_sim, tasks):
    envs = [CardEnv(t) for t in tasks]

    sim_net = sim_net.cpu()
    scores = np.zeros(len(np.unique(df_for_sim["uid"])))
    steps = np.zeros(len(np.unique(df_for_sim["uid"])))
    success = np.zeros(len(np.unique(df_for_sim["uid"])))
    for i, uid in enumerate(tqdm(np.unique(df_for_sim["uid"]), desc="Simulation")):
        score = 0
        step = 0
        success_ = 0
        for l, d in enumerate(
            np.array(
                df_for_sim[df_for_sim["uid"] == uid][
                    ["trialtype", "AA", "card1", "card2", "card3", "card4"]
                ]
            )
        ):
            trtype, aa, *cards = d
            task_id = tasks.index(trtype)
            score_, step_ = simulate_trial(
                sim_net, task_id, envs[task_id], uid, aa, np.array(cards)
            )
            score += score_
            step += step_
            success_ += 1 if score_ > 0 else 0
        scores[i] = score / (l + 1)
        steps[i] = step / (l + 1)
        success[i] = success_ / (l + 1)

    avg_score = df_for_sim.groupby("uid")["score"].mean()
    avg_steps = df_for_sim.groupby("uid")["numMoves"].mean()
    avg_success = df_for_sim.groupby("uid")["success"].mean()
    return scores, steps, success, avg_score, avg_steps, avg_success
