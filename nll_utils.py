import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import TaskSpecificDataset


def stage_wise_NLL(net, mult_val_df, device):
    """
    Stage wise NLL for comparison to Hunt baseline
    """
    with torch.no_grad():
        CE = nn.CrossEntropyLoss(reduction="none")
        loss1sample = torch.tensor(0.0)
        loss1guess = torch.tensor(0.0)
        loss2sample = torch.tensor(0.0)
        loss2guess = torch.tensor(0.0)
        loss2abSample = torch.tensor(0.0)
        loss3sample = torch.tensor(0.0)
        loss3guess = torch.tensor(0.0)
        loss4sample = torch.tensor(0.0)
        net.eval()
        mult_val = TaskSpecificDataset(mult_val_df, 4)
        for i, batch_data in enumerate(
            tqdm(
                DataLoader(
                    mult_val,
                    batch_size=1,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                ),
                desc="Model NLL Calculation"
            )
        ):
            batch_data = [t.to(device) for t in batch_data]

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

            subjemb = net.sub_emb(uid)
            outputs = net.parallel_nets[0](
                stage1inp, stage2inp, stage3inp, stage4inp, subjemb
            )

            (stage1out, stage2out, stage3out, stage4out, _) = outputs

            aa = stage1inp[:, 1].cpu()

            guess_1 = stage1tgt[:, 0].cpu()
            rowAchosen = stage1tgt[:, 1].cpu()

            guess_2 = stage2tgt[:, 0].cpu()
            sampleA_2 = stage2tgt[:, 2].cpu()

            guess_3 = stage3tgt[:, 0].cpu()
            guess_4 = stage4tgt.cpu()

            stage2mask = stage2mask.cpu()
            sampleA_2 = sampleA_2.cpu()

            # Stage1
            loss1sample += CE(stage1out[0].cpu(), guess_1).mean()
            if int(guess_1) > 0:
                loss1guess += (
                    guess_1 * CE(stage1out[1].cpu(), rowAchosen)
                ).sum() / guess_1.sum()
            # Stage2
            if int(stage2mask) > 0 and aa == 0:
                loss2sample += (
                    stage2mask * CE(stage2out[0].cpu(), guess_2)
                ).sum() / stage2mask.sum()
                if int(guess_2) > 0:
                    loss2guess += (
                        stage2mask * guess_2 * CE(stage2out[1].cpu(), rowAchosen)
                    ).sum() / (stage2mask * guess_2).sum()
                else:
                    loss2abSample += (
                        stage2mask
                        * (1 - guess_2)
                        * (1 - aa)
                        * CE(stage2out[2].cpu(), sampleA_2)
                    ).sum() / (stage2mask * (1 - guess_2) * (1 - aa)).sum()
    i += 1
    print(
        loss1sample.item() / i,
        loss1guess.item() / i,
        loss2sample.item() / i,
        loss2guess.item() / i,
        loss2abSample.item() / i,
        loss3sample.item() / i,
        loss3guess.item() / i,
        loss4sample.item() / i,
    )


def baseline_stage_wise_NLL(valsAA, valsAB, vals2, mult_val_df):
    with torch.no_grad():
        CE = nn.CrossEntropyLoss(reduction="none")
        loss1sample = torch.tensor(0.0)
        loss1guess = torch.tensor(0.0)
        loss2sample = torch.tensor(0.0)
        loss2guess = torch.tensor(0.0)
        loss2abSample = torch.tensor(0.0)
        loss3sample = torch.tensor(0.0)
        loss3guess = torch.tensor(0.0)
        loss4sample = torch.tensor(0.0)

        mult_val = TaskSpecificDataset(mult_val_df, 4)
        for i, batch_data in enumerate(
            tqdm(
                DataLoader(
                    mult_val,
                    batch_size=1,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                ),
                desc = "Baseline NLL Calculation"
            )
        ):
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

            aa = stage1inp[:, 1]

            card1 = int(stage1inp[:, 0] * 4.5 + 5.5) - 1
            card2 = int(stage2inp[:, 0] * 4.5 + 5.5) - 1

            guess_1 = stage1tgt[:, 0]
            rowAchosen = stage1tgt[:, 1]

            guess_2 = stage2tgt[:, 0]
            sampleA_2 = stage2tgt[:, 2]

            if int(aa) == 1:
                vals = valsAA
            else:
                vals = valsAB

            # Stage1
            loss1sample += CE(
                torch.tensor(
                    [[vals[2, card1], vals[0, card1] + vals[1, card1]]],
                    dtype=torch.float,
                ),
                guess_1,
            ).mean()
            if int(guess_1) > 0:
                loss1guess += (
                    guess_1
                    * CE(
                        torch.tensor(
                            [[vals[1, card1], vals[2, card1]]], dtype=torch.float
                        ),
                        rowAchosen,
                    )
                ).sum() / guess_1.sum()
            # Stage2
            if int(stage2mask) > 0 and aa == 0:
                loss2sample += (
                    stage2mask
                    * CE(
                        torch.tensor(
                            [
                                [
                                    vals2[card1, card2, 2:4].sum(),
                                    vals2[card1, card2, 0:2].sum(),
                                ]
                            ],
                            dtype=torch.float,
                        ),
                        guess_2,
                    )
                ).sum() / stage2mask.sum()
                if int(guess_2) > 0:
                    loss2guess += (
                        stage2mask
                        * guess_2
                        * CE(
                            torch.tensor(
                                [[vals2[card1, card2, 1], vals2[card1, card2, 0]]],
                                dtype=torch.float,
                            ),
                            rowAchosen,
                        )
                    ).sum() / (stage2mask * guess_2).sum()
                else:
                    loss2abSample += (
                        stage2mask
                        * (1 - guess_2)
                        * (1 - aa)
                        * CE(
                            torch.tensor(
                                [[vals2[card1, card2, 3], vals2[card1, card2, 2]]],
                                dtype=torch.float,
                            ),
                            sampleA_2,
                        )
                    ).sum() / (stage2mask * (1 - guess_2) * (1 - aa)).sum()

    i += 1
    print(
        loss1sample.item() / i,
        loss1guess.item() / i,
        loss2sample.item() / i,
        loss2guess.item() / i,
        loss2abSample.item() / i,
        loss3sample.item() / i,
        loss3guess.item() / i,
        loss4sample.item() / i,
    )
