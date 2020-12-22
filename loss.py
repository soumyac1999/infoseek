import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.CE = nn.CrossEntropyLoss(reduction="none")

    def forward(self, batch_data, outputs):
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

        (stage1out, stage2out, stage3out, stage4out, _) = outputs

        aa = stage1inp[:, 1]

        guess_1 = stage1tgt[:, 0]
        rowAchosen = stage1tgt[:, 1]

        guess_2 = stage2tgt[:, 0]
        sampleA_2 = stage2tgt[:, 2]

        guess_3 = stage3tgt[:, 0]
        guess_4 = stage4tgt

        # Stage1
        loss = self.CE(stage1out[0], guess_1).mean()
        if guess_1.sum() > 0:
            loss += (guess_1 * self.CE(stage1out[1], rowAchosen)).sum() / guess_1.sum()
        # Stage2
        if stage2mask.sum() > 0:
            loss += (
                stage2mask * self.CE(stage2out[0], guess_2)
            ).sum() / stage2mask.sum()
        else:
            return loss
        if (stage2mask * guess_2).sum() > 0:
            loss += (stage2mask * guess_2 * self.CE(stage2out[1], rowAchosen)).sum() / (
                stage2mask * guess_2
            ).sum()
        if (stage2mask * (1 - guess_2) * (1 - aa)).sum() > 0:
            loss += (
                stage2mask * (1 - guess_2) * (1 - aa) * self.CE(stage2out[2], sampleA_2)
            ).sum() / (stage2mask * (1 - guess_2) * (1 - aa)).sum()
        # Stage3
        if (stage3mask).sum() > 0:
            loss += (
                stage3mask * self.CE(stage3out[0], guess_3)
            ).sum() / stage3mask.sum()
        else:
            return loss
        if (stage3mask * guess_3).sum() > 0:
            loss += (stage3mask * guess_3 * self.CE(stage3out[1], rowAchosen)).sum() / (
                stage3mask * guess_3
            ).sum()
        # Stage4
        if stage4mask.sum() > 0:
            loss += (
                stage4mask * self.CE(stage4out[0], guess_4)
            ).sum() / stage4mask.sum()

        return loss
