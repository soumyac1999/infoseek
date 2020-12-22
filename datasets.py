import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TaskSpecificDataset(Dataset):
    def __init__(self, df, task_id):
        super(TaskSpecificDataset, self).__init__()
        self.df = pd.DataFrame(df[df["trialtype"] == task_id])
        self.indices = self.df.index
        self.len = len(self.indices)

        self.card1val = np.array(self.df["card1val"])
        self.card2val = np.array(self.df["card2val"])
        self.card3val = np.array(self.df["card3val"])
        self.card4val = np.array(self.df["card4val"])
        self.aa = np.array(self.df["AA"])
        self.stage2mask = np.array(self.df["stage2mask"])
        self.stage3mask = np.array(self.df["stage3mask"])
        self.stage4mask = np.array(self.df["stage4mask"])
        self.rowAchosen = np.array(self.df["rowAchosen"])
        self.guess1 = np.array(self.df["guess@1"])
        self.guess2 = np.array(self.df["guess@2"])
        self.guess3 = np.array(self.df["guess@3"])
        self.sampleA2 = np.array(self.df["sampleA@2"])
        self.uid = np.array(self.df["uid"])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise StopIteration

        stage1inp = [(self.card1val[idx] - 5.5) / 4.5, self.aa[idx]]
        stage1tgt = (self.guess1[idx], self.rowAchosen[idx])

        stage2mask = self.stage2mask[idx]
        stage2inp = [(self.card2val[idx] - 5.5) / 4.5]
        stage2tgt = (self.guess2[idx], self.rowAchosen[idx], self.sampleA2[idx])

        stage3mask = self.stage3mask[idx]
        stage3inp = [(self.card3val[idx] - 5.5) / 4.5]
        stage3tgt = (self.guess3[idx], self.rowAchosen[idx])

        stage4mask = self.stage4mask[idx]
        stage4inp = [(self.card4val[idx] - 5.5) / 4.5]
        stage4tgt = self.rowAchosen[idx]

        uid = self.uid[idx]

        return (
            torch.tensor(stage1inp, dtype=torch.float),
            torch.tensor(stage1tgt, dtype=torch.long),
            torch.tensor(stage2mask, dtype=torch.float),
            torch.tensor(stage2inp, dtype=torch.float),
            torch.tensor(stage2tgt, dtype=torch.long),
            torch.tensor(stage3mask, dtype=torch.float),
            torch.tensor(stage3inp, dtype=torch.float),
            torch.tensor(stage3tgt, dtype=torch.long),
            torch.tensor(stage4mask, dtype=torch.float),
            torch.tensor(stage4inp, dtype=torch.float),
            torch.tensor(stage4tgt, dtype=torch.long),
            torch.tensor(uid, dtype=torch.long),
        )


class MultiTaskDataset(Dataset):
    def __init__(self, df, task_ids):
        super(MultiTaskDataset, self).__init__()
        self.taskSpecificDatasets = [
            TaskSpecificDataset(df, task_id) for task_id in task_ids
        ]
        self.task_lens = [len(d) for d in self.taskSpecificDatasets]
        self.len = max(self.task_lens)
        self.df = pd.concat(
            [d.df for d in self.taskSpecificDatasets], ignore_index=True
        )

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if idx >= self.len:
            raise StopIteration

        ret = [
            ds[idx % self.task_lens[i]]
            for i, ds in enumerate(self.taskSpecificDatasets)
        ]
        return ret
