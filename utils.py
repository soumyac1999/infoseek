import numpy as np
import pandas as pd
from tqdm import tqdm


def equal_points_bin(x, n_bins):
    return np.quantile(x, np.arange(n_bins) / n_bins)


def split_df_by_uid(df, frac=0.8):
    grouped = df.groupby("uid")
    train = []
    val = []
    for i in tqdm(np.unique(df["uid"]), desc="Dataset Split"):
        df_i = grouped.get_group(i)
        train_i = df_i.sample(frac=frac)
        val_i = df_i.drop(train_i.index)
        train.append(train_i)
        val.append(val_i)

    train_df = pd.concat(train)
    val_df = pd.concat(val)

    return train_df, val_df
