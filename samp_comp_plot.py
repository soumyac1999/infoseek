import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import sem

df = pd.read_csv("data.txt", sep="\t").groupby("Other_subs").agg([np.mean, sem])

plt.figure(figsize=(9, 3))
plt.subplot(131)
plt.errorbar(x=df.index, y=df["corr(scores)"]["mean"], yerr=df["corr(scores)"]["sem"])
plt.xscale("log")
plt.title("Scores corr.")
plt.xlabel("No of other subjs")
plt.ylabel("corr")

plt.subplot(132)
plt.errorbar(x=df.index, y=df["corr(steps)"]["mean"], yerr=df["corr(steps)"]["sem"])
plt.xscale("log")
plt.title("Steps corr.")
plt.xlabel("No of other subjs")
plt.ylabel("corr")

plt.subplot(133)
plt.errorbar(
    x=df.index, y=df["corr(accuracy)"]["mean"], yerr=df["corr(accuracy)"]["sem"]
)
plt.xscale("log")
plt.title("Accuracy corr.")
plt.xlabel("No of other subjs")
plt.ylabel("corr")


plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.errorbar(
    x=np.arange(len(df.index)),
    y=df["corr(scores)"]["mean"],
    yerr=df["corr(scores)"]["sem"],
)
plt.xticks(np.arange(len(df.index)), df.index)
plt.title("Scores corr.")
plt.xlabel("No of other subjs")
plt.ylabel("corr")

plt.subplot(132)
plt.errorbar(
    x=np.arange(len(df.index)),
    y=df["corr(steps)"]["mean"],
    yerr=df["corr(steps)"]["sem"],
)
plt.xticks(np.arange(len(df.index)), df.index)
plt.title("Steps corr.")
plt.xlabel("No of other subjs")
plt.ylabel("corr")

plt.subplot(133)
plt.errorbar(
    x=np.arange(len(df.index)),
    y=df["corr(accuracy)"]["mean"],
    yerr=df["corr(accuracy)"]["sem"],
)
plt.xticks(np.arange(len(df.index)), df.index)
plt.title("Accuracy corr.")
plt.xlabel("No of other subjs")
plt.ylabel("corr")


plt.tight_layout()
plt.show()
