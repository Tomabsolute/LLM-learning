import numpy as np
import pandas as pd

df = pd.DataFrame({
    "model": ["DPO", "DPO", "PPO", "PPO", "DRPO", "DRPO"],
    "epoch": [1, 2, 1, 2, 1, 2],
    "loss": [0.5, 0.3, 0.8, 0.6, 0.4, 0.2],
    "reward": [0.6, 0.8, 0.5, 0.7, 0.7, 0.9]
})

def analyze_training(df):
    result = df.groupby("model").agg(mean_loss = ("loss","mean"),mean_reward = ("reward","mean"))
    result["score"] = result["mean_reward"] - result["mean_loss"]
    result["performance"] = np.where(result.score >= 0.3,"Good","Normal")
    result = result.sort_values(by = "score",ascending = False)
    result = result.reset_index()
    return result


print(analyze_training(df))