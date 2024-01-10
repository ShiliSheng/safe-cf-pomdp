import numpy as np
import pandas as pd
import os

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    path = "./results/Obstacle/"
    result = []
    for experiment_setting in os.listdir(path):
        if experiment_setting == '.DS_Store': continue
        file = path + experiment_setting + "/result.csv"
        df = pd.read_csv(file, index_col=0)
        data = {
            "Shield Level":                     df["Shield Level"].iloc[0],
            "Predict Horizon":                  df["Predict Horizon"].iloc[0],
            "Failure Rate":                     df["Failure Rate"].iloc[0],
            "Number of Dynamic Agents":         df["Number of Dynamic Agents"].iloc[0],
            "Minimum Distance to Agents":       df["Minimum Distance to Agents"].mean(),
            'Cumulative Undiscounted Reward':   df['Cumulative Undiscounted Reward'].mean(),
            'Number of Unsafe State':           df['Number of Unsafe State'].mean(),
            'Reached Target':                   df['Reached Target'].mean(),
            'Action Time spent per action in seconds': df['Action Time spent per action in seconds'].mean(),
        }
        
        result.append(pd.DataFrame([data], columns = data.keys()))
    res = pd.concat(result, ignore_index = True).round(2)
    res = res.sort_values(by = ["Shield Level", "Failure Rate"])
    # print(res.values, res.columns)
    print(res)
    res.to_csv("./results/r.csv")