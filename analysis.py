from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def get_statistics(path):
    np.set_printoptions(precision=2, suppress=True)
    result = []
    record_min_distance = []
    record_reward = []
    N_agents = 100
    for experiment_setting in os.listdir(path):
        # if not os.path.isdir(path + experiment_setting ): continue
        if not os.path.exists(path + experiment_setting + '/summary.csv'): continue
        print(experiment_setting)
        file = path + experiment_setting + "/summary.csv"
        day_time = ''.join(experiment_setting.split("-")[8:])
        df = pd.read_csv(file, index_col=0)
        n_episodes = len(df)
        
        if N_agents !=100 and N_agents != df["Number of Dynamic Agents"].iloc[0] :
            continue

        # for i_episode in range(n_episodes):
        #     episode_file = os.path.join(path, experiment_setting, "Episode-{}.pkl".format(i_episode))
        #     with open(episode_file, 'rb') as episode_file_pkl:
        #         episode_data = pickle.load(episode_file_pkl)
        #     index = 0
        #     cnt_action_taken = 0
        #     cumulative_time = 0
        #     max_plan_time = 0
        #     cnt_close_to_agents = 0
        #     while index+1 < len(episode_data):
        #         if episode_data[index+1]["Action Step"] > 1:
        #             plan_time = episode_data[index+1]["Clock Time"] - episode_data[index]["Clock Time"]
        #             cumulative_time += plan_time
        #             cnt_action_taken += 1
        #             max_plan_time = max(max_plan_time, plan_time)
        #             if episode_data[index + 1]["Current Minimum Distance to Agents"] < episode_data[index + 1]["Safe Distance"]:
        #                 cnt_close_to_agents += 1
        #         index+=1
        #     df.loc[i_episode, "max_plan_time"] = max_plan_time
        #     df.loc[i_episode, "average_plan_time"] = cumulative_time / cnt_action_taken
        #     df.loc[i_episode, "Number of being unsafe distance to agents"] = cnt_close_to_agents

        # for i in range(len(df)):
        #     df.loc[i, 'Trip safe with pedetrains'] = 1 if df.loc[i,"Minimum Distance to Agents"] >= df.loc[i,"Safe Distance"] else 0
        #     df.loc[i,'Trip reached and safe with pedestrians'] = 1 if df.loc[i, 'Reached Target'] and df.loc[i, 'Trip safe with pedetrains'] else 0
        #     df.loc[i, 'Trip safe with obstacles'] = 1 if df.loc[i,"Number of Unsafe State"]  == 0 else 0
        #     df.loc[i,'Trip reached and safe with obstacles'] = 1 if df.loc[i, 'Reached Target'] and df.loc[i, 'Trip safe with obstacles'] else 0
        #     df.loc[i, "Trip reached and safe with both"] = 1 if df.loc[i,'Trip reached and safe with pedestrians'] and df.loc[i,'Trip reached and safe with obstacles'] else 0
        
        data = {
            "Shield Level":                                 df["Shield Level"].iloc[0],
            "Predict Horizon":                              df["Predict Horizon"].iloc[0] if df["Shield Level"].iloc[0] ==1 else "-",
            "Failure Rate":                                 df["Failure Rate"].iloc[0] if df["Shield Level"].iloc[0] ==1 else "-",
            "Number of Dynamic Agents":                     df["Number of Dynamic Agents"].iloc[0],
            "Percentage of time being safe to dynamic agents":          df["Percentage of time being safe to dynamic agents"].mean(),
            'Rate of Reaching Target':                                  df['Reached Target'].mean(),
            "Mean of Minimum Distance to Agents":            df["Minimum Distance to Agents"].mean(),
            "SD of Minimum Distance to Agents":            df["Minimum Distance to Agents"].std(),
            "Min of Minimum Distance to Agents":            df["Minimum Distance to Agents"].min(),
            'Cumulative Undiscounted Reward':               df['Cumulative Undiscounted Reward'].mean(),
            "Number of Unsafe Action":                      df["Number of Unsafe Action"].mean(),
            "Number of Action Steps":                       df["Number of Action Steps"].mean(),
            'Average Number of times being unsafe to static obstacles': df["Number of times being unsafe to static obstacles"].mean(),
            'Average Number of times being unsafe to dynamic agents':   df["Number of times being unsafe to dynamic agents"].mean(),
            "Percentage of time being safe to static agents":           df["Percentage of time being safe to static agents"].mean(),
            'Action Time spent per action in seconds':                  df['Action Time spent per action in seconds'].mean(),
            "Setting":                                      experiment_setting,
            "Episodes":                                     len(df),
            # 'Average number of Collision with Static Obstalces':    df['Number of Unsafe State'].mean(),
            # 'Average number of Unsafe Distance to agents':          df["Number of being unsafe distance to agents"].mean(),
            # 'Rate of Reach and safe with pedestrains':              df['Trip reached and safe with pedestrians'].mean(),
            # 'Rate of Reach and safe with obstacles':                df['Trip reached and safe with obstacles'].mean(),
            # 'Rate of Reach and safe with both':                     df["Trip reached and safe with both"].mean(),
            # 'Rate safe with pedestrains given reached':             (df['Trip reached and safe with pedestrians'].sum()) / (df['Reached Target'].sum()),
            # 'Rate safe with obstacles given reached':               (df['Trip reached and safe with obstacles'].sum()) / (df['Reached Target'].sum()),
            # 'Rate safe with both given reached':                    (df['Trip reached and safe with both'].sum()) / (df['Reached Target'].sum()),
            # 'Average of average plan time':     df['average_plan_time'].mean(),
            # 'Max of max plan time':             df['max_plan_time'].max(),
        }
        if data["Shield Level"] != 0:
            setting = "S{}-H{}-a{}".format(data["Shield Level"], data["Predict Horizon"], data["Failure Rate"])
        else:
            setting = 'S0'

        min_distance = pd.DataFrame()
        min_distance[setting] = df["Minimum Distance to Agents"]
        record_min_distance.append(min_distance)

        reward = pd.DataFrame()
        reward[setting] = df["Cumulative Undiscounted Reward"]
        record_reward.append(reward)

        result.append(pd.DataFrame([data], columns = data.keys()))
        print(df["Percentage of time being safe to static agents"].mean(), data["Percentage of time being safe to dynamic agents"])

    res = pd.concat(result, ignore_index = True)
    res = res.sort_values(by = ["Shield Level", "Failure Rate", "Number of Dynamic Agents"])
    # print(res.values, res.columns)
    res.to_csv( path + "stat2.csv", index = False)
    
    # for record, name in [(record_min_distance, "Minimum Distance to Pedestrians"), (record_reward, "Reward")]:
    #     df = pd.concat(record, axis= 1)
    #     df = df.reindex(sorted(df.columns), axis=1)
    #     df.boxplot()
    #     plt.grid(False)
    #     plt.xticks(rotation=30)
    #     plt.ylabel(name)
    #     plt.title("Number of pedestrians {}".format(N_agents))
    #     plt.axhline(y=0.5, color='r', linestyle='--', label='Safety Distance')
    #     plt.savefig(path + name + "_N{}.png".format(N_agents), dpi = 300, bbox_inches= "tight")
    #     plt.show()

    #     f_statistic, p_value = f_oneway(*[df[col] for col in df.columns])
    #     print(f'F-statistic: {f_statistic}\nP-value: {p_value}')

    #     # Check the p-value to determine if there is a significant difference
    #     alpha = 0.05
    #     if p_value < alpha:
    #         print("Reject the null hypothesis: There is a significant difference between groups.")
    #     else:
    #         print("Fail to reject the null hypothesis: There is no significant difference between groups.")
    #     # Tukey's HSD post hoc test
    #     melted_df = pd.melt(df)
    #     posthoc = pairwise_tukeyhsd(melted_df['value'], melted_df['variable'], alpha=0.05)

    #     # Print the results of the post hoc test
    #     print(posthoc)

if __name__ == "__main__":
    # plot_results()
    # path = './results/Obstacle-SDD-bookstore-video1-0-0-60-60/shield_1-lookback_4-prediction_5-failure-0.1-agents-10-2024-01-14-11-47/'
    # plot_figure(path)
    path = './results/Obstacle-ETH-0-0-22-22/'
    get_statistics(path = "./results/Obstacle-ETH-0-0-22-22/")
    get_statistics(path = "./results/Obstacle-SDD-deathCircle-video1-0-0-50-60/")
    get_statistics(path = "./results/Obstacle-SDD-bookstore-video1-0-0-50-45/")
    # df = pd.read_table(path + "stat.csv", sep = ",")
    # print(df.columns)
    # # # Assuming df is your DataFrame
    # # df = df.loc[df["Shield Level"] == 1, :]
    # X = df[["Shield Level", 'Failure Rate', 'Predict Horizon', 'Number of Dynamic Agents']]
    
    # cols = [
    #    'Average Minimum Distance to Agents',
    #    'Min of Minimum Distance to Agents', 'Cumulative Undiscounted Reward',
    #    'Number of Unsafe Action',
    #    'Average number of Collision with Static Obstalces',
    #    'Average number of Unsafe Distance to agents',
    #    'Rate of Reaching Target', 'Rate of Reach and safe with pedestrains',
    #    'Rate of Reach and safe with obstacles',
    #    'Rate of Reach and safe with both',
    #    'Rate safe with pedestrains given reached',
    #    'Rate safe with obstacles given reached',
    #    'Rate safe with both given reached',
    #    'Action Time spent per action in seconds',
    #    'Average of average plan time', 'Max of max plan time']
    # for col in cols:
    #     print(col,"---")
    #     y = df[col]
        
    #     # Add a constant term to the independent variables
    #     X = sm.add_constant(X)

    #     # Fit the logistic regression model
    #     try:
    #         model = sm.Logit(y, X)
    #         result = model.fit()
    #         # Display the summary of the regression
    #         print("+++++++++++++++++++++++++++++++++++++++")
    #         print(col)
    #         print(result.summary())
    #     except:
    #         continue
    # pass