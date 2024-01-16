from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import ast
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
from PIL import Image

def plot_gif(figure_path):
    image_files = sorted([f for f in os.listdir(figure_path) if f.endswith('.jpg')])
    images = [Image.open(os.path.join(figure_path, f)) for f in image_files]
    gif_path = figure_path.replace("figures", "gifs")
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    imageio.mimsave(gif_path + 'output.gif', images, duration=0.5)  

def plot_figure_from_data(data, figure_path):
    state_ground_truth = data["Robot State"]
    Y_cur_agents = data["Dynamcic Agents"]
    H = data["Predict Horizon"]
    cur_time = data["Current Time Step"]
    cur_time = data["Action Step"]
    i_episode = data["Episode"]
    cur_min_distance = data["Current Minimum Distance to Agents"]
    action = data["Selected Action"]
    shieldLevel = data["Shield Level"]
    estimated_regions = data["Estimated Regions"]
    estimation_moving_agents = data["Dynamic Agents Prediction"]
    belief = data["Belief States"]
    end_states  = data["End States"]
    targets  = data["Target States"]
    obstacles = data["Stacic Obstacles"]
    disallowed_actions = data["Disallowed Actions"]
    minX = data.get("minX", 0)
    minY = data.get("minY", 0)
    maxX = data.get("maxX", 22)
    maxY = data.get("maxY", 22)
    state_size = 100
    if maxY < 30:
        state_size = None
    elif maxY < 80:
        state_size = 5
    else:
        state_size = 2
    fig, ax = plt.subplots()   
    ax.set_aspect('equal')             
    plt.xlim(minX-0.5, maxX+0.5)
    plt.ylim(minY-0.5, maxY+0.5)
    # circle = patches.Circle(state_ground_truth, radius=, edgecolor='black', facecolor='none')
    # ax.add_patch(circle)
    plt.scatter(state_ground_truth[0], state_ground_truth[1], marker = 'o', color = "black", s = state_size)

    width = 1
    height = 1
    for x, y in targets:
        rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, facecolor= "green", alpha = 0.9)
        ax.add_patch(rect)
        # plt.scatter(x, y, marker = '*', alpha = 1, color = "green")
    for x, y in obstacles:
        rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, facecolor= "red", alpha = 0.9)
        ax.add_patch(rect)
        # plt.scatter(x, y, marker = 's', alpha = 1, color = "red")

    for x, y in belief:
        rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, facecolor= "blue", alpha = 0.5)
        ax.add_patch(rect)
        # plt.scatter(x, y, marker = 'H', alpha=0.5, color = "black")

    cmap = plt.get_cmap('tab10')
    for i in range(0, len(Y_cur_agents), 2):
        plt.scatter(Y_cur_agents[i], Y_cur_agents[i+1], marker = '.', color = cmap(i//2), alpha=1)

    for tau in range(1, H + 1):
        for j in range(0, len(estimation_moving_agents[tau]), 2):
            x, y = estimation_moving_agents[tau][j], estimation_moving_agents[tau][j + 1]
            r = estimated_regions[tau]
            a = (0.5 - 0.2) / (1 - H)
            b = (0.5 * H - 0.2) / (H - 1)
            plt.scatter(x, y, color = cmap(j//2), marker = '.', alpha = a * tau + b)
            circle = patches.Circle((x,y), radius = r, edgecolor=cmap(i//2), facecolor=cmap(j//2), alpha =  a * tau + b)
            ax.add_patch(circle)
    # plt.title("Time: {}, Action: {}, cur_min_distance:{}".format(cur_time, action, str(cur_min_distance)[:3]))
    
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    
    if maxY < 30:
        info = "Step: {}, Action: {}, Disallowed actions: {}".format(cur_time, action, disallowed_actions)
        plt.text(-0.5, -3, info)
        info = "Minimum distance to pedestrains: {}".format(round(cur_min_distance, 2))
        plt.text(-0.5, -4, info)
    elif maxY < 70:
        info = "Step: {}, Action: {}, Disallowed actions: {}".format(cur_time, action, disallowed_actions)
        plt.text(-0.5, -7, info)
        info = "Minimum distance to pedestrains: {}".format(round(cur_min_distance, 2))
        plt.text(-0.5, -11, info)


    if figure_path[-1] != '/':
        figure_path += '/'
    plt.savefig(figure_path+ "figure_{}.jpg".format(str(cur_time).zfill(3)), dpi=300 , bbox_inches="tight")
    plt.close()

def plot_figure(path):
    for record in os.listdir(path):
        if "Episode" not in record or "pkl" not in record: continue
        with open(path + record, 'rb') as file:
            data = pickle.load(file)
        figure_path = path.replace("results", "figures") + "Episode-{}/".format(data[0]["Episode"])
        for index, row in enumerate(data):
            if index + 1 < len(data) and data[index+1]["Action Step"] == 0: continue
            plot_figure_from_data(row, figure_path)
        plot_gif(figure_path)

def get_statistics(path):
    np.set_printoptions(precision=2, suppress=True)
    
    result = []
    record_min_distance = []
    record_reward = []
    for experiment_setting in os.listdir(path):
        print(experiment_setting)
        # if not os.path.isdir(path + experiment_setting ): continue
        if not os.path.exists(path + experiment_setting + '/summary.csv'): continue
        file = path + experiment_setting + "/summary.csv"
        df = pd.read_csv(file, index_col=0)
        if len(df) < 10: continue
        needed = ['shield_0-lookback_4-prediction_3-failure-0.2-agents-5-2024-01-13-12-54', 
        'shield_1-lookback_4-prediction_3-failure-0.05-agents-5-2024-01-13-22-09',
        'shield_1-lookback_4-prediction_3-failure-0.2-agents-5-2024-01-13-18-16',]
        # if experiment_setting not in needed: continue
        data = {
            "Setting":                          experiment_setting,
            "Episodes":                         len(df),
            "Shield Level":                     df["Shield Level"].iloc[0],
            "Predict Horizon":                  df["Predict Horizon"].iloc[0],
            "Failure Rate":                     df["Failure Rate"].iloc[0],
            "Number of Dynamic Agents":         df["Number of Dynamic Agents"].iloc[0],
            "Average Minimum Distance to Agents":       df["Minimum Distance to Agents"].mean(),
            "Min of Minimum Distance to Agents":       df["Minimum Distance to Agents"].min(),
            'Cumulative Undiscounted Reward':   df['Cumulative Undiscounted Reward'].mean(),
            "Number of Unsafe Action":          df["Number of Unsafe Action"].mean(),
            'Number of Collision with Static Obstalces':           df['Number of Unsafe State'].mean(),
            'Number of Unsafe Distance to agents':          sum(df["Minimum Distance to Agents"] < 0.5),
            'Reached Target':                   df['Reached Target'].mean(),
            'Action Time spent per action in seconds': df['Action Time spent per action in seconds'].mean(),
        }
        setting = "{}-{}-{}".format(data["Shield Level"], data["Predict Horizon"], data["Failure Rate"])
        min_distance = pd.DataFrame()
        min_distance[setting] = df["Minimum Distance to Agents"]
        record_min_distance.append(min_distance)

        reward = pd.DataFrame()
        reward[setting] = df["Cumulative Undiscounted Reward"]
        record_reward.append(reward)

        result.append(pd.DataFrame([data], columns = data.keys()))

    res = pd.concat(result, ignore_index = True).round(2)
    res = res.sort_values(by = ["Shield Level", "Failure Rate"])
    # print(res.values, res.columns)
    res.to_csv( path + "stat.csv")

    # for record, name in [(record_min_distance, "Min. Distance"), (record_reward, "Reward")]:
    #     df = pd.concat(record, axis= 1)
    #     df = df.reindex(sorted(df.columns), axis=1)
    #     df.boxplot()
    #     plt.grid(False)
    #     plt.xticks(rotation=30)
    #     plt.savefig(path + name + ".png", dpi = 300, bbox_inches= "tight")
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




def plot_results():
    plot_figure(path = "./results/ObstacleGridSize-22/shield_1-lookback_4-prediction_3-failure_0.1-2024-01-10-10-02/")



if __name__ == "__main__":
    

    # plot_results()

    # path = './results/Obstacle-SDD-bookstore-video1-0-0-60-60/shield_1-lookback_4-prediction_5-failure-0.1-agents-10-2024-01-14-11-47/'
    # plot_figure(path)

    path = '/results/Obstacle-ETH-0-0-22-22.'
    get_statistics(path = "./results/Obstacle-ETH-0-0-22-22/")
    pass
