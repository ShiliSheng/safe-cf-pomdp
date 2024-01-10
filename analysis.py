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
    static_obstacles = data["Stacic Obstacles"]
    disallowed_actions = data["Disallowed Actions"]

    fig, ax = plt.subplots()   
    ax.set_aspect('equal')             
    plt.xlim(-0.5, 22)
    plt.ylim(-0.5, 22)
    # circle = patches.Circle(state_ground_truth, radius=0.5, edgecolor='black', facecolor='none')
    # ax.add_patch(circle)
    plt.scatter(state_ground_truth[0], state_ground_truth[1], marker = 'o', color = "black")
    width = 1
    height = 1
    for x, y in belief:
        rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, facecolor= "blue", alpha = 0.5)
        ax.add_patch(rect)
        # plt.scatter(x, y, marker = 'H', alpha=0.5, color = "black")
    for x, y in end_states:
        rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, facecolor= "green", alpha = 1)
        ax.add_patch(rect)
        # plt.scatter(x, y, marker = '*', alpha = 1, color = "green")
    for x, y in static_obstacles:
        rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, facecolor= "red", alpha = 1)
        ax.add_patch(rect)
        # plt.scatter(x, y, marker = 's', alpha = 1, color = "red")

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
    
    info = "Step: {}, Action: {}, Disallowed actions: {}".format(cur_time, action, disallowed_actions)
    plt.text(-0.5, -3, info)
    info = "Minimum distance to pedestrains: {}".format(round(cur_min_distance, 2))
    plt.text(-0.5, -4, info)

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

def get_statistics():
    np.set_printoptions(precision=2, suppress=True)
    path = "./results/Obstacle/"
    result = []
    for experiment_setting in os.listdir(path):
        if experiment_setting == '.DS_Store': continue
        file = path + experiment_setting + "/summary.csv"
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

def plot_results():
    plot_figure(path = "./results/ObstacleGridSize-22/shield_1-lookback_4-prediction_3-failure_0.1-2024-01-10-10-02/")

if __name__ == "__main__":
    plot_results()
    # get_statistics()
    pass
