from MDP_TG.mdp import Motion_MDP_label, Motion_MDP, compute_accept_states
from MDP_TG.dra import Dra, Dfa, Product_Dra, Product_Dfa
from MDP_TG.vi4wr import syn_plan_prefix, syn_plan_prefix_dfa
from networkx.classes.digraph import DiGraph
import pickle
import numpy as np
import time
import math
import copy
import random
from model import create_scenario_obstacle
from model import Model
from predictor import Predictor
from pomcp import POMCP
from collections import defaultdict
import pandas as pd
from itertools import chain
from sortedcontainers import SortedList
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import imageio
import os
import copy
from datetime import datetime

def get_min_distance(state_ground_truth, Y_cur):
    cx, cy = state_ground_truth
    minD = float("inf")
    for j in range(len(Y_cur) // 2):
        px, py = Y_cur[2 * j], Y_cur[2 * j + 1]
        t = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        minD = min(minD, t)
    return minD

def plot_gif(figure_path):
    image_files = sorted([f for f in os.listdir(figure_path) if f.endswith('.jpg')])
    images = [Image.open(os.path.join(figure_path, f)) for f in image_files]
    gif_path = figure_path.replace("figures", "gifs")
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    imageio.mimsave(gif_path + 'output.gif', images, duration=0.5)  

def plot_figure(state_ground_truth, pomcp, Y_cur, estimation_moving_agents,  H, log_time, i_episode, cur_time, minD, textinfo):
    shieldLevel = pomcp.shieldLevel
    fig, ax = plt.subplots()   
    ax.set_aspect('equal')             
    plt.xlim(-0.5, 22)
    plt.ylim(-0.5, 22)
    # circle = patches.Circle(state_ground_truth, radius=0.5, edgecolor='black', facecolor='none')
    # ax.add_patch(circle)
    plt.scatter(state_ground_truth[0], state_ground_truth[1], marker = 'o', color = "black")
    width = 1
    height = 1
    for x, y in pomcp.root.belief:
        rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, facecolor= "blue", alpha = 0.5)
        ax.add_patch(rect)
        # plt.scatter(x, y, marker = 'H', alpha=0.5, color = "black")
    for x, y in pomcp.pomdp.end_states:
        rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, facecolor= "green", alpha = 1)
        ax.add_patch(rect)
        # plt.scatter(x, y, marker = '*', alpha = 1, color = "green")
    for x, y in pomcp.pomdp.obstacles:
        rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, facecolor= "red", alpha = 1)
        ax.add_patch(rect)
        # plt.scatter(x, y, marker = 's', alpha = 1, color = "red")

    cmap = plt.get_cmap('tab10')
    for i in range(0, len(Y_cur), 2):
        plt.scatter(Y_cur[i], Y_cur[i+1], marker = '.', color = cmap(i//2), alpha=1)

    for tau in range(1, H + 1):
        for j in range(0, len(estimation_moving_agents[cur_time][tau]), 2):
            x, y = estimation_moving_agents[cur_time][tau][j], estimation_moving_agents[cur_time][tau][j + 1]
            r = constraints[cur_time + 1][tau]
            a = (0.5 - 0.2) / (1 - H)
            b = (0.5 * H - 0.2) / (H - 1)
            plt.scatter(x, y, color = cmap(j//2), marker = '.', alpha = a * tau + b)
            circle = patches.Circle((x,y), radius = r, edgecolor=cmap(i//2), facecolor=cmap(j//2), alpha =  a * tau + b)
            ax.add_patch(circle)
    plt.title("Time: {}, Action: {}, MinDistance:{}".format(cur_time, pomcp.pomdp.actions[actionIndex], str(minD)[:3]))
    figure_path = "./figures/{}-ShieldLevel-{}/Episode_{}/".format(log_time, shieldLevel, i_episode)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    plt.text(0, 6, textinfo)
    plt.savefig(figure_path + "figure_{}.jpg".format(str(cur_time).zfill(3)), dpi=300)
    plt.close()

if __name__ == "__main__":
    log_time = f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    max_steps = 100
    explore_constant = 1000
    shieldLevel = 1

    #Settings for LSTM trajectory prediction

    history_length, prediction_length, model_file = 4, 3, 'model_weights-4-3_2024-01-02_11-50.pth'
    #  = + 'model_weights.pth'
    trained_model_path = './OpenTraj/datasets/ETH/seq_eth/' + model_file

    prediction_model = Predictor(history_length, prediction_length)
    prediction_model.load_model(trained_model_path)
    history_length = prediction_model.history_length
    H = prediction_length = prediction_model.prediction_length

    # POMDP
    pomdp = create_scenario_obstacle()

    pomcp = POMCP(pomdp, shieldLevel, prediction_model.prediction_length, explore_constant)

    motion_mdp, AccStates = pomcp.pomdp.compute_accepting_states() 
    # pomcp.pomdp.set_states_observations(motion_mdp)
    observation_successor_map = pomcp.pomdp.compute_H_step_space(H)
    step = 0
    num_episodes = 1

    # Settings for conformal prediction
    path = "./OpenTraj/datasets/ETH/seq_eth/"
    test_dataset =  pd.read_csv(path + "test_dynammic_agents_0.csv", index_col = 0)
    starting_col_index = 0 # in the dataframe, where is the starting col of values
    num_agents_tracked = 3
    
    target_failure_prob_delta = 0.1
    acp_learing_gamma = 0.2
    safeDistance = 0.5

    for i_episode in range(num_episodes):
        dynamic_agents = prediction_model.create_online_dataset(test_dataset, num_agents_tracked)
        pomcp.reset_root()
        state_ground_truth = pomcp.root.sample_state_from_belief()
        print(state_ground_truth, "current state")
        cur_time = -1
        discounted_reward = 0
        undiscounted_reward = 0

        error = [[0] * (H+1) for _ in range(max_steps + 10)]
        constraints = [[0] * (H+1) for _ in range(max_steps + 10)]    
        estimation_moving_agents = [[0 for _ in range(num_agents_tracked * 2)] * (H+1) for _ in range(max_steps + 10)] 
        # cf_scores = [[0] * (H+1) for _ in range(max_steps + 10)]
        # cf_scores = defaultdict(lambda: SortedList([float('inf')]))
        cf_scores = defaultdict(SortedList)
        failure_prob_delta = [[0] * (H+1) for _ in range(max_steps + 10)]
        for tau in range(1, H + 1):
            failure_prob_delta[0][tau] = target_failure_prob_delta

        done = False
        while not done and cur_time < max_steps:
            done = state_ground_truth in pomcp.pomdp.end_states
            cur_time += 1

            # Line 3, 4 TODO preprocessing
            estimation = prediction_model.predict(dynamic_agents, cur_time, starting_col_index )
            for i, row in enumerate(estimation):
                estimation_moving_agents[cur_time][i+1] = row

            if cur_time < H: # assuming the agent is not starting until Timestamp H
                continue

            Y_cur = dynamic_agents.loc[cur_time,:].values[starting_col_index:] # Check index

            minD = get_min_distance(state_ground_truth, Y_cur)

            for tau in range(1, H + 1):
                # Line 7
                Y_est = estimation_moving_agents[cur_time-tau][tau]
                estimation_error = prediction_model.compute_prediction_error(Y_cur, Y_est)# TODO check formula of Line 7
                # cf_scores[cur_time][tau] = estimation_error
                cf_scores[tau].add(estimation_error)
                if (estimation_error > 1):
                    print("________",cur_time, tau, cur_time-tau)
                    print(Y_cur)
                    print(Y_est)
                # print("estimation_error", estimation_error)
                # Line 6
                error[cur_time][tau] = 0 if estimation_error <= constraints[cur_time][tau] else 1 # a
                failure_prob_delta[cur_time + 1][tau] = failure_prob_delta[cur_time][tau] +  acp_learing_gamma * (target_failure_prob_delta * error[cur_time][tau])

                # Line 8
                # N = (cur_time + 1 - tau)
                # N = len(cf_scores[tau]) - 1 # TODO not include inf, simple add N = len() needs to be double-checked
                # q = math.ceil(N * (1 - failure_prob_delta[cur_time+1][tau]))
                # q_index = q - 1

                N = len(cf_scores[tau])
                q = math.ceil((N+1) * (1 - failure_prob_delta[cur_time+1][tau]))

                # q_level = q / N
                if q > N:
                    radius = 1 + 0.1 * tau
                else:
                    radius = cf_scores[tau][q - 1] # 0-indexed
                # Line 9
                # values = [cf_scores[k][tau] for k in range(tau, cur_time+1)] + [float("inf")]
                # values.sort()
                # radius = values[q-1]
                print("____++++", radius, q, "N", N, "qlevel",len(cf_scores[tau]),  failure_prob_delta[cur_time + 1][tau], error[cur_time][tau], cf_scores[tau] )

                constraints[cur_time + 1][tau] = radius
                # print("radius", radius, cf_scores[tau], q-1, N)
            # print(constraints[cur_time + 1][tau], "_______________")
            if q > N: continue
             
            ACP_step = defaultdict(list)
            ACP_step = pomdp.build_restrictive_region(estimation_moving_agents[cur_time], constraints[cur_time+1], H, safeDistance)

            # state_observation_map_copy = copy.deepcopy(pomcp.pomdp.state_observation_map)    # save state_map
            # observation_state_map_copy = copy.deepcopy(pomcp.pomdp.observation_state_map)    # save state_map

            obs_current_node = pomcp.get_observation(state_ground_truth)

            obs_mdp, Winning_obs, A_valid, observation_state_map_change_record, state_observation_map_change_record  \
            = pomcp.pomdp.online_compute_winning_region(obs_current_node, AccStates, observation_successor_map, H, ACP_step)

            actionIndex = pomcp.select_action()                                             # compute using generated WR and updated state_map
            next_state_ground_truth = pomcp.step(state_ground_truth, actionIndex)
            reward = pomcp.step_reward(state_ground_truth, actionIndex)
            textinfo = str(pomcp.root.illegalActionIndexes)
            plot_figure(state_ground_truth, pomcp, Y_cur, estimation_moving_agents,  H, log_time, i_episode, cur_time, minD, textinfo)
            print("=====step", cur_time, "s", state_ground_truth, "action", pomcp.pomdp.actions[actionIndex],  "s'", next_state_ground_truth,
                    "observation", obs_current_node, pomcp.root.belief, Y_cur, "undiscounted reward", undiscounted_reward, pomcp.pomdp.obstacles,
                    pomcp.R_max, pomcp.R_min,)
            
            print("+++++",ACP_step,"*****")

            obs_nxt_node = pomcp.get_observation(next_state_ground_truth)
            pomcp.update(actionIndex, obs_nxt_node)
            state_ground_truth = next_state_ground_truth

            #----reset observation-state and state_observation map to default----
            pomcp.pomdp.restore_states_from_change(observation_state_map_change_record, state_observation_map_change_record )                    
            # print(pomcp.pomdp.observation_state_map == pomcp.pomdp.observation_state_map_default )
            # print(pomcp.pomdp.state_observation_map == pomcp.pomdp.state_observation_map_default )
            discounted_reward += pomcp.gamma * reward
            undiscounted_reward += reward
            
        plot_gif(figure_path = "./figures/{}-ShieldLevel-{}/Episode_{}/".format(log_time, shieldLevel, i_episode))