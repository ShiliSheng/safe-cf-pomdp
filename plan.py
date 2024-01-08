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

def get_min_distance(state_ground_truth, Y_cur_agents):
    cx, cy = state_ground_truth
    cur_min_distance = float("inf")
    for j in range(len(Y_cur_agents) // 2):
        px, py = Y_cur_agents[2 * j], Y_cur_agents[2 * j + 1]
        t = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        cur_min_distance = min(cur_min_distance, t)
    return cur_min_distance

def plot_gif(figure_path):
    image_files = sorted([f for f in os.listdir(figure_path) if f.endswith('.jpg')])
    images = [Image.open(os.path.join(figure_path, f)) for f in image_files]
    gif_path = figure_path.replace("figures", "gifs")
    if not os.path.exists(gif_path):
        os.makedirs(gif_path)
    imageio.mimsave(gif_path + 'output.gif', images, duration=0.5)  

def plot_figure(state_ground_truth, pomcp, Y_cur_agents, estimation_moving_agents,  H, log_time, i_episode, cur_time, cur_min_distance, textinfo):
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
    for i in range(0, len(Y_cur_agents), 2):
        plt.scatter(Y_cur_agents[i], Y_cur_agents[i+1], marker = '.', color = cmap(i//2), alpha=1)

    for tau in range(1, H + 1):
        for j in range(0, len(estimation_moving_agents[cur_time][tau]), 2):
            x, y = estimation_moving_agents[cur_time][tau][j], estimation_moving_agents[cur_time][tau][j + 1]
            r = constraints[cur_time + 1][tau]
            a = (0.5 - 0.2) / (1 - H)
            b = (0.5 * H - 0.2) / (H - 1)
            plt.scatter(x, y, color = cmap(j//2), marker = '.', alpha = a * tau + b)
            circle = patches.Circle((x,y), radius = r, edgecolor=cmap(i//2), facecolor=cmap(j//2), alpha =  a * tau + b)
            ax.add_patch(circle)
    plt.title("Time: {}, Action: {}, cur_min_distance:{}".format(cur_time, pomcp.pomdp.actions[actionIndex], str(cur_min_distance)[:3]))
    figure_path = "./figures/{}-ShieldLevel-{}/Episode_{}/".format(log_time, shieldLevel, i_episode)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    plt.text(0, 6, textinfo)
    plt.savefig(figure_path + "figure_{}.jpg".format(str(cur_time).zfill(3)), dpi=300)
    plt.close()

# def record_data(episode_result, episode_index, step,
#                         shieldLevel , history_length, 
#                         prediction_length , target_failure_prob_delta
#                         acp_learning_gamma, safe_distance, 
#                         cumulative_discounted_reward, , cumulative_undiscounted_reward, reward, 
#                         robot_state = state_ground_truth,  robot_belief = belief, 
#                         observation = obs_nxt_node, cur_min_distance = cur_min_distance,
#                         selected_action = action, disallowed_actions = disallowed_actions,
#                         dynamic_agents = Y_cur_agents, estimated_region = estimated_region,
#                         static_obstacles = pomdp.obstacles, refuel_stations = pomdp.refuel_stations, rocks = pomdp.rocks
#                         ):
        
#     record_data(episode_result, episode_index = episode_index, step = cur_time,
#                 shieldLevel = shieldLevel, history_length = history_length, 
#                 prediction_length = prediction_length , target_failure_prob_delta = target_failure_prob_delta,
#                 acp_learning_gamma = acp_learing_gamma,  safe_distance = safe_distance,
#                 cumulative_discounted_reward = cumulative_discounted_reward, 
#                 cumulative_undiscounted_reward = cumulative_undiscounted_reward, reward = reward,
#                 robot_state = state_ground_truth,  robot_belief = pomcp.root.belief.keys(), 
#                 observation = obs_nxt_node, cur_min_distance = cur_min_distance,
#                 selected_action = action, disallowed_actions = disallowed_actions,
#                 dynamic_agents = Y_cur_agents, estimated_region = estimated_region,
#                 static_obstacles = pomdp.obstacles, refuel_stations = pomdp.refuel_stations, rocks = pomdp.rocks
#                 )
#     return

if __name__ == "__main__":
    record_columns = ['Episode', 'Step', 'Shield Level', 'Look-back Length', 'Predict Horizon', 'Failure Rate', 
               'ACP Learning', 'Safe Distance', 'Cumulative Discounted Reward', 'Cumulative Undiscounted Reward', 
               'Step Reward', 'Robot State', 'Belief States', 'Current Minimum Distance to Agents', 'Selected Action',
                 'Disallowed Actions', 'Dynamcic Agents', 'Estimated Region', 'Stacic Obstacles', 'Refule Stations', 'Rocks']
    log_time = f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    
    max_steps = 100
    explore_constant = 1000
    shieldLevel = 1
    step = 0
    num_episodes = 3

    #Settings for LSTM trajectory prediction
    history_length, prediction_length, model_file = 4, 3, 'model_weights-4-3_2024-01-02_11-50.pth'
    trained_model_path = './OpenTraj/datasets/ETH/seq_eth/' + model_file
    prediction_model = Predictor(history_length, prediction_length)
    prediction_model.load_model(trained_model_path)
    history_length = prediction_model.history_length
    H = prediction_length = prediction_model.prediction_length

    # Settings for conformal prediction
    path = "./OpenTraj/datasets/ETH/seq_eth/"
    num_agents_tracked = 3
    test_dataset =  pd.read_csv(path + "test_dynammic_agents_0.csv", index_col = 0)
    starting_col_index = 0 # in the dataframe, where is the starting col of values
    target_failure_prob_delta = 0.1
    acp_learing_gamma = 0.2
    safe_distance = 0.5

    file_path = './results/' +  log_time + "-shield_{}-lookback_{}-prediction_{}-failure_{}/".format(shieldLevel,history_length, prediction_length, target_failure_prob_delta)
    if not os.path.exists(file_path): os.makedirs(file_path)

    result = pd.DataFrame(columns = record_columns)
    for episode_index in range(num_episodes):
        
        episode_result = pd.DataFrame()
        pomdp = create_scenario_obstacle(random_seed = 42 + episode_index)
        pomcp = POMCP(pomdp, shieldLevel, prediction_model.prediction_length, explore_constant)
        pomcp.reset_root()

        motion_mdp, AccStates = pomcp.pomdp.compute_accepting_states() 
        observation_successor_map = pomcp.pomdp.compute_H_step_space(H)

        dynamic_agents = prediction_model.create_online_dataset(test_dataset, num_agents_tracked)
        
        state_ground_truth = pomcp.root.sample_state_from_belief()
        cur_time = -1
        
        reward = 0
        cumulative_discounted_reward = 0
        cumulative_undiscounted_reward = 0
        discount = 1
        cur_min_distance = 1000
        action = ""

        error = [[0] * (H+1) for _ in range(max_steps + 10)]
        constraints = [[0] * (H+1) for _ in range(max_steps + 10)]    
        estimation_moving_agents = [[0 for _ in range(num_agents_tracked * 2)] * (H+1) for _ in range(max_steps + 10)] 
        cf_scores = defaultdict(SortedList)         # cf_scores = defaultdict(lambda: SortedList([float('inf')]))
        failure_prob_delta = [[0] * (H+1) for _ in range(max_steps + 10)]
        for tau in range(1, H + 1):
            failure_prob_delta[0][tau] = target_failure_prob_delta

        data = {
            "Episode": episode_index, "Step": step, 
            "Shield Level": shieldLevel, "Look-back Length": history_length, "Predict Horizon": prediction_length, 
            "Failure Rate": target_failure_prob_delta, "ACP Learning": acp_learing_gamma, "Safe Distance": safe_distance,
            "Cumulative Discounted Reward": cumulative_discounted_reward, 
            "Cumulative Undiscounted Reward": cumulative_undiscounted_reward, "Step Reward": reward,
            "Robot State": state_ground_truth, "Belief States": list(pomcp.root.belief.keys()), 
            "Refule Stations":pomcp.pomdp.refule_stations, "Rocks": pomcp.pomdp.rocks
            }
        # def append_row(episode_result, data):
        #     data_length = len(episode_result)
        #     for key in data:
        #         episode_result.loc[data_length, key] = data[key]
        episode_result = pd.concat([episode_result, pd.DataFrame([data], columns = data.keys())])

        done = False
        while not done and cur_time < max_steps:
            done = state_ground_truth in pomcp.pomdp.end_states
            cur_time += 1
            
            estimation = prediction_model.predict(dynamic_agents, cur_time, starting_col_index )  # Line 3, 4 
            for i, row in enumerate(estimation): estimation_moving_agents[cur_time][i+1] = row

            if cur_time < H: continue # assuming the agent is not starting until Timestamp H

            Y_cur_agents = dynamic_agents.loc[cur_time,:].values[starting_col_index:] # Check index
            cur_min_distance = get_min_distance(state_ground_truth, Y_cur_agents)

            for tau in range(1, H + 1):
                Y_est = estimation_moving_agents[cur_time-tau][tau]                                       # Line 7
                estimation_error = prediction_model.compute_prediction_error(Y_cur_agents, Y_est)                # TODO check formula of Line 7
                cf_scores[tau].add(estimation_error)

                error[cur_time][tau] = 0 if estimation_error <= constraints[cur_time][tau] else 1         # Line 6
                failure_prob_delta[cur_time + 1][tau] = failure_prob_delta[cur_time][tau] +  \
                                                        acp_learing_gamma * (target_failure_prob_delta * error[cur_time][tau])

                N = len(cf_scores[tau])
                q = math.ceil((N+1) * (1 - failure_prob_delta[cur_time+1][tau]))                            # Line 8
                estimated_region = constraints[cur_time + 1][tau] = 1 + 0.1 * tau if q > N else cf_scores[tau][q - 1]                                  # 0-indexed
                # print("____++++", constraints[cur_time + 1][tau], q, "N", N, "qlevel",len(cf_scores[tau]),  failure_prob_delta[cur_time + 1][tau], error[cur_time][tau], cf_scores[tau] )

            if q <= N: 
                ACP_step = pomdp.build_restrictive_region(estimation_moving_agents[cur_time], constraints[cur_time+1], H, safe_distance)

                obs_current_node = pomcp.get_observation(state_ground_truth)
                obs_mdp, Winning_obs, A_valid, observation_state_map_change_record, state_observation_map_change_record  \
                        = pomcp.pomdp.online_compute_winning_region(obs_current_node, AccStates, observation_successor_map, H, ACP_step)

                actionIndex = pomcp.select_action()          # compute using generated WR and updated state_map
                action = pomcp.pomdp.actions[actionIndex]
                # disallowed_actions = [
                next_state_ground_truth = pomcp.step(state_ground_truth, actionIndex)
                obs_nxt_node = pomcp.get_observation(next_state_ground_truth)
                pomcp.update(actionIndex, obs_nxt_node)

                reward = pomcp.step_reward(state_ground_truth, actionIndex)
                cumulative_discounted_reward  += reward * discount
                discount *= pomcp.gamma
                cumulative_discounted_reward += reward

                textinfo = str(pomcp.root.illegalActionIndexes)

                print("=====step", cur_time, "s", state_ground_truth, "action", pomcp.pomdp.actions[actionIndex],  "s'", next_state_ground_truth,
                        "observation", obs_current_node, pomcp.root.belief, Y_cur_agents, "undiscounted reward", cumulative_undiscounted_reward, pomcp.pomdp.obstacles,
                        pomcp.R_max, pomcp.R_min, ACP_step)
                print("constraints", constraints[cur_time+1])
                plot_figure(state_ground_truth, pomcp, Y_cur_agents, estimation_moving_agents,  H, log_time, episode_index, cur_time, cur_min_distance, textinfo)

                state_ground_truth = next_state_ground_truth
                pomcp.pomdp.restore_states_from_change(observation_state_map_change_record, state_observation_map_change_record )                    
                # print(pomcp.pomdp.observation_state_map == pomcp.pomdp.observation_state_map_default, pomcp.pomdp.state_observation_map == pomcp.pomdp.state_observation_map_default )

            
            data = {
                "Episode": episode_index, "Step": step, 
                "Shield Level": shieldLevel, "Look-back Length": history_length, "Predict Horizon": prediction_length, 
                "Failure Rate": target_failure_prob_delta, "ACP Learning": acp_learing_gamma, "Safe Distance": safe_distance,
                "Cumulative Discounted Reward": cumulative_discounted_reward, 
                "Cumulative Undiscounted Reward": cumulative_undiscounted_reward, "Step Reward": reward,
                "Robot State": state_ground_truth, "Belief States": list(pomcp.root.belief.keys()), 
                "Current Minimum Distance to Agents": cur_min_distance,  
                "Selected Action": action, "Disallowed Actions": [pomcp.actions[idx] for idx in pomcp.root.illegalActionIndexes],
                "Dynamcic Agents": Y_cur_agents, "Estimated Region": estimated_region, "Stacic Obstacles": pomcp.pomdp.obstacles, 
                "Refule Stations":pomcp.pomdp.refule_stations, "Rocks": pomcp.pomdp.rocks
            }
            episode_result = pd.concat([episode_result, pd.DataFrame([data], columns = data.keys())])

        episode_path = file_path + "Episode{}/".format(episode_index)
        if not os.path.exists(episode_path): os.makedirs(episode_path)
        episode_result.to_csv(episode_path + "{}.csv".format(episode_index))
        plot_gif(figure_path = "./figures/{}-ShieldLevel-{}/Episode_{}/".format(log_time, shieldLevel, episode_index))
        result = pd.concat([result, episode_result])
    result.to_csv(file_path + "result.csv")