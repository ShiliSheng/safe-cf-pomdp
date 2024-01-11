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
import os, yaml
import copy
from datetime import datetime

# def print(*args, **kwargs):
#     return

def get_min_distance(state_ground_truth, Y_cur_agents):
    cx, cy = state_ground_truth[0], state_ground_truth[1]
    cur_min_distance = float("inf")
    for j in range(len(Y_cur_agents) // 2):
        px, py = Y_cur_agents[2 * j], Y_cur_agents[2 * j + 1]
        t = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
        cur_min_distance = min(cur_min_distance, t)
    return cur_min_distance

def test(grid_size, shieldLevel, target_failure_prob_delta, num_agents_tracked = 3, num_episodes = 20):
    log_time = f"{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    max_steps = 200
    explore_constant = 1000
    # shieldLevel = 1
    # num_episodes = 20

    #Settings for LSTM trajectory prediction
    history_length, prediction_length, model_file = 4, 3, 'model_weights-4-3_2024-01-02_11-50.pth'
    trained_model_path = './OpenTraj/datasets/ETH/seq_eth/' + model_file
    prediction_model = Predictor(history_length, prediction_length)
    prediction_model.load_model(trained_model_path)
    history_length = prediction_model.history_length
    H = prediction_length = prediction_model.prediction_length

    # Settings for conformal prediction
    path = "./OpenTraj/datasets/ETH/seq_eth/"
    test_dataset =  pd.read_csv(path + "test_dynammic_agents_0.csv", index_col = 0)
    starting_col_index = 0 # in the dataframe, where is the starting col of values
    # target_failure_prob_delta

    acp_learing_gamma = 0.2
    safe_distance = 0.5
    model_type = "Obstacle"

    file_path = './results/' + model_type +  "GridSize-{}".format(grid_size) + "/"  + "shield_{}-lookback_{}-prediction_{}-failure_{}".format(shieldLevel,history_length, prediction_length, target_failure_prob_delta) + "-" + log_time +"/"
    if not os.path.exists(file_path): os.makedirs(file_path)

    results = []
    pomdp = create_scenario_obstacle(random_seed = 42)
    for episode_index in range(num_episodes):
        print(file_path, episode_index)
        step_record = []
        pomcp = POMCP(pomdp, shieldLevel, prediction_model.prediction_length, explore_constant)
        pomcp.reset_root()
        motion_mdp, AccStates = pomcp.pomdp.compute_accepting_states() 
        observation_successor_map = pomcp.pomdp.compute_H_step_space(H)

        dynamic_agents = prediction_model.create_online_dataset(test_dataset, num_agents_tracked)
        
        state_ground_truth = pomcp.root.sample_state_from_belief()
        reward = 0
        cumulative_discounted_reward = 0
        cumulative_undiscounted_reward = 0
        discount = 1
        cur_min_distance = 1000
        count_unsafe_action = 0
        count_unsafe_state = 0
        action = ""

        error = [[0] * (H+1) for _ in range(max_steps + 10)]
        constraints = [[0] * (H+1) for _ in range(max_steps + 10)]    
        estimation_moving_agents = [[0 for _ in range(num_agents_tracked * 2)] * (H+1) for _ in range(max_steps + 10)] 
        cf_scores = defaultdict(SortedList)         # cf_scores = defaultdict(lambda: SortedList([float('inf')]))
        failure_prob_delta = [[0] * (H+1) for _ in range(max_steps + 10)]
        for tau in range(1, H + 1):
            failure_prob_delta[0][tau] = target_failure_prob_delta

        done = state_ground_truth in pomcp.pomdp.end_states
        cur_time = 0
        action_step = 0
        clock_time = time.time()

        while not done and cur_time < max_steps:
            done = state_ground_truth in pomcp.pomdp.end_states
            if (state_ground_truth[0], state_ground_truth[1]) in pomcp.pomdp.obstacles:
                count_unsafe_state += 1
            
            estimation = prediction_model.predict(dynamic_agents, cur_time, starting_col_index )  # Line 3, 4 
            
            for i, row in enumerate(estimation): estimation_moving_agents[cur_time][i+1] = row
            
            Y_cur_agents = dynamic_agents.loc[cur_time,:].values[starting_col_index:] # Check index

            cur_agents = set([(Y_cur_agents[i], Y_cur_agents[i+1]) for i in range(0, len(Y_cur_agents), 2)])

            cur_min_distance = get_min_distance(state_ground_truth, Y_cur_agents)

            if cur_time < H: 
                data = {
                    "Episode": episode_index, "Current Time Step": cur_time, "Action Step": action_step, "Clock Time": time.time() - clock_time,
                    "Shield Level": shieldLevel, "Look-back Length": history_length, "Predict Horizon": prediction_length, 
                    "Failure Rate": target_failure_prob_delta, "ACP Learning": acp_learing_gamma, "Safe Distance": safe_distance,
                    "Cumulative Discounted Reward": cumulative_discounted_reward, 
                    "Cumulative Undiscounted Reward": cumulative_undiscounted_reward, "Step Reward": reward,
                    "Robot State": state_ground_truth, "Belief States": list(pomcp.root.belief.keys()), 
                    "Current Minimum Distance to Agents": cur_min_distance,   "Dynamcic Agents": Y_cur_agents, 
                    "Stacic Obstacles": pomcp.pomdp.obstacles,  "Refule Stations":pomcp.pomdp.refule_stations, "Rocks": pomcp.pomdp.rocks,
                    "Number of Unsafe Action": count_unsafe_action, "Number of Unsafe State": count_unsafe_state, "done": done,
                    "Number of Dynamic Agents": num_agents_tracked, 
                    # "Selected Action": action, "Disallowed Actions": [pomcp.pomdp.actions[idx] for idx in pomcp.root.illegalActionIndexes],
                    # "Estimated Regions": estimated_region, 
                }
                step_record.append(data)
                cur_time += 1
                continue # assuming the agent is not starting until Timestamp H

            for tau in range(1, H + 1):
                Y_est = estimation_moving_agents[cur_time-tau][tau]                                       # Line 7
                
                estimation_error = prediction_model.compute_prediction_error(Y_cur_agents, Y_est)         # TODO check formula of Line 7
                
                cf_scores[tau].add(estimation_error)

                error[cur_time][tau] = 0 if estimation_error <= constraints[cur_time][tau] else 1         # Line 6
                failure_prob_delta[cur_time + 1][tau] = failure_prob_delta[cur_time][tau] +  \
                                                        acp_learing_gamma * (target_failure_prob_delta * error[cur_time][tau])

                N = len(cf_scores[tau])
                q = math.ceil((N+1) * (1 - failure_prob_delta[cur_time+1][tau]))                            # Line 8
                constraints[cur_time + 1][tau] = 1 + 0.1 * tau if q > N else cf_scores[tau][q - 1]                                  # 0-indexed
                # print("____++++", constraints[cur_time + 1][tau], q, "N", N, "qlevel",len(cf_scores[tau]),  failure_prob_delta[cur_time + 1][tau], error[cur_time][tau], cf_scores[tau] )

            if q > N: 
                data = {
                    "Episode": episode_index, "Current Time Step": cur_time, "Action Step": action_step, "Clock Time": time.time() - clock_time,
                    "Shield Level": shieldLevel, "Look-back Length": history_length, "Predict Horizon": prediction_length, 
                    "Failure Rate": target_failure_prob_delta, "ACP Learning": acp_learing_gamma, "Safe Distance": safe_distance,
                    "Cumulative Discounted Reward": cumulative_discounted_reward, 
                    "Cumulative Undiscounted Reward": cumulative_undiscounted_reward, "Step Reward": reward,
                    "Robot State": state_ground_truth, "Belief States": list(pomcp.root.belief.keys()), 
                    "Current Minimum Distance to Agents": cur_min_distance,   "Dynamcic Agents": Y_cur_agents, 
                    "Stacic Obstacles": pomcp.pomdp.obstacles,  "Refule Stations":pomcp.pomdp.refule_stations, "Rocks": pomcp.pomdp.rocks,
                    "Number of Unsafe Action": count_unsafe_action, "Number of Unsafe State": count_unsafe_state, "done": done,
                    "Number of Dynamic Agents": num_agents_tracked, 
                    # "Selected Action": action, "Disallowed Actions": [pomcp.pomdp.actions[idx] for idx in pomcp.root.illegalActionIndexes],
                    # "Estimated Regions": estimated_region, 
                }
                step_record.append(data)
                cur_time += 1
                continue
            ACP_step = pomdp.build_restrictive_region(estimation_moving_agents[cur_time], constraints[cur_time+1], H, safe_distance)
            obs_current_node = pomcp.get_observation(state_ground_truth)
            # t1 = time.time()
            obs_mdp, Winning_obs, A_valid, observation_state_map_change_record, state_observation_map_change_record  \
                    = pomcp.pomdp.online_compute_winning_region(obs_current_node, AccStates, observation_successor_map, H, ACP_step)
            # t2 = time.time()
            # print("time for winning", t2 - t1)
            
            # for cur_agent_state in cur_agents: pomdp.state_reward[cur_agent_state] -= 50
            actionIndex = pomcp.select_action()          # compute using generated WR and updated state_map
            # for cur_agent_state in cur_agents: pomdp.state_reward[cur_agent_state] += 50

            if actionIndex == -1:
                count_unsafe_action += 1
                if pomcp.pomdp.preferred_actions:
                    actionIndex = random.choice(pomcp.pomdp.preferred_actions)
                else:
                    actionIndex = random.choice([idx for idx in range(len(pomcp.pomdp.actions))])
            # t3 = time.time()
            # print("time for actiong", t3 - t2)
            action = pomcp.pomdp.actions[actionIndex]
            reward = pomcp.step_reward(state_ground_truth, actionIndex)
            cumulative_discounted_reward  += reward * discount
            discount *= pomcp.gamma
            cumulative_undiscounted_reward += reward

            data = {
                "Episode": episode_index, "Current Time Step": cur_time, "Action Step": action_step, "Clock Time": time.time() - clock_time,
                "Shield Level": shieldLevel, "Look-back Length": history_length, "Predict Horizon": prediction_length, 
                "Failure Rate": target_failure_prob_delta, "ACP Learning": acp_learing_gamma, "Safe Distance": safe_distance,
                "Cumulative Discounted Reward": cumulative_discounted_reward, 
                "Cumulative Undiscounted Reward": cumulative_undiscounted_reward, "Step Reward": reward,
                "Robot State": state_ground_truth, "Belief States": list(pomcp.root.belief.keys()), 
                "Current Minimum Distance to Agents": cur_min_distance,   "Dynamcic Agents": Y_cur_agents, 
                "End States": pomcp.pomdp.end_states,
                "Stacic Obstacles": pomcp.pomdp.obstacles,  "Refule Stations":pomcp.pomdp.refule_stations, "Rocks": pomcp.pomdp.rocks,
                "Number of Unsafe Action": count_unsafe_action, "Number of Unsafe State": count_unsafe_state, "done": done,
                "Number of Dynamic Agents": num_agents_tracked, 
                "Selected Action": action, "Disallowed Actions": [pomcp.pomdp.actions[idx] for idx in pomcp.root.illegalActionIndexes],
                "Estimated Regions": constraints[cur_time + 1], "Dynamic Agents Prediction": estimation_moving_agents[cur_time],
            }
            step_record.append(data)
            action_step += 1
            next_state_ground_truth = pomcp.step(state_ground_truth, actionIndex)
            obs_nxt_node = pomcp.get_observation(next_state_ground_truth)
            pomcp.update(actionIndex, obs_nxt_node)
            state_ground_truth = next_state_ground_truth
            pomcp.pomdp.restore_states_from_change(observation_state_map_change_record, state_observation_map_change_record )                    
            # print(pomcp.pomdp.observation_state_map == pomcp.pomdp.observation_state_map_default, pomcp.pomdp.state_observation_map == pomcp.pomdp.state_observation_map_default )
            cur_time += 1

        with open(file_path + "Episode-{}.pkl".format(episode_index), 'wb') as file:
            pickle.dump(step_record, file)

        # episode_log = pd.concat([pd.DataFrame([dt], columns = dt.keys()) for dt in step_record], ignore_index=True)
        # episode_log.to_csv(file_path + "Episode-{}.csv".format(episode_index))
        episode_min_dist = min([r["Current Minimum Distance to Agents"] for r in step_record])
        index_of_action_step_1 = -1
        for k, r in enumerate(step_record):
            if k+1 < len(step_record) and step_record[k+1]["Action Step"] == 1:
                index_of_action_step_1 = k
        action_time_spent = step_record[-1]["Clock Time"] - step_record[index_of_action_step_1]["Clock Time"]

        experiment_data = {
            "Grid Size": grid_size,
            "Episode": episode_index, "Number of Dynamic Agents": num_agents_tracked,  "Model": model_type,
            "Shield Level": shieldLevel, "Look-back Length": history_length, "Predict Horizon": prediction_length, 
            "Failure Rate": target_failure_prob_delta, "ACP Learning": acp_learing_gamma, "Safe Distance": safe_distance,
            "Reached Target": 1 if done else 0,
            "Minimum Distance to Agents": episode_min_dist,
            "Number of Unsafe State": count_unsafe_state,
            "Number of Unsafe Action": count_unsafe_action,
            "Cumulative Discounted Reward": cumulative_discounted_reward,
            "Cumulative Undiscounted Reward": cumulative_undiscounted_reward,
            "Max Step": max_steps,
            "Number of Time Steps": cur_time,
            "Number of Action Steps": action_step,
            "Time spent in seconds": step_record[-1]["Clock Time"],
            "Action Time spent in seconds": action_time_spent,
            "Action Time spent per action in seconds": action_time_spent / action_step, 
            "POMCP Number of Simulations": pomcp.numSimulations,
            "POMCP constant": pomcp.c,
        }
        
        results.append(pd.DataFrame([experiment_data], columns = experiment_data.keys()))
    result = pd.concat(results, ignore_index=True)
    result.to_csv(file_path + "summary.csv")

if __name__ == "__main__":
    setting = [
                (0, 0.1, 5), (1, 0.05, 5), (1, 0.1, 5), (1, 0.2, 5)
                ]
    
    for grid_size in [22]:
        for prediction_horizon in [3, 5, 10]:
            for num_agents_tracked in [5, 10, 15]:
                for target_failure_prob_delta in [0.05, 0.1, 0.2]:
                    for shieldLevel in [0, 1]:
                        test(grid_size, shieldLevel, target_failure_prob_delta, num_agents_tracked, num_episodes = 100)
    
    # make sure to set up predicition model
    # adjust agents locations
    # adjust grid size                        
                        
    # test(grid_size = 22, shieldLevel = 1, target_failure_prob_delta = 0.1, num_agents_tracked = 3, num_episodes = 3)
    # pass