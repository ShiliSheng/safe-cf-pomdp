from MDP_TG.mdp import Motion_MDP_label, Motion_MDP, compute_accept_states
from MDP_TG.dra import Dra, Dfa, Product_Dra, Product_Dfa
from MDP_TG.vi4wr import syn_plan_prefix, syn_plan_prefix_dfa
from networkx.classes.digraph import DiGraph
import pickle
import numpy as np
import time
import math
import random
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

# def get_cf_score(dynamic_agents, cur_time, estimation):
#     values = dynamic_agents.loc[cur_time,:].values[2:]
#     return np.linalg.norm(values - estimation)
#     # score = 0
    # num_agents = get_num_agents(dynamic_agents)
    # for i in range(2, num_agents, 2):
    #     score += (dynamic_agents.loc[cur_time, ] - B[i][0]) ** 2 + (A[i][1] - B[i][0]) ** 2
    # return score ** (0.5)

def compute_prediction_error(A, B):
    N = len(A)
    distance = 0
    for i in range(0, N, 2):
        distance = (A[i] - B[i]) ** 2 + (A[i+1] - B[i+1]) ** 2
    distance = distance ** 0.5 / N
    # distance = np.linalg.norm(A - B)
    # print("distance",distance)
    return distance

if __name__ == "__main__":
    #Settings for LSTM trajectory prediction
    prediction_model = Predictor()
    prediction_model.load_model('./OpenTraj/datasets/ETH/seq_eth/model_weights.pth')
    history_length = prediction_model.history_length
    H = prediction_length = prediction_model.prediction_length

    # POMDP
    U = actions = ['N', 'S', 'E', 'W', 'ST']
    C = cost = [-1, -1, -1, -1, -1]

    transition_prob = [[] for _ in range(len(actions))]
    transition_prob[0] = [0.1, 0.8, 0.1] # S
    transition_prob[1] = [0.1, 0.8, 0.1] # N
    transition_prob[2] = [0.1, 0.8, 0.1] # E
    transition_prob[3] = [0.1, 0.8, 0.1] # W
    transition_prob[4] = [1]             # ST

    WS_transition = [[] for _ in range(len(actions))]
    WS_transition[0] = [(-2, 2), (0, 2), (2, 2)]       # S
    WS_transition[1] = [(-2, -2), (0, -2), (2, -2)]    # N
    WS_transition[2] = [(2, -2), (2, 0), (2, 2)]       # E
    WS_transition[3] = [(-2, -2), (-2, 0), (-2, 2)]    # W
    WS_transition[4] = [(0, 0)]                         # ST

    obstacles =  [(3, 7)]
    target = [(19, 19)]
    end_states = set([(19,1)])

    robot_nodes = set()
    for i in range(1, 20, 2):
        for j in range(1, 20, 2):
            node = (i, j)
            robot_nodes.add(node) 

    initial_belief_support = [(5,5), (5,7)]
    initial_belief = {}
    for state in initial_belief_support:
        initial_belief[state] = 1 / len(initial_belief_support)

    pomdp = Model(robot_nodes, actions, cost, WS_transition, transition_prob,
                     initial_belief, obstacles, target, end_states)

    pomcp = POMCP(pomdp)

    motion_mdp, AccStates = pomcp.pomdp.compute_accepting_states() 
    observation_successor_map = pomcp.pomdp.compute_H_step_space(motion_mdp, H)
    step = 0
    discounted_reward = 0
    undiscounted_redward = 0
    num_episodes = 1
    max_steps = 15

    # Settings for conformal prediction
    path = "./OpenTraj/datasets/ETH/seq_eth/"
    dynamic_agents =  pd.read_csv(path + "test_dynammic_agents_1.csv")
    starting_col_index = 1 # in the dataframe, where is the starting col of values

    # print(dynamic_agents.head())
    #           agent1, anget2  angentN
    # t=0       (x,y)       .        .
    # ...           .       .        .  
    # t=inf         .       .       (x,y)
    num_agents_tracked = len(dynamic_agents.columns) // 2 -1
    target_failure_prob_delta = 0.1
    acp_learing_gamma = 0.08
    failure_prob_delta = [[0] * (H+1) for _ in range(max_steps + 10)]
    error = [[0] * (H+1) for _ in range(max_steps + 10)]
    constraints = [[0] * (H+1) for _ in range(max_steps + 10)]    
    estimation_moving_agents = [[0 for _ in range(num_agents_tracked * 2)] * (H+1) for _ in range(max_steps + 10)] 
    # cf_scores = [[0] * (H+1) for _ in range(max_steps + 10)]
    cf_scores = defaultdict(lambda: SortedList([float('inf')]))

    for tau in range(1, H + 1):
        failure_prob_delta[0][tau] = target_failure_prob_delta

    for _ in range(num_episodes):
        pomcp.reset_root()
        state_ground_truth = pomcp.root.sample_state_from_belief()
        print(state_ground_truth, "current state")
        obs_current_node = pomcp.get_observation(state_ground_truth)
        print("current observation", obs_current_node)
        cur_time = -1

        while cur_time < max_steps:
            cur_time += 1

            # Line 3, 4 TODO preprocessing
            estimation = prediction_model.predict(dynamic_agents, cur_time, starting_col_index )
            for i, row in enumerate(estimation):
                estimation_moving_agents[cur_time][i+1] = row

            if cur_time < H: # assuming the agent is not starting until Timestamp H
                continue

            Y_cur = dynamic_agents.loc[cur_time,:].values[starting_col_index:] # TODO check index

            for tau in range(1, H + 1):
                # Line 7
                Y_est = estimation_moving_agents[cur_time-tau][tau]
                estimation_error = compute_prediction_error(Y_cur, Y_est)# TODO check formula of Line 7
                # cf_scores[cur_time][tau] = estimation_error
                cf_scores[tau].add(estimation_error)

                # Line 6
                error[cur_time][tau] = 0 if estimation_error <= constraints[cur_time][tau] else 1 # a
                failure_prob_delta[cur_time + 1][tau] = failure_prob_delta[cur_time][tau] +  acp_learing_gamma * (target_failure_prob_delta * error[cur_time][tau])

                # Line 8
                # N = (cur_time + 1 - tau)
                N = len(cf_scores[tau]) - 1
                q = math.ceil(N * (1 - failure_prob_delta[cur_time+1][tau]))

                # Line 9
                # values = [cf_scores[k][tau] for k in range(tau, cur_time+1)] + [float("inf")]
                # values.sort()
                # radius = values[q-1]
                radius = cf_scores[tau][q - 1]
                constraints[cur_time + 1][tau] = radius
                
            print(constraints[cur_time + 1][tau], "_______________")

            ACP_step = pomdp.build_restrictive_region(estimation_moving_agents[cur_time], constraints[cur_time+1][tau], H)
            obs_mdp, Winning_observation = pomcp.pomdp.online_compute_winning_region(obs_current_node, AccStates, observation_successor_map, H, ACP_step)
            
            actionIndex = pomcp.select_action()
            next_state_ground_truth = pomcp.step(state_ground_truth, actionIndex)
            reward = pomcp.step_reward(state_ground_truth, actionIndex)
            obs_current_node = pomcp.get_observation(next_state_ground_truth)

            fig, ax = plt.subplots()   
            ax.set_aspect('equal')             
            plt.xlim(0, 22)
            plt.ylim(0, 22)
            # circle = patches.Circle(state_ground_truth, radius=0.5, edgecolor='black', facecolor='none')
            # ax.add_patch(circle)

            plt.scatter(state_ground_truth[0], state_ground_truth[1], marker = '*', color = "black")
            for (x, y) in pomcp.root.belief:
                plt.scatter(x, y, marker = '*', alpha=0.5, color = "black")
            
            cmap = plt.get_cmap('tab10')
            for i in range(0, len(Y_cur), 2):
                plt.scatter(Y_cur[i], Y_cur[i+1], marker = '.', color = cmap(i//2), alpha=1)

            for tau in range(1, H + 1):
                for j in range(0, len(estimation_moving_agents[cur_time][tau]), 2):
                    x, y = estimation_moving_agents[cur_time][tau][j], estimation_moving_agents[cur_time][tau][j + 1]
                    r = constraints[cur_time + 1][tau]
                    # true_x, true_y = dynamic_agents.loc[cur_time + tau, :].values[j + 1], dynamic_agents.loc[cur_time + tau, :].values[j + 2]
                    # print("radius", x, y, r, true_x, true_y)
                    a = (0.5 - 0.2) / (1 - H)
                    b = (0.5 * H - 0.2) / (H - 1)
                    plt.scatter(x, y, color = cmap(j//2), marker = '.', alpha = a * tau + b)
                    circle = patches.Circle((x,y), radius = r, edgecolor=cmap(i//2), facecolor=cmap(j//2), alpha =  a * tau + b)
                    ax.add_patch(circle)
            plt.title("Time: {}, Action: {}".format(cur_time, actions[actionIndex]))
            plt.savefig("./figures/figure_{}.jpg".format(str(cur_time).zfill(3)), dpi=300)

            print("=====step", cur_time, "s", "action", actions[actionIndex], state_ground_truth, "s'", next_state_ground_truth,
                    "observation", obs_current_node, pomcp.root.belief, Y_cur)
            pomcp.update(actionIndex, obs_current_node)
            state_ground_truth = next_state_ground_truth
            discounted_reward += pomcp.gamma * reward
            undiscounted_redward += reward
        
        image_files = sorted([f for f in os.listdir('./figures/') if f.endswith('.jpg')])
        images = [Image.open(os.path.join('./figures/', f)) for f in image_files]
        imageio.mimsave('./data/output.gif', images, duration=0.5)  # 设置每帧之间的时间间隔（单位：秒）


