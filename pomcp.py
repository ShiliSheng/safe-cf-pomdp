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
from model import create_scenario

from predictor import Predictor
from collections import defaultdict
import pandas as pd
from itertools import chain
from sortedcontainers import SortedList
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import imageio
import os
import sys
class ForcedExitException(Exception):
    pass
# class POMCPBelief:
#     def __init__(self) -> None:
#         self.particles = []
#         self.uniqueStates = set()
    
#     def copy(self, orignal):
#         self.particles = [x for x in orignal.particles]
#         self.uniqueStates = set(self.particles)
    
#     def empty(self):
#         return not self.particles

#     def is_depleted(self):
#         return not self.particles
    
#     def add_particle(self, state):
#         self.particles.append(state)
#         self.uniqueStates.add(state)

#     def is_state_in_belief_support(self, state):
#         return state in self.uniqueStates
    
#     def update_belief_suport(self, state):
#         self.uniqueStates.add(state)

#     def size(self):
#         return len(self.particles)

class POMCPNode:
    def __init__(self) -> None:
        self.belief = defaultdict(int)
        self.belief_particles = []
        self.clear()

    def clear(self):
        self.parent = None
        self.children = {}
        self.h = -1
        self.isQNode = False
        self.v = 0
        self.n = 0
        self.time = 0
        self.illegalActionIndexes = set()
    
    def add_particle(self, state):
        self.belief[state] = self.belief.get(state, 0) + 1;
    
    def remove_particle(self, state):
        if state in self.belief and self.belief[state] > 0:
            self.belief[state] -= 1
            if self.belief[state] == 0:
                del self.belief[state]

    def add_illegal_action_index(self, actionIndex):
        self.illegalActionIndexes.add(actionIndex)
        return self.remove_child(actionIndex)
        
    def is_action_index_illegal(self, actionIndex):
        return actionIndex in self.illegalActionIndexes

    def set_h_action(self, isAction):
        self.isQNode = isAction

    def getH(self):
        return self.h
    
    def getParent(self):
        return self.parent
    
    def getV(self):
        return self.v / self.n
    
    def increaseV(self, value):
        self.v += value
        self.n += 1

    def add_child(self, node, index):
        self.children[index] = node
    
    def remove_child(self, index):
        if index not in self.children: return
        qchild = self.children[index]
        if qchild:
            self.v -= qchild.v
            self.n -= qchild.n
            if self.n < 0:
                self.v = 0
                self.n = 0
        del self.children[index]
        if len(self.children) == 0:
            if not self.parent: # already root:
                return -1
            qParent = self.parent
            vparent = qParent.parent
            return vparent.remove_child(qParent.h)
        return 0   
    
    def get_child_by_action_index(self, index):
        if index in self.children: return self.children[index]
        child = POMCPNode()
        child.h = index
        child.parent = self
        self.add_child(child, index)
        return child

    def check_child_by_observation_index(self, index):
        return index in self.children
    
    def get_child_by_observation_index(self, index):
        if index not in self.children:
            child = POMCPNode()
            child.h = index
            child.parent = self
            self.add_child(child, index)
        return self.children[index]
    
    def sample_state_from_belief(self):
        # return random.choice(self.belief_particles)
        states, counts = zip(*self.belief.items())
        total_counts = sum(cnt for cnt in counts)
        probs = (cnt / total_counts for cnt in counts)
        return random.choices(states, weights=probs, k=1)[0]

    def get_belief_suport(self):
        return self.belief.keys()
    
    def get_time(self):
        return self.time

    def have_state_in_belief_support(self, state):
        return state in self.belief
    
class POMCP:
    def __init__(self, pomdp, shieldLevel = 0, shieldHorizon = 5,  
                 constant = 10, maxDepth = 100, gamma = 0.99, 
                 numSimulations = 2 ** 12, pomcp_init_R_max = 0,
                 ):

        # def __init__(self, initial_belief, actions, robot_state_action_map, state_to_observation, state_action_reward_map, 
        #              end_states, constant = 1000, maxDepth = 100, targets = set()):
        #e (float): Threshold value below which the expected sum of discounted rewards for the POMDP is considered 0. Default value is 0.005.
        # c (float): Parameter that controls the importance of exploration in the UCB heuristic. Default value is 1.
        # no_particles (int): Controls the maximum number of particles that will be kept at each node 
        #                       and the number of particles that will be sampled from the posterior belief after an action is taken.
        self.numSimulations = numSimulations
        self.gamma = gamma
        self.e = 0.05
        self.noParticles = 1200
        self.K = 10000
        self.TreeDepth = 0
        self.PeakTreeDepth = 0
        self.constant = constant
        self.maxDepth = maxDepth
        self.verbose = 0
        self.pomdp = pomdp
        self.target = pomdp.targets
        self.end_states = pomdp.end_states

        self.root = None
        self.is_min = False
        self.stateOfInteste = None
        self.varNames = []
        self.variabelIndexX = None
        self.variableIndexY = None
        self.stateSuccessorsHashSet = {}
        self.stateSuccessorArryList = {}
        self.stateSuccessorCumProb = {}
        self.shieldLevel = shieldLevel
        self.horizon = shieldHorizon
        self.initializePOMCP()
        self.R_max = float("-inf")
        self.R_min = float("inf")
        self.pomcp_init_R_max = pomcp_init_R_max

    def initializePOMCP(self):
        self.TreeDepth = 0
        self.PeakTreeDepth = 0
        self.initialUCB(1000, 100)
        # this.shieldLevel = NO_SHIELD; 
        # 		this.useLocalShields = false;

    def fastUCB(self, N, n, logN):
        if N < 1000 and n < 100: return self.UCB[N][n]
        if n == 0: return float("inf")
        return (logN / n) ** 0.5 * self.constant
    
    def initialUCB(self, UCB_N, UCB_n):
        self.UCB = [[0] * UCB_n for _ in range(UCB_N)]
        for N in range(UCB_N):
            for n in range(UCB_n):
                if n == 0: self.UCB[N][n] = float("inf")
                else: self.UCB[N][n] = math.log(N + 1) / n

    def set_num_simulations(self, n):
        self.num_simulations = n

    def set_verbose(self, v):
        self.verbose = v

    def set_root(self, node):
        self.root = node

    def reset_root(self):
        self.root = POMCPNode()
        for key, prob in self.pomdp.initial_belief.items():
            self.root.belief[key] = prob * self.K
        # for state, state_fre in self.root.belief.items():
        #     for _ in range(state_fre):
        #         self.root.belief_particles.append(state)
        self.R_max = float("-inf")
        self.R_min = float("inf")

    def draw_from_probabilities(self, probabilities):
        states, probs = zip(*probabilities.items())
        next_state = random.choices(states, weights=probs, k=1)[0]
        return next_state
    
    def step(self, state, actionIndex):
        probabilities = self.pomdp.robot_state_action_map[state][actionIndex]
        states, probs = zip(*probabilities.items())
        next_state = random.choices(states, weights=probs, k=1)[0]
        return next_state
    
    def get_observation(self, state):
        if (state not in self.pomdp.state_observation_map):
            print("error")
        return self.pomdp.state_observation_map[state]
    
    def invigorate_belief(self, parent, child, action_index, obs):
        # fill child belief with particles
        child_belief_size = sum(cnt for cnt in child.belief.values())

        states, counts = zip(*self.root.belief.items())
        sum_count = sum(counts)
        probs = [cnt / sum_count for cnt in counts]

        while child_belief_size < self.K:
            # s = parent.sample_state_from_belief() # sample
            s = random.choices(states, weights=probs, k=1)[0]

            next_state = self.step(s, action_index)
            obs_sample = self.get_observation(next_state)
            if obs_sample == obs:
                child.belief[next_state] = child.belief.get(next_state, 0) + 1
                child_belief_size += 1

    def update(self, actionIndex, obs):
        qnode = self.root.get_child_by_action_index(actionIndex)
        vnode = qnode.get_child_by_action_index(obs)
        self.invigorate_belief(self.root, vnode, actionIndex, obs)
        vnode.clear()
        self.root = vnode

    def get_default_action(self):
        return self.pompd.actions[0]

    def select_action(self):
        distableTrue = False
        if distableTrue: return -1
        self.UCT_search()
        actionIndex = self.greedyUCB(self.root, False)
        return actionIndex    

    def UCT_search(self):
        states, counts = zip(*self.root.belief.items())
        sum_count = sum(counts)
        probs = [cnt / sum_count for cnt in counts]
        for n in range(self.numSimulations):
            # state = self.root.sample_state_from_belief() # sample
            state = random.choices(states, weights=probs, k=1)[0]
            if self.verbose >= 2: print("====Start UCT search with sample state", state, "nums Search", n)
            self.TreeDepth = 0
            self.PeakTreeDepth = 0
            reward = self.simulateV(state, self.root)
            self.R_max = max(self.R_max, reward)
            self.R_min = min(self.R_min, reward)

            if (self.verbose >= 2):
                print("==MCTS after num simulation", n)
        for actionIndex in self.root.children:
            qnode = self.root.get_child_by_action_index(actionIndex)
            # print("MCTS",actionIndex, qnode.v, qnode.n,qnode.v / (1+qnode.n) )
        # if self.verbose >= 1:
        #     print("finishing all simulations", self.numSimulations)
            
    # def check_winning(self, state, time_count):
    #     return self.pomdp.check_winning(state, time_count)
    
    def check_winning_set(self, belief_support, time_count):
        return self.pomdp.check_winning_set(belief_support, time_count)

    def get_observation_from_beleif(self, belief):
        for state in belief:
            return self.get_observation(state)

    def expand(self, parent, state):
        available_actions = self.get_legal_actions(state)
        for actionIndex in available_actions:
            if self.shieldLevel >= 1 and parent.is_action_index_illegal(actionIndex):
                continue
            qnode = POMCPNode()
            qnode.set_h_action(True)
            qnode.h = actionIndex
            qnode.parent = parent
            parent.add_child(qnode, actionIndex)
            # TODO
            # if self.shieldLevel == 1 and self.TreeDepth == 0 and self.isActionShieldedForNode(parent, actionIndex):
            #     parent.add_illegal_action_index(actionIndex)
            
            # if self.constant != 0:
            #     if actionIndex in self.pomdp.preferred_actions:
            #         qnode.v = self.pomcp_init_R_max
            #         qnode.n = 10

        if not parent.children:
            # print("add default available actions")
            for actionIndex in available_actions:
                qnode = POMCPNode()
                qnode.set_h_action(True)
                qnode.h = actionIndex
                qnode.parent = parent
                parent.add_child(qnode, actionIndex)
    
    def expand_node(self, state):
        vnode = POMCPNode()
        # vnode.belief[state] += 1 # node should not be added into belief before checking.
        available_actions = self.get_legal_actions(state)
        for actionIndex in available_actions:
            qnode = POMCPNode()
            qnode.h = actionIndex
            qnode.set_h_action(True)
            qnode.parent = vnode
            vnode.add_child(qnode, actionIndex)
        return vnode

    def simulateV(self, state, vnode):
        if (self.TreeDepth >= self.maxDepth): 
            return 0
        self.PeakTreeDepth = self.TreeDepth
        if not vnode.children:
            self.expand(vnode, state)

        actionIndex = self.greedyUCB(vnode, True)
        winning = True
        
        already_in = vnode.have_state_in_belief_support(state)
        
        if self.TreeDepth >= 1:
            vnode.add_particle(state)
        
        if self.shieldLevel > 0 and self.TreeDepth <= self.horizon and not already_in :
            # check if this state is winning or not   
            # winning = self.check_winning(state, self.TreeDepth)
            winning = self.check_winning_set(vnode.belief.keys(), self.TreeDepth)
            if not winning:
                vnode.remove_particle(state)
                qparent = vnode.getParent() 
                parentActionIndex = qparent.getH()
                vparent = qparent.getParent() 
                vparent.add_illegal_action_index(parentActionIndex)
                # print("not winning", self.TreeDepth, parentActionIndex, state, self.get_observation(state))

        qnode = vnode.get_child_by_action_index(actionIndex)
        total_reward = self.simulateQ(state, qnode, actionIndex)
        vnode.increaseV(total_reward)
        return total_reward
    
    def simulateQ(self, state, qnode, actionIndex):
        delayed_reward = 0
        nextState = self.step(state, actionIndex)
        observation = self.get_observation(nextState)
        done = nextState in self.end_states
        immediate_reward = self.step_reward(state, actionIndex)
        total_reward = 0

        if self.verbose >= 3:
            print("uct action = ", self.pomdp.actions[actionIndex], "reward=", immediate_reward, "state", nextState)

        state = nextState
        vnode = None
        if qnode.check_child_by_observation_index(observation):
            vnode = qnode.get_child_by_observation_index(observation)
        
        para_expand_count = 1
        
        # if (not vnode and (not done) and (qnode.n >= para_expand_count)):
        #     # vnode = POMCPNode()
        #     vnode = self.expand_node(state)
        #     vnode.h = observation
        #     vnode.parent = qnode
        #     qnode.add_child(vnode, observation)

        if (not vnode and (not done) and (qnode.n >= para_expand_count)):
            vnode = POMCPNode()
            self.expand(vnode, state)
            vnode.h = observation
            vnode.parent = qnode
            qnode.add_child(vnode, observation)

        if not done:
            self.TreeDepth += 1
            if vnode:
                delayed_reward += self.simulateV(state, vnode)
            else:
                rollout_rewad = self.rollout(state)
                delayed_reward += rollout_rewad
            self.TreeDepth -= 1
        else:
            total_reward += self.get_state_reward(state)
        total_reward += immediate_reward + self.gamma * delayed_reward
        qnode.increaseV(total_reward)
        return total_reward
    
    def is_current_belief_winning(self, vnode, time):
        if vnode == self.root:
            for state in self.root.belief:
                obs = self.get_observation(state)
                if not self.is_winning((obs, time)):
                    return False
        obs = vnode.getH()
        return self.is_winning((obs, time))  

    def is_action_index_shielded_for_node(self, parent, actionIndex): 
        if not self.pomdp.winning_obs:
            return False
        for state in parent.get_belief_support():
            for nxt_state in self.get_next_states(state, actionIndex):
                nxt_obs = self.get_observation(nxt_state)
                if not self.is_winning((nxt_obs, time+1)):
                    return True
        return False

    def get_next_states(self, state, actionIndex):
        if state not in self.pomdp.robot_state_action_map: return set()
        if actionIndex not in self.pomdp.robot_state_action_map[state]: return set()
        nxts =  self.pomdp.robot_state_action_map[state][actionIndex].keys()
        return nxts

    def is_winning(self, obs_time):
        return obs_time in self.pomdp.winning_obs

    def get_legal_actions(self, state):
        return set(self.pomdp.robot_state_action_map[state].keys())

    def step_reward(self, state, actionIndex):
        if state not in self.pomdp.state_action_reward_map:
            return float("-inf")
        if actionIndex not in self.pomdp.state_action_reward_map[state]:
            return float("-inf")
        return self.pomdp.state_action_reward_map[state][actionIndex] + self.pomdp.state_reward[state]
    
    def get_state_reward(self, state):
        return self.pomdp.state_reward[state]
    
    def get_random_action_index(self, state): # to be improved
        if self.pomdp.preferred_actions:
            return random.choice(self.pomdp.preferred_actions)
        available_action_index = self.pomdp.robot_state_action_map[state].keys()
        return random.choice(list(available_action_index))
    
    def rollout(self, state):
        total_reward = 0
        discount = 1
        done = False
        if self.verbose >= 3: print("starting rollout")
        numStep = 0
        remainTree = self.maxDepth - self.TreeDepth

        while (not done and numStep < remainTree):
            actionIndex = self.get_random_action_index(state)
            next_state = self.step(state, actionIndex)
            reward = self.step_reward(state, actionIndex)
            done = next_state in self.end_states
            if self.verbose >= 4:
                print("state", state, "action", self.actions[actionIndex], "reward", reward, "depth", numStep, "totalR", total_reward)
            total_reward += reward * discount
            discount *= self.gamma
            numStep += 1
            state = next_state
        """
        if done: print("Done")
        """
        total_reward += self.get_state_reward(state) * discount 
        return total_reward

    def greedyUCB(self, vnode, ucb):
        besta = []
        bestq = float("-inf")
        N = vnode.n
        logN = math.log(N + 1)
        children = vnode.children
        action_index_candidates = []
        for i in children:
            if i in vnode.illegalActionIndexes: continue
            action_index_candidates.append(i)
            # 			if (shieldLevel == ON_THE_FLY_SHIELD && vnode.isActionIndexIllegal(i)) {
            # //				System.out.println("shield level" + shieldLevel + " known illegal action " + allActions.get(i) +" for node " + vnode.getID() + " belief support" 	+ vnode.getBelief().getUniqueStatesInt());
            # 				continue;
            # 			}
            qnode = children[i]
            # print("-------",i, qnode)
            n = qnode.n
            if n == 0: return i
            q = qnode.getV()
            if ucb:
                q += self.fastUCB(N, n, logN)
            if q >= bestq:
                if q > bestq:
                    besta = []
                bestq = q
                besta.append(i)
#                   #//			if ( !ucb  && shieldLevel == 1  && isActionShieldedForNode(vnode, action) ) { // shiled only apply to the most up level
                # ////				System.out.println("shield Level = "+shieldLevel+ " Shielded Action = "  + action);
                # //				continue;
                # //			}
                # //			if (shieldLevel == 3 && vnode.isActionIllegal(action)) {
                # ////				System.out.println("shield level" + shieldLevel + " known illegal action" 
                # ////									+ action +" for node " + vnode.getID() + " belief support" 
                # ////									+ vnode.getBelief().getUniqueStatesInt());
                # //				continue;
                # //			}
                # //			if (shieldLevel == 3 && isActionShieldedForNode(vnode, action)) {
                # ////				System.out.println("shield level" + shieldLevel +" shielded action: " + action 
                # ////									+ "\n adding to illegal actions for node " + vnode.getID() 
                # ////									+ " belief support" +  vnode.getBelief().getUniqueStatesInt());
                # //				vnode.addIllegalActions(action);
                # //				continue;
                # //			}
        if besta:
            return random.choice(besta)
        else:
            if not action_index_candidates:
                return -1 if not ucb else random.choice(range(len(self.pomdp.actions)))
            actionIndex = random.choice(action_index_candidates)
            qParent = vnode.parent
            vParent = qParent.parent
            vParent.add_illegal_action_index(qParent.h)
            return actionIndex

    def get_action_index(self, action):
        if not self.action2Index:
            self.action2Index = {}
            for i, action in enumerate(self.actions):
                self.action2Index[action] = i
        return self.action2Index.get(action, -1)

def replay(H_default = -1):
    scene = './results/Obstacle-ETH-0-0-22-22/'
    setting = "shield_2-lookback_4-prediction_3-failure-0.1-agents-25-2024-02-06-16-59"
    episode = 54
    pkl_file = os.path.join(scene, setting, "Episode-{}.pkl".format(episode))
    question_step = 11
    scene_name  = 'ETH'

    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
    step = -1
    
    for d in data:
        if d.get("Action Step", - 1) == question_step:
            step = d
            break
    
    belief_support = step["Belief States"]
    belief = {state: 1/len(belief_support) for state in belief_support}
    
    pomdp = create_scenario(scene_name)
    pomdp.initial_belief = belief
    pomdp.preferred_actions = [0, 2]
    estimation_moving_agents_cur = step["Dynamic Agents Prediction"]
    constraints_cur_1 = step["Estimated Regions"]
    H = step["Predict Horizon"]
    if H_default > 0:
        H = H_default
    safe_distance = step["Safe Distance"]
    state_ground_truth = step["Robot State"]
    shield_level = step["Shield Level"]
    disallowed = step["Disallowed Actions"]
    shield_level = 0
    # print("disallowed", disallowed, state_ground_truth, shield_level)
    pomcp = POMCP(pomdp, shield_level, H,  constant = 170, maxDepth = 100, numSimulations = 2 ** 12)
    
    ACP_step = pomdp.build_restrictive_region(estimation_moving_agents_cur, constraints_cur_1, H, safe_distance)
    print("---==")
    print( ACP_step)
    print("---==")
    # ACP_step = defaultdict(list)
    obs_current_node = pomcp.get_observation(state_ground_truth)
    motion_mdp, AccStates, Avalid_states = pomcp.pomdp.compute_accepting_states() 
    observation_successor_map = pomcp.pomdp.compute_H_step_space(H)
    pomdp.online_compute_winning_region(obs_current_node, AccStates, Avalid_states, observation_successor_map, H, ACP_step)

    # print(ACP_step)
    # print(constraints_cur_1)

    # a, b = state_ground_truth
    # for h in ACP_step:
    #     # if h > 1: break
    #     for x, y in ACP_step[h]:
    #         dis = ((x-a) ** 2 + (y - b) ** 2) ** 0.5
    #         if dis < constraints_cur_1[h] + safe_distance:
    #             print(h, (x, y), (a, b), dis, constraints_cur_1[h], safe_distance)
    # # print(pomdp.obstacles)
    # # pomcp = POMCP(pomdp, 0, 3, 0)
    pomcp.reset_root()
    actionIndex = pomcp.select_action()
    print(pomcp.root.illegalActionIndexes, "___++")
    if actionIndex == -1:
        if pomcp.pomdp.preferred_actions:
            actionIndex = random.choice(pomcp.pomdp.preferred_actions)
        else:
            actionIndex = random.choice([idx for idx in range(len(pomcp.pomdp.actions))])
    print(actionIndex, "selected", pomcp.pomdp.actions[actionIndex])
    print(pomcp.root.illegalActionIndexes)
    return pomcp.pomdp.actions[actionIndex]
    # for state in pomcp.root.belief:
    #     for nxt in pomcp.pomdp.robot_state_action_map[state][2]:
    #         # print(nxt)
    #         for h in ACP_step:
    #             if nxt in ACP_step[h]:
    #                 print(state_ground_truth, state, nxt, h,  "should be shilded")
    # print()
    # print(pomcp.R_min, pomcp.R_max)
    
    #shieldLevel = 0, shieldHorizon = 5,  constant = 10, maxDepth = 100, numSimulations = 2 ** 12):

def test_const(scene_name,pomcp_numSimulation):
    pomdp = create_scenario(scene_name)
    # pomdp.preferred_actions = [] # ? do we need this
    print(pomdp.initial_belief)
    pomcp = POMCP(pomdp, shieldLevel=0, shieldHorizon=2,
                   constant=0, maxDepth = 200, gamma=0.95, numSimulations=pomcp_numSimulation)
    pomcp.reset_root()
    pomcp.select_action()
    print(pomcp.R_max, pomcp.R_min)

if __name__ == "__main__":
    replay()
    pass