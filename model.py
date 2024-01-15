from MDP_TG.mdp import Motion_MDP_label, Motion_MDP, compute_accept_states
from MDP_TG.dra import Dra, Dfa, Product_Dra, Product_Dfa
from MDP_TG.vi4wr import syn_plan_prefix, syn_plan_prefix_dfa
from networkx.classes.digraph import DiGraph
import pickle
import time
import random 
import os
import collections
import math
import copy
from collections import defaultdict
import numpy as np
import copy
import matplotlib.pyplot as plt
def print(*args, **kwargs):
    return

class Model:
    def __init__(self, robot_nodes, actions, robot_edges, cost, initial_belief,
                targets, end_states, minX, minY, maxX, maxY,  model_name,
                state_reward = {}, preferred_actions = [], obstacles = [],
                refule_stations = set(),  rocks = set(), unsafe_states = set()):
        self.robot_nodes = robot_nodes # set of states
        self.actions = actions
        self.robot_edges = robot_edges
        self.cost = cost

        self.motion_mdp = Motion_MDP(self.robot_nodes, self.robot_edges, self.actions)
        self.derive_state_transion_map_from_mdp()
        self.initial_belief_support = list(initial_belief.keys())
        self.initial_belief = initial_belief
        
        self.targets = targets
        self.end_states = end_states
        self.minX = minX
        self.minY = minY
        self.maxX = maxX
        self.maxY = maxY
        self.model_name = 'Obstacle-' + model_name + "-{}-{}-{}-{}".format(minX, minY, maxX, maxY)

        self.state_reward = state_reward
        self.preferred_actions = preferred_actions

        self.obstacles = obstacles  
        self.refule_stations = refule_stations
        self.rocks = rocks # [(x, y)] 
        self.unsafe_states = unsafe_states

        # will be set later for pomcp
        self.state_observation_map = defaultdict(tuple)
        self.observation_state_map = defaultdict(list)

        # will be computed online
        self.winning_obs = set() 

    def derive_state_transion_map_from_mdp(self):
        # self.robot_state_action_map = robot_state_action_map       # (state, actionIndex) : {next_state: prob}
        # self.state_action_reward_map = state_action_reward_map       # (state, actionIndex) : cost
        robot_state_action_map = defaultdict(dict)
        state_action_reward_map = defaultdict(dict)
        action_to_index = {action: actionIndex for actionIndex, action in enumerate(self.actions)}
        for fnode in self.motion_mdp.nodes():
            # if (fnode != (8, 0)): continue
            for tnode in self.motion_mdp.successors(fnode):
                record = self.motion_mdp[fnode][tnode]['prop']
                # print(fnode, record)
                for actionTuple, (prob, cost) in record.items():
                    action = "".join(actionTuple)
                    actionIndex = action_to_index[action]
                    succ_prop = {tnode: prob}
                    if actionIndex not in robot_state_action_map[fnode]: robot_state_action_map[fnode][actionIndex] = {} 
                    robot_state_action_map[fnode][actionIndex].update(succ_prop)
                    state_action_reward_map[fnode][actionIndex] = cost
                # print(fnode, tnode,record)

        for fnode in robot_state_action_map:
            for actionIndex in range(len(self.actions)):
                if actionIndex not in robot_state_action_map[fnode]:
                    action = self.actions[actionIndex]
                    robot_state_action_map[fnode][actionIndex] = {fnode:1}
        for fnode in state_action_reward_map:
            for actionIndex in range(len(self.actions)):
                if actionIndex not in state_action_reward_map[fnode]:
                    state_action_reward_map[fnode][actionIndex] = self.cost[actionIndex]

        self.robot_state_action_map = robot_state_action_map
        self.state_action_reward_map = state_action_reward_map
        
    def set_states_observations_with_predefined(self, state_observation_map):
        self.state_observation_map = state_observation_map
        # define obversation-states
        self.observation_state_map = {}
        for state, obs in state_observation_map.items():
            if obs not in self.observation_state_map: self.observation_state_map[obs] = []
            self.observation_state_map[obs].append(state)
        self.obs_nodes = set(key for key in self.observation_state_map)
        self.save_states()

    def save_states(self):
        self.observation_state_map_default = copy.deepcopy(self.observation_state_map) 
        self.state_observation_map_default = copy.deepcopy(self.state_observation_map)

    def compute_accepting_states(self):
        #compute safe space with respect to static enviornment
        motion_mdp = self.motion_mdp
        self.successor_mdp = dict()
        for node in motion_mdp.nodes():
            self.successor_mdp[node]= motion_mdp.successors(node)

        Sf = compute_accept_states(motion_mdp, self.obstacles, self.targets)
        # prob 1 to reach target and avoid obstacles

        # print("Sf------------")
        AccStates = []
        for S_fi in Sf[0]:
            for MEC in S_fi:
                for sf in MEC:
                    if sf not in AccStates:
                        AccStates.append(tuple(sf))
        print('Number of satisfying states: %s' % len(AccStates))
        # print(AccStates)

        f_accept_node = open('pomdp_states/accept_node.dat','w')
        for nd_id, nd in enumerate(AccStates):
            #print(nd)
            #(ts_node_x, ts_node_y)
            f_accept_node.write('%s,%s\n' %(nd[0], nd[1]))
        f_accept_node.close()
        self.motion_mdp = motion_mdp
        return motion_mdp, AccStates
        
    def online_compute_winning_region(self, obs_initial_node, AccStates, observation_successor_map, H, ACP, dfa = []):
        #--------------ONLINE-------------------------
        # Build the N-step reachable support belief MDP, the target set for the support belief MDP is given by AccStates (which is computed offline)
        # ACP_step: computed adaptive conformal prediction constraints
        U = self.actions
        C = self.cost
        obstacle_static = set(self.obstacles)
        obstacle_new = dict()
        for i in range(H):
            obstacle_new[i+1] = obstacle_static.union(ACP[i+1])
        # print( "------- obstacle_new", obstacle_new,)
        observation_state_map_change_record = set()
        state_observation_map_change_record = set()
        #----add time counter----
        SS = dict()
        observation_target = set()
        observation_obstacle = set()
        obs_initial_node_count = (obs_initial_node, 0)
        H_step_obs = observation_successor_map[obs_initial_node, H]
        obs_nodes_reachable = dict()
        obs_nodes_reachable[obs_initial_node_count] = {frozenset(['target']): 1.0}
        for oc in range(1, H+1):
            for o_node in self.obs_nodes:
                onode_count = (o_node, oc)
                support_set = set(self.observation_state_map[o_node])
                SS[oc] = support_set.intersection(obstacle_new[oc])
                if len(SS[oc]) > 0:
                    o_node_not_obstacle = support_set.difference(SS[oc])
                    for ws_node in SS[oc]:
                        obs_nodes_reachable[(ws_node, oc)] = {frozenset(['obstacle']): 1.0}
                        observation_state_map_change_record.add(ws_node)
                        state_observation_map_change_record.add(ws_node)
                        self.observation_state_map[ws_node] = [ws_node]     # 
                        self.state_observation_map[ws_node] = ws_node       #
                        ws_obstacle = (ws_node, oc)
                        observation_obstacle.add(ws_obstacle)

                    if len(o_node_not_obstacle) > 0:
                        observation_state_map_change_record.add(o_node)
                        self.observation_state_map[o_node] = o_node_not_obstacle
                        if o_node_not_obstacle.issubset(set(AccStates)):
                            obs_nodes_reachable[onode_count] = {frozenset(['target']): 1.0}
                            observation_target.add(onode_count)
                        else:
                            obs_nodes_reachable[onode_count] = {frozenset(): 1.0}  
                elif support_set.issubset(set(AccStates)):
                    obs_nodes_reachable[onode_count] = {frozenset(['target']): 1.0}
                    observation_target.add(onode_count)
                else:
                    obs_nodes_reachable[onode_count] = {frozenset(): 1.0}
        print('Number of target observation states: %s' %len(observation_target))
        print('Number of obstacle observation states: %s' %len(observation_obstacle))

        obs_initial_dict = obs_nodes_reachable[obs_initial_node_count]
        obs_initial_label = obs_initial_dict.keys()
        
        obs_edges = dict()
        for o_node in obs_nodes_reachable.keys():
            oc = o_node[1]
            support_set = list(self.observation_state_map[o_node[0]])
            #print(support_set)
            for node in support_set:  
                for k, u in enumerate(U):
                    tnode_set = self.robot_state_action_map[node][k]
                    for ttnode in list(tnode_set.keys()):
                        t_obs = self.state_observation_map[ttnode]
                        if oc < H: 
                            tnode = (t_obs, oc+1)
                        else:
                            tnode = (t_obs, oc)
                        if tnode in obs_nodes_reachable:  
                            obs_edges[(o_node, u, tnode)] = (1, C[k])

        obs_mdp = Motion_MDP_label(obs_nodes_reachable, obs_edges, U, obs_initial_node_count, obs_initial_label)

        #----
        self.successor_obs_mdp = dict()
        for node in obs_mdp:
            self.successor_obs_mdp[node]= obs_mdp.successors(node)

        #----
        Winning_obs = observation_target
        A_valid = dict()
        for s in Winning_obs:
            A_valid[s] = obs_mdp.nodes[s]['act'].copy()
            if not A_valid[s]:
                print("Isolated state")

        for s in Winning_obs:
            U_to_remove = set()
            for u in A_valid[s]:
                for t in obs_mdp.successors(s):
                    if ((u in list(obs_mdp[s][t]['prop'].keys())) and (t not in Winning_obs)):
                        U_to_remove.add(u)
            A_valid[s].difference_update(U_to_remove)
        print('Number of winning states in observation space: %s' % len(Winning_obs))

        f_accept_observation = open('pomdp_states/accept_observation.dat','w')
        for nd_id, nd in enumerate(Winning_obs):
            # ts_node_id, ts_node_x, ts_node_y, ts_node_d
            f_accept_observation.write('%s,%s,%s\n' %(nd[0], nd[1], A_valid[nd]))
        f_accept_observation.close()

        self.winning_obs = Winning_obs
        return obs_mdp, Winning_obs, A_valid, observation_state_map_change_record, state_observation_map_change_record

    def compute_H_step_space(self, H):
        #Compute the H-step recahable support belief states, idea: o -> s -> s' -> o'
        motion_mdp = self.motion_mdp
        #----calculate H-step reachable set------------
        observation_successor_map = defaultdict(list)
        for o_node in self.obs_nodes:
            init_obs = set()
            init_obs.add(o_node)
            observation_successor_map[o_node, 0] = init_obs

        for o_node in self.obs_nodes:
            succ_obs = set()
            support_set = self.observation_state_map[o_node]
            for fnode in support_set:
                for tnode in motion_mdp.successors(fnode):
                    obs = self.state_observation_map[tnode]
                    if obs not in succ_obs:
                        succ_obs.add(obs)
            observation_successor_map[o_node, 1] = succ_obs

        if H > 1:
            for o_node in self.obs_nodes:
                for i in range(2, H+1):
                    succ_obs = observation_successor_map[o_node, i-1]
                    succ_step = set()
                    for oo_node in succ_obs:
                        for ooo_node in observation_successor_map[oo_node, 1]:
                            if ooo_node not in succ_step:
                                succ_step.add(ooo_node)
                    observation_successor_map[o_node, i] = succ_step
        return observation_successor_map

    def restore_states_from_change(self, observation_state_map_change_record, state_observation_map_change_record):
        for key in observation_state_map_change_record:
            if key in self.observation_state_map_default:
                self.observation_state_map[key] = self.observation_state_map_default[key]
            else:
                self.observation_state_map.pop(key, None)
        
        for key in state_observation_map_change_record:
            if key in self.state_observation_map_default:
                self.state_observation_map[key] = self.state_observation_map_default[key]
            else:
                self.state_observation_map.pop(key, None)

    def restore_states_from_default(self):
        self.observation_state_map = copy.deepcopy(self.observation_state_map_default)
        self.state_observation_map = copy.deepcopy(self.state_observation_map_default)

    def write_model(self):
        file = open('./pomdp_states/state_trainsiton.dat','w')
        for state in self.robot_state_action_map:
            for actionIndex in self.robot_state_action_map[state]:
                u = self.actions[actionIndex]
                file.write('%s,%s,%s\n'%(state, u, self.robot_state_action_map[state][actionIndex]))

        file = open('./pomdp_states/state_observation.dat','w')
        for state, obs in self.state_observation_map.items():
            file.write('%s,%s\n' %(state, obs))
        file.close()

        file = open('./pomdp_states/observation_state.dat','w')
        for obs, states in self.observation_state_map.items():
            file.write('%s,%s\n' %(obs, states))
        file.close()
   
    def plot_map(self):
        minX, minY, maxX, maxY = self.minX, self.minY, self.maxX, self.maxY
        belief = self.initial_belief_support
        targets = self.targets
        obstacles = self.obstacles
        model_name = self.model_name
        fig, ax = plt.subplots()   
        ax.set_aspect('equal')
        width = height = 1
        plt.xlim(minX-0.5, maxX+0.5)
        plt.ylim(minY-0.5, maxY+0.5)
        for x, y in belief:
            rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, facecolor= "blue", alpha = 0.5)
            ax.add_patch(rect)
            # plt.scatter(x, y, marker = 'H', alpha=0.5, color = "black")
        for x, y in targets:
            rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, facecolor= "green", alpha = 1)
            ax.add_patch(rect)
            # plt.scatter(x, y, marker = '*', alpha = 1, color = "green")
        for x, y in obstacles:
            rect = plt.Rectangle((x - 0.5, y - 0.5), width, height, facecolor= "red", alpha = 1)
            ax.add_patch(rect)
            # plt.scatter(x, y, marker = 's', alpha = 1, color = "red")
        
        figure_path = os.path.join("./results/", model_name )
        if not os.path.exists(figure_path):
            os.makedirs(figure_path)
        figure_file = os.path.join(figure_path, "map.png")
        plt.savefig(figure_file,  transparent=True,  bbox_inches = "tight", dpi = 300) 
        plt.show()
    def find_next_states(self, state):
        queue = collections.deque([state])
        visited = set([state])
        while queue:
            print(queue, "n")
            for _ in range(len(queue)):
                s = queue.popleft()
                for actionIndex in self.robot_state_action_map[s]:
                    for nxt in self.robot_state_action_map[s][actionIndex]:
                        if nxt in visited: continue
                        queue.append(nxt)
                        visited.add(nxt)
        
    def build_restrictive_region(self, estimations, radius_set, H, safeDistance = 0):
        ACP = defaultdict(list)
        dx = 1
        dy = 1
        for tau in range(1, H + 1):
            radius = radius_set[tau] + safeDistance
            for i in range(len(estimations[tau]) // 2):
                x, y = estimations[tau][2 * i], estimations[tau][2 * i+1]
                lx, rx = math.floor(x - radius), math.ceil(x + radius)
                by, uy = math.floor(y - radius), math.ceil(y + radius)
                for nx in np.arange(lx, rx, dx):
                    for ny in np.arange(by, uy, dy):
                        if (nx, ny) in self.robot_state_action_map and (nx-x)**2 + (ny-y)** 2 < radius ** 2:
                            ACP[tau].append((nx, ny))
        return ACP


    # def set_transition_prob(self):
        for fnode in self.robot_nodes: 
            for actionIndex, action in enumerate(self.actions):
                u = self.actions[actionIndex]
                c = self.cost[actionIndex]
                cumulative_prob = 0
                succ_set = dict()
                n_next_positions = len(self.transiton[actionIndex])
                for i in range(n_next_positions):
                    dx, dy = self.transiton[actionIndex][i]
                    prob = self.transition_prob[actionIndex][i]
                    tnode = (fnode[0] + dx, fnode[1] + dy)
                    if tnode in self.robot_nodes:
                        cumulative_prob += prob
                        self.robot_edges[(fnode, u, tnode)] = (prob, c)
                        succ_prop = {tnode: prob}
                        succ_set.update(succ_prop)
                if not succ_set:     # if no successor, stay the same state
                    succ_set[fnode] = 1
                else:                # make prob sum to 1
                    for tnode in succ_set:
                        succ_set[tnode] += (1 - cumulative_prob) / len(succ_set)
                if fnode not in self.state_action_reward_map: self.state_action_reward_map[fnode] = {}
                self.state_action_reward_map[fnode][actionIndex] = c
                if fnode not in self.robot_state_action_map: self.robot_state_action_map[fnode] = {}
                self.robot_state_action_map[fnode][actionIndex] = succ_set
                self.state_tra[actionIndex][fnode] = succ_set
  
def compute_dfa():
    #----compute DFA----
    #reach_avoid = '! obstacle U target'
    statenum = 3
    init = 1 
    edges = {(1, 1): ['00'], 
            (1, 2): ['01' '11'], 
            (1, 3): ['10'], 
            (2, 2): ['00', '01', '10', '11'],
            (3, 3): ['00', '01', '10', '11'], 
            }
    aps = ['obstacle', 'target']
    acc = [[{2}]]
    dfa = Dfa(statenum, init, edges, aps, acc)
    print('DFA done.')
    return dfa

def create_scenario_obstacle(minX, minY, maxX, maxY, 
                             initial_belief_support,
                             targets, end_states,
                             obstacles, model_name, preferred_actions):
    # 
    # ------- states
    robot_nodes = set()
    for i in range(minX, maxX + 1, 1):
        for j in range(minY, maxY + 1, 1):
            node = (i, j)
            robot_nodes.add(node) 

    # ------- action transition
    U = actions = ['N', 'S', 'E', 'W']
    C = cost = [-1, -1, -1, -1]
    
    slippery_prob = 0.2
 
    robot_edges = {}
    for fnode in robot_nodes: 
        i, j = fnode
        for action_index, action in enumerate(actions):
            action_cost = cost[action_index]
            if action_index == 0:
                nxt_states = {(i, j + 2):slippery_prob, (i, j + 1): (1-slippery_prob)}
            elif action_index == 1:
                nxt_states = {(i, j - 2):slippery_prob, (i, j - 1): (1-slippery_prob)}
            elif action_index == 2:
                nxt_states = {(i + 2, j):slippery_prob, (i + 1, j): (1-slippery_prob)}
            elif action_index == 3:
                nxt_states = {(i - 2, j):slippery_prob, (i - 1, j): (1-slippery_prob)}
            
            for tnode, prob in nxt_states.items():
                if prob == 0 or tnode not in robot_nodes: continue
                robot_edges[(fnode, action, tnode)] = (prob, action_cost)


    initial_belief = {}
    for state in initial_belief_support:
        initial_belief[state] = 1 / len(initial_belief_support)   

    state_reward = defaultdict(int)
    for state in targets:
        state_reward[state] += 1000
    for state in obstacles:
        state_reward[state] += -10

    pomdp = Model(robot_nodes, actions, robot_edges, cost,
                    initial_belief, targets, end_states, 
                    minX, minY, maxX, maxY, model_name,
                    state_reward, preferred_actions, obstacles)
    
    motion_mdp, AccStates = pomdp.compute_accepting_states()

    # set state-observation
    state_observation_map = dict()
    for fx, fy in motion_mdp.nodes():
        if (fx, fy) in obstacles:
            state_observation_map[(fx, fy)] = (fx, fy) 
        if (fx, fy) in targets:
            state_observation_map[(fx, fy)] = (10000007, 10000007)
        else:
            state_observation_map[(fx, fy)] = (10000 + fx//4, 10000 + fy//4)
    
    # set state->obs, obs->state, obs()
    pomdp.set_states_observations_with_predefined(state_observation_map )

    return pomdp

def create_scenario(scene):
    # U = actions = ['N', 'S', 'E', 'W']
    if scene == 'ETH':
        minX, minY, maxX, maxY = 0, 0, 22 , 22
        initial_belief_support = [(0, 0)]
        targets = set()
        targets.add((20, 22))
        end_states = set(list(targets))
        obstacles = set()
        for x in range(6, maxX - 5):
            obstacles.add((x, 3))
            obstacles.add((x, 19))
        
        for x in [5, maxX - 5]:
            for y in range(0, 4):
                obstacles.add((x, y))
                obstacles.add((x, maxY - y))
            
        for y in range(7, 16, 4):
            obstacles.add((5, y))
            obstacles.add((maxX-5, y))

        preferred_actions = [0, 2]

    if scene == 'SDD-bookstore-video1':
        minX, minY, maxX, maxY = 0, 0, 60, 60
        initial_belief_support = [(12, 60)]
        targets = set([(60, 0), (59, 0), (60,1), (59, 1)])
        end_states = set(list(targets))
        obstacles = set()
        for x in range(3, 8, 1):
            for y in range(3, 8, 1):
                obstacles.add((x, y))
            for y in range(25, 34, 1):
                obstacles.add((x, y))
        for x in range(0, 8, 1):
            for y in range(40, 60+1, 1):
                obstacles.add((x, y))
        
        for y in range(8, 60, 8):
            end = 45 if y < 40 else 60
            for x in range(25, end + 1):
                obstacles.add((x, y))

        for y in range(56, 61):
            for x in range(25, 61):
                obstacles.add((x, y))
        preferred_actions = [1, 2]

    pomdp = create_scenario_obstacle(minX, minY, maxX, maxY, 
                             initial_belief_support,
                             targets, end_states,
                             obstacles, scene, preferred_actions)
    print("==", pomdp.obstacles)
    pomdp.write_model()
    return pomdp

def test_scenario(pomdp):
    H = 3
    motion_mdp, AccStates = pomdp.compute_accepting_states()
    observation_successor_map = pomdp.compute_H_step_space(H)

    obs_current_node = pomdp.state_observation_map[pomdp.initial_belief_support[0]]

    ACP_step = defaultdict(list)
    # ACP_step[1] =  [(5, 5), (5, 9)]
    # ACP_step[2] =  [(5, 5), (5, 9)]
    # ACP_step[3] =  [(5, 5), (5, 9)]
    dfa = compute_dfa()  
    obs_mdp, Winning_obs, A_valid, observation_state_map_change_record, state_observation_map_change_record  \
             = pomdp.online_compute_winning_region(obs_current_node, AccStates, observation_successor_map, H, ACP_step, dfa)
    print("+++++++++ Winning Obs")
    print(Winning_obs)
    print("obstacle states", pomdp.obstacles)
    print(pomdp.targets)
    print(pomdp.state_reward)
    for obstacle in pomdp.obstacles:
        for i in range(1, H+1):
            obs_obstacle = pomdp.state_observation_map.get(obstacle, (-1, -1))
            if ((obs_obstacle), i ) in Winning_obs:
                print("error static obstacle in Winning Region", (obstacle, i))

    for i in range(1, H+1):
        for obstacle in ACP_step[i]:
            if ((obstacle), i ) in Winning_obs:
                print("error dynamic obstacle in Winning Region", (obstacle, i))

if __name__ == "__main__":
    # pomdp = create_scenario('ETH')
    pomdp = create_scenario("SDD-bookstore-video1")
    # test_scenario(pomdp)
    pomdp.plot_map()
    # pomdp = create_scenario_obstacle()
    # pomdp.write_model()
    # print(pomdp.obstacles)
    # create_scenario_refuel()
    # create_scenario_rock()
    pass
