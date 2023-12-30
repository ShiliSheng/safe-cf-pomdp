from MDP_TG.mdp import Motion_MDP_label, Motion_MDP, compute_accept_states
from MDP_TG.dra import Dra, Dfa, Product_Dra, Product_Dfa
from MDP_TG.vi4wr import syn_plan_prefix, syn_plan_prefix_dfa
from networkx.classes.digraph import DiGraph
import pickle
import time
import random 
import collections
import math
import copy
from collections import defaultdict
import numpy as np
import copy
# def print(*args, **kwargs):
#     return
class Model:
    def __init__(self, robot_nodes, actions, robot_edges, cost, initial_belief,
                 obstacles = [], targets = [], end_states = set(), state_reward = {}, preferred_actions = []):
        self.robot_nodes = robot_nodes # set of states
        self.actions = actions
        self.robot_edges = robot_edges
        self.cost = cost

        self.motion_mdp = Motion_MDP(self.robot_nodes, self.robot_edges, self.actions)
        self.derive_state_transion_map_from_mdp()

        self.initial_belief_support = list(initial_belief.keys())
        self.initial_belief = initial_belief
        self.obstacles = obstacles
        self.targets = targets
        self.end_states = end_states
        self.state_reward = state_reward
        self.preferred_actions = preferred_actions

        # will be set later for pomdp
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
                    robot_state_action_map[fnode][actionIndex] = {fnode:1}
        for fnode in state_action_reward_map:
            for actionIndex in range(len(self.actions)):
                if actionIndex not in state_action_reward_map[fnode]:
                    state_action_reward_map[fnode][actionIndex] = self.cost[actionIndex]

        self.robot_state_action_map = robot_state_action_map
        self.state_action_reward_map = state_action_reward_map
        
    def set_states_observations_with_predefined(self, state_observation_map, observation_state_map, obs_nodes):
        self.state_observation_map = state_observation_map
        self.observation_state_map = observation_state_map
        self.obs_nodes = obs_nodes
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
        AccStates = []
        for S_fi in Sf[0]:
            for MEC in S_fi:
                for sf in MEC:
                    if sf not in AccStates:
                        AccStates.append(tuple(sf))
        print('Number of satisfying states: %s' % len(AccStates))
        print(AccStates)

        f_accept_node = open('data/accept_node.dat','w')
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
        print( "-------", obstacle_new,)
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
                        self.observation_state_map[ws_node] = [ws_node]
                        self.state_observation_map[ws_node] = ws_node
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

        f_accept_observation = open('data/accept_observation.dat','w')
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

    def write_transitons(self):
        file = open('data/state_trainsiton.dat','w')
        for state in self.robot_state_action_map:
            for actionIndex in self.robot_state_action_map[state]:
                u = self.actions[actionIndex]
                file.write('%s,%s,%s\n'%(state, u, self.state_tra[actionIndex][state]))

    def write_state_observation(self):
        file = open('data/state_observation.dat','w')
        for state, obs in self.state_observation_map.items():
            file.write('%s,%s\n' %(state, obs))
        file.close()

    def write_observation_state(self):
        file = open('data/observation_state.dat','w')
        for obs, states in self.observation_state_map.items():
            file.write('%s,%s\n' %(obs, states))
        file.close()
   
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
        
    def build_restrictive_region(self, estimations, radius, H, safeDistance = 0):
        radius += safeDistance
        ACP = defaultdict(list)
        dx = 1
        dy = 1
        for tau in range(1, H + 1):
            for i in range(len(estimations[tau]) // 2):
                x, y = estimations[tau][i], estimations[tau][i+1]
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

    # def compute_H_step_space(self, motion_mdp, H):
    #     #Compute the H-step recahable support belief states, idea: o -> s -> s' -> o'
    #     motion_mdp = self.motion_mdp
    #     #----calculate H-step reachable set------------
    #     observation_successor_map = defaultdict(list)
    #     for o_node in self.obs_nodes:
    #         init_obs = set()
    #         init_obs.add(o_node)
    #         observation_successor_map[o_node, 0] = init_obs

    #     for o_node in self.obs_nodes:
    #         succ_obs = set()
    #         support_set = self.observation_state_map[o_node]
    #         for fnode in support_set:
    #             for tnode in motion_mdp.successors(fnode):
    #                 obs = self.state_observation_map[tnode]
    #                 if obs not in succ_obs:
    #                     succ_obs.add(obs)
    #         observation_successor_map[o_node, 1] = succ_obs

    #     if H > 1:
    #         for o_node in self.obs_nodes:
    #             for i in range(2, H+1):
    #                 succ_obs = observation_successor_map[o_node, i-1]
    #                 succ_step = set()
    #                 for oo_node in succ_obs:
    #                     for ooo_node in observation_successor_map[oo_node, 1]:
    #                         if ooo_node not in succ_step:
    #                             succ_step.add(ooo_node)
    #                 observation_successor_map[o_node, i] = succ_step
    #     return observation_successor_map

    # def set_states_observations(self, motion_mdp): 
    #     self.obs_nodes = set()
    #     for i in range(2, 20, 4):
    #         for j in range(2, 20, 4):
    #             onode = (i, j)
    #             self.obs_nodes.add(onode)

    #     for obs in self.obstacles:
    #         self.obs_nodes.add(obs)

    #     #----
    #     self.observation_state_map = dict()
    #     for o_node in self.obs_nodes:
    #         if o_node in self.obstacles:
    #             self.observation_state_map[o_node] = [o_node]
    #         else:
    #             ox = o_node[0]
    #             oy = o_node[1]
    #             support = set()
    #             for fnode in motion_mdp.nodes():
    #                 if fnode not in self.obstacles: 
    #                     fx = fnode[0]
    #                     fy = fnode[1]    
    #                     if (abs(fx-ox) <= 2) and (abs(fy-oy) <= 2):
    #                         state = fnode
    #                         support.add(state)
    #             self.observation_state_map[o_node] = support

    #     #----
    #     self.state_observation_map = dict()
    #     for fnode in motion_mdp.nodes(): 
    #         if fnode in self.obstacles:
    #             self.state_observation_map[fnode] = fnode
    #         else:
    #             fx = fnode[0]
    #             fy = fnode[1] 
    #             # support_obs = set()  
    #             for o_node in self.obs_nodes:
    #                 if o_node not in self.obstacles:
    #                     ox = o_node[0]
    #                     oy = o_node[1]  
    #                     if (abs(fx-ox) <= 2) and (abs(fy-oy) <= 2):
    #                         # support_obs.add(o_node)
    #                         self.state_observation_map[fnode] = o_node
    #                         break
    
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

def test_case1():
    U = actions = ['N', 'S', 'E', 'W', 'ST']
    C = cost = [3, 3, 3, 3, 1]

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

    minX, minY = float("inf"), float("inf")
    maxX, maxY = float("-inf"), float("-inf")
    robot_nodes = set()
    for i in range(1, 20, 2):
        for j in range(1, 20, 2):
            node = (i, j)
            robot_nodes.add(node) 
            minX = min(minX, i)
            minY = min(minY, j)
            maxX = max(maxX, i)
            maxY = max(maxY, j)

    targets = set([(maxX, maxY)])
    end_states = set([(maxX, maxY)])

    obstacles = set()
    random.seed(42)
    n_obstacles = 5
    robot_nodes_list = list(robot_nodes)
    while (len(obstacles) < n_obstacles):
        obstacle = random.choice(robot_nodes_list)
        if obstacle not in targets: 
            obstacles.add(obstacle)

    initial_belief_support = [(minX,minY)]
    initial_belief = {}
    for state in initial_belief_support:
        initial_belief[state] = 1 / len(initial_belief_support)   

    pomdp = Model(robot_nodes, actions, cost, WS_transition, transition_prob,
                    initial_belief, obstacles, targets, end_states)
    pomdp.set_transition_prob()
    motion_mdp, AccStates = pomdp.compute_accepting_states()

    obs_nodes = set()
    for i in range(2, 24, 4):
        for j in range(2, 24, 4):
            onode = (i, j)
            obs_nodes.add(onode)

    for obs in obstacles:
        obs_nodes.add(obs)

    state_observation_map = dict()
    for fnode in motion_mdp.nodes(): 
        if fnode in obstacles or fnode in targets:
            state_observation_map[fnode] = fnode
        else:
            fx = fnode[0]
            fy = fnode[1] 
            for o_node in obs_nodes:
                ox = o_node[0]
                oy = o_node[1]  
                if (abs(fx-ox) <= 2) and (abs(fy-oy) <= 2):
                    state_observation_map[fnode] = o_node
                    # break

    # define obversation-states
    observation_state_map = defaultdict(list)
    for state, obs in state_observation_map.items():
        observation_state_map[obs].append(state)
                    
    # pomdp.set_states_observations(motion_mdp)
    pomdp.set_states_observations_with_predefined(state_observation_map, observation_state_map, obs_nodes)

    H = 3 # Horizon
    observation_successor_map = pomdp.compute_H_step_space(motion_mdp, H)
    pomdp.set_states_observations(motion_mdp)
    pomdp.save_states()
    
    observation_state_map_default = copy.deepcopy(pomdp.observation_state_map)
    state_observation_map_default = copy.deepcopy(pomdp.state_observation_map)

    dfa = compute_dfa()

    #---Online planning starts----
    obs_current_node = pomdp.state_observation_map[initial_belief_support[0]]
    
    ACP_step = defaultdict(list)
    # ACP_step[1] =  [(5, 5), (5, 9)]
    # ACP_step[2] =  [(5, 5), (5, 9)]
    # ACP_step[3] =  [(5, 5), (5, 9)]
    # estimation = [[], [5, 5, 7, 3]]
    # estimation = [[],[1,3,4,5],[2,1.2,5.5,6.6],[]]
    # ACP_step = pomdp.build_restrictive_region(estimation, 1, 3)
    # print("ACP",ACP_step)

    obs_mdp, Winning_obs, A_valid, observation_state_map_change_record, state_observation_map_change_record  \
             = pomdp.online_compute_winning_region(obs_current_node, AccStates, observation_successor_map, H, ACP_step, dfa)

    print(Winning_obs)

    errorFree = True
    for tau in ACP_step:
        for x, y in ACP_step[tau]:
            obs = pomdp.state_observation_map[(x, y)]
            if (obs, tau) in Winning_obs:
                print((x, y), (obs, tau), "error!")
                errorFree = False
    print("Error Free:", errorFree)
    print(A_valid)
    # for state in initial_belief_support:
    #     for actionIndex in range(len(pomdp.actions)):
    #         print(actionIndex, pomdp.robot_state_action_map[state][actionIndex])
    #         for nxt in pomdp.robot_state_action_map[state][actionIndex]:
    #             print(nxt, pomdp.state_observation_map[nxt])

    # #----
    # pomdp.observation_state_map.clear()
    # pomdp.observation_state_map.update(observation_state_map_default)
    # pomdp.state_observation_map.clear()
    # pomdp.state_observation_map.update(state_observation_map_default)

# def create_scenario_obstacle_avoidance():
#     #----- states
#     dx = 1
#     dy = 1
#     robot_nodes = set()
#     mx = 24
#     my = 24
#     nx = ny = 2
#     mx_row = -1
#     mx_col = -1
#     for row in range(0, mx, nx):
#         mx_row = max(mx_row, row)
#         for col in range(0, my, ny):
#             mx_col = max(mx_col, col)
#             for i in range(nx):
#                 for j in range(ny):
#                     robot_nodes.add((row + i, col + j))

#     # define target
#     targets = set()
#     end_states = set()
#     for i in range(nx):
#         for j in range(ny):
#             targets.add((mx_row + i, mx_col + j))
#             end_states.add((mx_row + i, mx_col + j))
     
#     for target in targets:
#         print(target in robot_nodes, target)
    
#     # randomly define obstacles
#     random.seed(42)
#     n_obstacles = 5
#     obstacles = set()
#     robot_nodes_list = list(robot_nodes)
#     while (len(obstacles) < n_obstacles):
#         obstacle = random.choice(robot_nodes_list)
#         if obstacle not in targets: 
#             obstacles.add(obstacle)

    
#     # define initial belief
#     initial_belief_support = [(1,1)]
#     initial_belief = {}
#     for state in initial_belief_support:
#         initial_belief[state] = 1 / len(initial_belief_support)   

#     #----- actions
#     U = actions = ['N', 'S', 'E', 'W', 'ST']
#     C = cost = [3, 3, 3, 3, 1]

#     transition_prob = [[] for _ in range(len(actions))]
#     transition_prob[0] = [0.1, 0.8, 0.1] # S
#     transition_prob[1] = [0.1, 0.8, 0.1] # N
#     transition_prob[2] = [0.1, 0.8, 0.1] # E
#     transition_prob[3] = [0.1, 0.8, 0.1] # W
#     transition_prob[4] = [1]             # ST

#     WS_transition = [[] for _ in range(len(actions))]
#     WS_transition[0] = [(-1, 1), (0, 1), (1, 1)]       # S
#     WS_transition[1] = [(-1, -1), (0, -1), (1, -1)]    # N
#     WS_transition[2] = [(1, -1), (1, 0), (1, 1)]       # E
#     WS_transition[3] = [(-1, -1), (-1, 0), (-1, 1)]    # W
#     WS_transition[4] = [(0, 0)]                         # ST

#     # # define state-observation
#     # state_observation_map = defaultdict(tuple)
#     # for row in range(0, mx, nx):
#     #     for col in range(0, my, ny):
#     #         for i in range(nx):
#     #             for j in range(ny):
#     #                 state_observation_map[(row + i, col + j)] = (row, col)

#     # for obstacle in obstacles:
#     #     state_observation_map[obstacle] = obstacle
#     # for target in targets:
#     #     state_observation_map[target] = target
    
#     # # define obversation-states
#     # observation_state_map = defaultdict(list)
#     # obs_nodes = set()
#     # for state, obs in state_observation_map.items():
#     #     observation_state_map[obs].append(state)
#     #     obs_nodes.add(obs)

#     pomdp = Model(robot_nodes, actions, cost, WS_transition, transition_prob,
#                         initial_belief, obstacles, targets, end_states)
#     pomdp.set_transition_prob()

#     motion_mdp, AccStates = pomdp.compute_accepting_states()

#  # define state-observation
#     state_observation_map = defaultdict(tuple)
#     # for row in range(0, mx, nx):
#     #     for col in range(0, my, ny):
#     #         for i in range(nx):
#     #             for j in range(ny):
#     #                 state_observation_map[(row + i, col + j)] = (row, col)
#     for x, y in motion_mdp.nodes:
#         if (x, y) in obstacle or (x, y) in target:
#             state_observation_map[(x, y)] = (x, y)
#         else:
#             state_observation_map[(x, y)] = (x//2, y//2)
    
#     # define obversation-states
#     observation_state_map = defaultdict(list)
#     obs_nodes = set()
#     for state, obs in state_observation_map.items():
#         observation_state_map[obs].append(state)
#         obs_nodes.add(obs)
#     pomdp.set_states_observations_with_predefined(state_observation_map, observation_state_map, obs_nodes)
#     pomdp.save_states()
#     #### computing
#     return pomdp

# def obstacle_avoidance():
#     pomdp = create_scenario_obstacle_avoidance()
#     H = 3
#     motion_mdp, AccStates = pomdp.compute_accepting_states()
#     observation_successor_map = pomdp.compute_H_step_space(motion_mdp, H)
#     obs_current_node = pomdp.state_observation_map[pomdp.initial_belief_support[0]]

#     ACP_step = defaultdict(list)
#     # ACP_step[1] =  [(5, 5), (5, 9)]
#     # ACP_step[2] =  [(5, 5), (5, 9)]
#     # ACP_step[3] =  [(5, 5), (5, 9)]

#     dfa = compute_dfa()  
#     obs_mdp, Winning_obs, A_valid, observation_state_map_change_record, state_observation_map_change_record  \
#              = pomdp.online_compute_winning_region(obs_current_node, AccStates, observation_successor_map, H, ACP_step, dfa)
#     print(Winning_obs)
#     pomdp.display_state_observation()
#     pomdp.display_observation_state()
#     print(pomdp.obstacles)
#     # for key, val in pomdp.robot_state_action_map.items():
#     #     print("ke ",key, val)
#     print(pomdp.target)

def create_scenario_obstacle():
    # ------- states
    minX, minY = float("inf"), float("inf")
    maxX, maxY = float("-inf"), float("-inf")
    robot_nodes = set()
    for i in range(0, 20, 1):
        for j in range(0, 20, 1):
            node = (i, j)
            robot_nodes.add(node) 
            minX = min(minX, i)
            minY = min(minY, j)
            maxX = max(maxX, i)
            maxY = max(maxY, j)

    targets = set([(maxX, maxY)])
    end_states = set([(maxX, maxY)])

    obstacles = set()
    random.seed(42)
    n_obstacles = 5
    robot_nodes_list = list(robot_nodes)
    while (len(obstacles) < n_obstacles):
        obstacle = random.choice(robot_nodes_list)
        if obstacle not in targets: 
            obstacles.add(obstacle)

    # ------- action transition
    U = actions = ['N', 'S', 'E', 'W', 'ST']
    C = cost = [-1, -1, -1, -1, -3]
    preferred_actions = [0, 2]
    slippery_prob = 0.2
    transition_prob = [[slippery_prob, 1 - slippery_prob]   for _ in range(len(actions))]
    WS_transition = [[] for _ in range(len(actions))]

    WS_transition[0] = [(0, 2), (0, 1)]       # S
    WS_transition[1] = [(0, -2), (0, -1)]     # N
    WS_transition[2] = [(2, 0), (1, 0)]       # E
    WS_transition[3] = [(-2, 0), (-1, 0)]     # W
    transition_prob[4] = [1]                            # ST
    WS_transition[4] = [(0, 0)]                         # ST

    robot_edges = dict()
    for fnode in robot_nodes: 
        for actionIndex, action in enumerate(actions):
            u = actions[actionIndex]
            c = cost[actionIndex]
            n_next_positions = len(WS_transition[actionIndex])
            for i in range(n_next_positions):
                dx, dy = WS_transition[actionIndex][i]
                prob = transition_prob[actionIndex][i]
                tnode = (fnode[0] + dx, fnode[1] + dy)
                if tnode in robot_nodes:
                    robot_edges[(fnode, u, tnode)] = (prob, c)

    initial_belief_support = [(minX,minY)]
    initial_belief_support = [(0, 0)]
    initial_belief = {}
    for state in initial_belief_support:
        initial_belief[state] = 1 / len(initial_belief_support)   

    state_reward = defaultdict(int)
    for state in targets:
        state_reward[state] = 10000

    pomdp = Model(robot_nodes, actions, robot_edges, cost,
                    initial_belief, obstacles, targets, end_states, state_reward, preferred_actions)
    motion_mdp, AccStates = pomdp.compute_accepting_states()

    # set state-observation
    state_observation_map = dict()
    for fx, fy in motion_mdp.nodes(): 
        if (fx, fy) in obstacles or (fx, fy) in targets:
            state_observation_map[(fx, fy)] = (fx, fy)
        else:
            state_observation_map[(fx, fy)] = (fx//4, fy//4)

    # define obversation-states
    observation_state_map = {}
    for state, obs in state_observation_map.items():
        if obs not in observation_state_map: observation_state_map[obs] = []
        observation_state_map[obs].append(state)
    
    obs_nodes = set(key for key in observation_state_map)

    pomdp.set_states_observations_with_predefined(state_observation_map, observation_state_map, obs_nodes)
    pomdp.save_states()
    return pomdp

def test_ostacle():
    pomdp = create_scenario_obstacle()
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
    print(Winning_obs)
    print(pomdp.obstacles)
    print(pomdp.targets)
    print(pomdp.state_reward)
    for obstacle in pomdp.obstacles:
        for i in range(1, H+1):
            if ((obstacle), i ) in Winning_obs:
                print("error static obstacle in Winning Region", (obstacle, i))
    
    for i in range(1, H+1):
        for obstacle in ACP_step[i]:
            if ((obstacle), i ) in Winning_obs:
                print("error dynamic obstacle in Winning Region", (obstacle, i))
    
    pomdp.write_state_observation()

def create_scenario_refuel():
    U = actions = ['N', 'S', 'E', 'W', 'ST', "Refuel"]
    preferred_actions = [0, 2]
    slippery_prob = 0.2

    transition_prob = [[slippery_prob, 1 - slippery_prob] for _ in range(len(actions))]
    WS_transition = [[] for _ in range(len(actions))]
    WS_transition[0] = [(0, 2, -1), (0, 1, -1)]       # S
    WS_transition[1] = [(0, -2, -1), (0, -1, -1)]     # N
    WS_transition[2] = [(2, 0, -1), (1, 0, -1)]       # E
    WS_transition[3] = [(-2, 0, -1), (-1, 0, -1)]     # W
    transition_prob[4] = [1]                            # ST
    WS_transition[4] = [(0, 0, -1)]                         # ST

    minX, minY = float("inf"), float("inf")
    maxX, maxY = float("-inf"), float("-inf")
    robot_nodes = set()
    locations = []
    max_energy = 9
    for i in range(0, 20, 1):
        for j in range(0, 20, 1):
            for energy in range(0, max_energy, 1):
                node = (i, j, energy)
                robot_nodes.add(node) 
                locations.append((i, j))
                minX = min(minX, i)
                minY = min(minY, j)
                maxX = max(maxX, i)
                maxY = max(maxY, j)

    for energy in range(max_energy):
        targets = set([(maxX, maxY, energy)])
        end_states = set([(maxX, maxY, energy)])

    obstacles = set()
    random.seed(42)
    n_obstacles = 0
    while (len(obstacles) < n_obstacles):
        obstacle = random.choice(locations)
        if obstacle not in targets: 
            obstacles.add(obstacle)

    stations = set()
    random.seed(42)
    n_stations = 5
    while (len(stations) < n_stations):
        station = random.choice(locations)
        if station not in targets: 
            stations.add(station)

    robot_edges = {} # TODO
    for i, j, energy in robot_nodes:
        for action_index, action in enumerate(actions):
            if action != "Refuel":
                if energy == 0:
                    nxt_state = {(i, j, 0): 1}
                elif action_index == 0:
                    nxt_state = {(i, j + 2, energy - 1):slippery_prob, (i, j + 1, energy - 1): (1-slippery_prob)}
                elif action_index == 1:
                    nxt_state = {(i, j - 2, energy - 1):slippery_prob, (i, j - 1, energy - 1): (1-slippery_prob)}
                elif action_index == 2:
                    nxt_state = {(i + 2, j, energy - 1):slippery_prob, (i + 1, j, energy - 1): (1-slippery_prob)}
                elif action_index == 3:
                    nxt_state = {(i - 2, j, energy - 1):slippery_prob, (i - 1, j, energy - 1): (1-slippery_prob)}
                elif action_index == 4:
                    nxt_state = {(i, j, energy - 1):slippery_prob, (i, j, energy - 1): (1-slippery_prob)}
            else:
                if (i, j) in stations:
                    nxt_state = {(i, j, max_energy): 1}
                else:
                    nxt_state = {(i, j, max(energy - 1, 0)): 1}

    initial_belief_support = [(minX, minY, max_energy)]
    initial_belief = {}
    for state in initial_belief_support:
        initial_belief[state] = 1 / len(initial_belief_support)   

    # # ------- state award
    C = cost = [-1, -1, -1, -1, -3, -1]
    state_reward = defaultdict(int)
    for state in targets:
        state_reward[state] = 10000

    # TODO
    pomdp = Model(robot_nodes, actions, robot_edges,  cost, 
                    initial_belief, obstacles, targets, end_states, state_reward, preferred_actions)
    pomdp.set_transition_prob() 
    motion_mdp, AccStates = pomdp.compute_accepting_states()

    # # set state-observation
    # TODO
    state_observation_map = dict()
    for fx, fy, energy in motion_mdp.nodes(): 
        if (fx, fy) in obstacles or (fx, fy) in targets:
            state_observation_map[(fx, fy)] = (fx, fy, energy // 2)
        elif (fx == 0 or fy == 0 or fx == maxX or fy == maxY):
            state_observation_map[(fx, fy)] = (fx, fy, energy // 2)
        else:
            state_observation_map[(fx, fy)] = (fx//4, fy//4, energy // 2)

    # define obversation-states
    observation_state_map = {}
    for state, obs in state_observation_map.items():
        if obs not in observation_state_map: observation_state_map[obs] = []
        observation_state_map[obs].append(state)
    
    obs_nodes = set(key for key in observation_state_map)

    pomdp.set_states_observations_with_predefined(state_observation_map, observation_state_map, obs_nodes)
    pomdp.save_states()
    return pomdp

def create_scenario_rock():
    pass

if __name__ == "__main__":
    test_ostacle()
    # obstacle_avoidance()
    pass
