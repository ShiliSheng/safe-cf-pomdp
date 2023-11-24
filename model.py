from MDP_TG.mdp import Motion_MDP_label, Motion_MDP, compute_accept_states
from MDP_TG.dra import Dra, Dfa, Product_Dra, Product_Dfa
from MDP_TG.vi4wr import syn_plan_prefix, syn_plan_prefix_dfa
from networkx.classes.digraph import DiGraph
import pickle
import time
import random 

class Model:
    def __init__(self, robot_nodes, actions, cost, transition, transiton_prob, initial_belief,
                 obstacles = [], target = [], end_states = set()):
        self.t0 = time.time()
        self.robot_nodes = robot_nodes # set of states
        self.actions = actions
        self.cost = cost
        self.transiton = transition
        self.transition_prob = transiton_prob

        self.initial_belief = initial_belief
        self.initial_belief_support = list(initial_belief.keys())
        self.obstacles = obstacles
        self.target = target
        self.state_tra = [{} for _ in range(len(self.actions))]
        self.robot_edges = dict()
        self.robot_state_action_map = dict()        # (state, actionIndex) : {next_state, prob}
        self.state_action_reward_map = dict()       # (state, actionIndex) : (cost)
        self.init_transition()
        self.motion_mdp = Motion_MDP(self.robot_nodes, self.robot_edges, self.actions)
        self.init_observations()
        self.pomcp = None
        self.end_states = end_states
        #state_observation_map
    def set_transition_prob(self, fnode, actionIndex):
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

    def init_transition(self):
        for fnode in self.robot_nodes: 
            for action_index, action in enumerate(self.actions):
                self.set_transition_prob(fnode, action_index)
    
    def display_state_transiton(self):
        print("++++++++++ state transition")
        for state in self.robot_state_action_map:
            for actionIndex in self.robot_state_action_map[state]:
                u = self.actions[actionIndex]
                print(state, u, self.robot_state_action_map[state][actionIndex])
        print("++++++++++")
        for actionIndex, u in enumerate(self.actions):
            for state in self.state_tra[actionIndex]:
                print(state, u, self.state_tra[actionIndex][state])

    def display_state_observation(self):
        for state, obs in self.state_observation_map.items():
            print("state", state, "obs = ", obs)

    def compute_accepting_states(self):
        #compute safe space with respect to static enviornment
        motion_mdp = self.motion_mdp
        self.successor_mdp = dict()
        for node in self.motion_mdp.nodes():
            self.successor_mdp[node]= motion_mdp.successors(node)

        Sf = compute_accept_states(motion_mdp, self.obstacles, self.target)
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
        
        return motion_mdp, AccStates

    def init_observations(self):
        # @pian, why state is mapped to observation set?
        # modified to state to observation, one to one map
        self.obs_nodes = set()
        for i in range(2, 20, 4):
            for j in range(2, 20, 4):
                onode = (i, j)
                self.obs_nodes.add(onode)

        #----
        self.observation_state_map = dict()
        for o_node in self.obs_nodes:
            ox = o_node[0]
            oy = o_node[1]
            support = set()
            for fnode in self.motion_mdp.nodes(): 
                fx = fnode[0]
                fy = fnode[1]    
                if (abs(fx-ox) <= 2) and (abs(fy-oy) <= 2):
                    state = fnode
                    support.add(state)
            self.observation_state_map[o_node] = support

        #----
        self.state_observation_map = dict()
        for fnode in self.motion_mdp.nodes(): 
            fx = fnode[0]
            fy = fnode[1] 
            for o_node in self.obs_nodes:
                ox = o_node[0]
                oy = o_node[1]  
                if (abs(fx-ox) <= 2) and (abs(fy-oy) <= 2):
                    self.state_observation_map[fnode] = o_node

    def compute_H_step_space(self, H):
        #Compute the H-step recahable support belief states, idea: o -> s -> s' -> o'
        #----calculate H-step reachable set------------
        observation_successor_map = dict()
        for o_node in self.obs_nodes:
            init_obs = set()
            init_obs.add(o_node)
            observation_successor_map[o_node, 0] = init_obs

        for o_node in self.obs_nodes:
            succ_obs = set()
            support_set = self.observation_state_map[o_node]
            for fnode in support_set:
                for tnode in self.motion_mdp.successors(fnode):
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

    def online_compute_winning_region(self, obs_initial_node, AccStates, observation_successor_map, H, ACP_step):
        #--------------ONLINE-------------------------
        # Build the N-step reachable support belief MDP, the target set for the support belief MDP is given by AccStates (which is computed offline)
        # ACP_step: computed adaptive conformal prediction constraints
        ACP = dict()
        for i in range(1, H+1): 
            ACP[i] = ()          
            #ACP[i] = ACP_step[i]
        obstacle_static = set(self.obstacles)
        obstacle_new = dict()
        for i in range(H):
            obstacle_new[i+1] = obstacle_static.union(ACP[i+1])

        #----add time counter----
        obs_initial_node_count = (obs_initial_node, 0)
        H_step_obs = observation_successor_map[obs_initial_node, H]
        obs_nodes_reachable = dict()
        obs_nodes_reachable[obs_initial_node_count] = {frozenset(): 1.0}
        for oc in range(1, H+1):
            for o_node in self.obs_nodes:
                #if o_node in observation_successor_map[obs_initial_node, oc]:
                onode_count = (o_node, oc)
                obs_nodes_reachable[onode_count] = {frozenset(): 1.0}

        SS = dict()
        observation_target = set()
        observation_obstacle = set()
        for o_node in obs_nodes_reachable.keys():
            support_set = self.observation_state_map[o_node[0]]
            if support_set.issubset(set(AccStates)):
                obs_nodes_reachable[(o_node)] = {frozenset(['target']): 1.0}
                observation_target.add(o_node)
            for i in range(1, H+1):
                SS[i] = support_set.intersection(obstacle_new[i])
                if oc == i and len(SS[i]) > 0:
                    obs_nodes_reachable[(o_node)] = {frozenset(['obstacle']): 1.0}
                    observation_obstacle.add(o_node)
        print('Number of target observation states: %s' %len(observation_target))
        print('Number of obstacle observation states: %s' %len(observation_obstacle))

        obs_initial_dict = obs_nodes_reachable[obs_initial_node_count]
        obs_initial_label = obs_initial_dict.keys()
        
        obs_edges = dict()
        for o_node in obs_nodes_reachable.keys():
            support_set = self.observation_state_map[o_node[0]]
            for node in support_set:  
                for k, u in enumerate(self.actions):
                    tnode_set = self.robot_state_action_map[node][k]
                    for ttnode in list(tnode_set.keys()):
                        t_obs_set = self.state_observation_map[ttnode]
                        for t_obs in t_obs_set:
                            if oc < H: 
                                tnode = (t_obs, oc+1)
                            else:
                                tnode = (t_obs, oc)
                            if tnode in obs_nodes_reachable:  
                                obs_edges[(o_node, u, tnode)] = (1, C[k])
        U = self.actions
        obs_mdp = Motion_MDP_label(obs_nodes_reachable, obs_edges, U, obs_initial_node_count, obs_initial_label)

        #----
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
        self.dfa = dfa
        print('DFA done.')

        #----
        accs = []
        for obs_state in obs_mdp.nodes():
            for obs_label, obs_prob in obs_mdp.nodes[obs_state]['label'].items():
                for acc in dfa.graph['accept']:
                    I = acc[0]
                    Ip = (obs_state, obs_label, I)
                    accs.append([Ip])
        print('Number of accepting states in observation space: %s' % len(accs))

        prod_dfa_obs = Product_Dfa(obs_mdp, dfa)
        print('Product DFA done')
        # ----

        self.successor_obs_mdp = dict()
        for node in obs_mdp:
            self.successor_obs_mdp[node]= obs_mdp.successors(node)

        #----
        Winning_obs = set()
        Obs_Sf = prod_dfa_obs.graph['accept']
        for S_f in Obs_Sf:
            for MEC in S_f:
                for sf in MEC:
                    if sf[1] not in set(frozenset({'obstacle'})):
                        Winning_obs.add(sf[0]) 
        print('Number of winning states in observation space: %s' % len(Winning_obs))

        f_accept_observation = open('data/accept_observation.dat','w')
        for nd_id, nd in enumerate(Winning_obs):
            # ts_node_id, ts_node_x, ts_node_y, ts_node_d
            f_accept_observation.write('%s,%s\n' %(nd[0], nd[1]))
        f_accept_observation.close()

        return obs_mdp, Winning_obs

    def check_winning(self, support_belief = [], actionIndex = 0, current_state = -1):
        #----Randomly choose the last step belief state-------------
        belief = (1/4, 1/4, 1/4, 1/4)

        #---The support states and the corresponding observation are-----
        if not support_belief:
            support_belief = [((5, 5), 1), 
                            ((5, 7), 1),
                            ((7, 5), 1),
                            ((7, 7), 1),
                            ]
            
        observation = self.get_observation_from_belief(support_belief) # how to get observation of belief support
        print(observation)
        obs_time = observation[1]


        #----Randomly choose an action---------
        action = self.actions[actionIndex]

        #----Make an observation in robot workspace------
        # how to get next obs        
        # next_stateWS = self.step(current_state, actionIndex)
        # observation_WS_next = self.state_observation_map[next_stateWS]
        observation_WS_next = (10, 6)
        oc_next = obs_time+1
        observation_next = (observation_WS_next, oc_next)
        print(observation_next)

        #----Check winning-------
        if observation_next.issubset(self.Winning_obs):
            print('The belief support is a winning state!')
        else:
            print('The belief support is a failure!')


if __name__ == "__main__":
    U = actions = ['N', 'S', 'E', 'W', 'ST']
    C = cost = [9, 3, 3, 3, 1]

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

    obstacles =  [(5, 1), (7, 3), (17, 7)]
    target = [(19, 19)]
    end_states = set([(19,1)])

    robot_nodes = set()
    for i in range(1, 20, 2):
        for j in range(1, 20, 2):
            node = (i, j)
            robot_nodes.add(node) 

    initial_belief_support = [((5, 5), 0), 
                            ((5, 7), 0),
                            ((7, 5), 0),
                            ((7, 7), 0),
                            ]
    initial_belief = {}
    for state in initial_belief_support:
        initial_belief[state] = 1 / len(initial_belief_support)   

    pomdp = Model(robot_nodes, actions, cost, WS_transition, transition_prob,
                    initial_belief, obstacles, target, end_states)

    motion_mdp, AccStates = pomdp.compute_accepting_states()

    H = 5 #Horizon
    observation_successor_map = pomdp.compute_H_step_space(H)

    #---Online planning starts----
    obs_current_node = (6, 6)
    ACP_step = dict() #conformal prediction constraints
    obs_mdp, Winning_observation = pomdp.online_compute_winning_region(obs_current_node, AccStates, observation_successor_map, H, ACP_step)
    #---winning region computation ends---

    # for key, value in pomdp.observation_state_map.items():
    #     print(key, value)