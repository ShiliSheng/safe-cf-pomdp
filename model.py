from MDP_TG.mdp import Motion_MDP
from MDP_TG.dra import Dra, Dfa, Product_Dra, Product_Dfa
from MDP_TG.vi4wr import syn_plan_prefix, syn_plan_prefix_dfa
from networkx.classes.digraph import DiGraph
from pomcp import POMCP
from pomcp import POMCPNode
import pickle
import time
import random 

class Model:
    def __init__(self, robot_nodes, actions, cost, transition, transiton_prob, obstacles = [], base1 = [], base2 = [], end_states = set()):
        self.t0 = time.time()
        self.robot_nodes = robot_nodes # {state: (label, prob)}
        self.actions = actions
        self.cost = cost
        self.transiton = transition
        self.transition_prob = transiton_prob
        self.obstacles = obstacles
        self.base1 = base1
        self.base2 = base2
        
        self.robot_edges = dict()
        self.state_tra = [{} for _ in range(len(self.actions))]
        self.robot_state_action_map = {}        # (state, actionIndex) : {next_state, prob}
        self.state_action_reward_map = {}       # (state, actionIndex) : (cost)
        self.init_transition()
        self.dra = None
        self.pomcp = None
        self.end_states = end_states

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
        for fnode in self.robot_nodes.keys(): 
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

    def compute_accepting_states(self, all_base):
        U = self.actions
        c = self.cost

        initial_node = (1, 1) # ?
        initial_label = frozenset()
        motion_mdp = Motion_MDP(self.robot_nodes, self.robot_edges, self.actions,
                                initial_node, initial_label)
        motion_mdp.dotify()

        t2 = time.time()
        print('MDP done, time: %s' % str(t2-self.t0))

        mdp_nodes_list = motion_mdp.nodes()
        successor_mdp = dict()
        for node in mdp_nodes_list:
            successor_mdp[node]= motion_mdp.successors(node)

        with open('pickle_data/robot_nodes_successor.pkl', 'wb') as pickle_file:
            pickle.dump(successor_mdp, pickle_file)
        with open('pickle_data/robot_state_action_map.pkl', 'wb') as pickle_file:
            pickle.dump(self.robot_state_action_map, pickle_file)

        all_base = '& F base1 & F base2 G ! obstacle'
        # all_base= '(F G base1) & (G ! obstacle)'
        dra = Dra(all_base)
        self.dra = dra

        t3 = time.time()
        print('DRA done, time: %s' % str(t3-t2))

        #----
        with open('pickle_data/dra_nodes.pkl', 'wb') as pickle_file:
            pickle.dump(dra.nodes(), pickle_file)
        #----

        # ----
        prod_dra = Product_Dra(motion_mdp, dra)
        prod_dra.dotify()
        t41 = time.time()
        print('Product DRA done, time: %s' % str(t41-t3))
        # ----
        prod_dra.compute_S_f()
        t42 = time.time()
        print('Compute accepting MEC done, time: %s' % str(t42-t41))

        AccStates = []
        for S_fi in prod_dra.Sf:
            for MEC in S_fi:
                for sf in MEC[0]:
                    AccStates.append(sf)

                index, v_new = syn_plan_prefix(prod_dra, MEC)
                for vf in v_new.keys():
                    if v_new[vf] >= 0.9 and vf not in AccStates: #can change the threshold               
                        AccStates.append(vf)
        print('Number of satisfying states for LTL: %s' % len(AccStates))
        t43 = time.time()

        f_accept_node = open('data/accept_node.dat','w')
        for nd_id, nd in enumerate(AccStates):
            # ts_node_id, ts_node_x, ts_node_y, ts_node_d
            f_accept_node.write('%s,%s,%d\n' %(nd[0], nd[1], nd[2]))
        f_accept_node.close()

        #--------------OFFLINE 2)------------------------
        #--Compute the H-step recahable support belief states, idea: o -> s -> s' -> o'
        prod_nodes_list = prod_dra.nodes()
        #----
        with open('pickle_data/prod_dra_nodes.pkl', 'wb') as pickle_file:
            pickle.dump(prod_nodes_list, pickle_file)
        #----

        successor_map = dict()
        for node in prod_nodes_list:
            successor_map[node]= prod_dra.successors(node)

        #----
        with open('pickle_data/product_dra_nodes_successor.pkl', 'wb') as pickle_file:
            pickle.dump(successor_map, pickle_file)
        #----

        #----
        prod_state_action_successor_map = dict()
        for fnode in prod_nodes_list:
            for actionIndex, act in enumerate(self.actions):
                succ_set = dict()
                for tnode in prod_dra.successors(fnode): 
                    # tnode_set_dict = self.robot_state_action_map[(fnode[0], act)]
                    tnode_set_dict = self.robot_state_action_map[fnode[0]][actionIndex]
                    if tnode[0] in tnode_set_dict.keys():
                        prob = tnode_set_dict[tnode[0]]
                        tnode_dict = {tnode: prob}
                        succ_set.update(tnode_dict)
                prod_state_action_successor_map[(fnode, act)] = succ_set

        #----
        with open('pickle_data/product_dra_state_action_successor.pkl', 'wb') as pickle_file:
            pickle.dump(prod_state_action_successor_map, pickle_file)
        #----

        obs_nodes = dict()
        for i in range(2, 20, 4):
            for j in range(2, 20, 4):
                for q in dra.nodes():
                    obs_nodes[((i, j), q)] = {frozenset(): 1.0}

        #----
        observation_map = dict()
        observation_map_WS = dict()
        for o_node in obs_nodes.keys():
            ox = o_node[0][0]
            oy = o_node[0][1]
            oz = o_node[1]
            support = set()
            support_WS = set()
            for fnode in prod_nodes_list: 
                fx = fnode[0][0]
                fy = fnode[0][1]
                fz = fnode[2]    
                if (abs(fx-ox) <= 2) and (abs(fy-oy) <= 2) and (fz == oz):
                    state = fnode
                    state_WS = fnode[0]
                    support.add(state)
                    support_WS.add(state_WS)
            observation_map[o_node] = support
            observation_map_WS[o_node] = support_WS

        file_dot = open('observation_state.dot', 'w')
        file_dot.write('observation -> support states { \n')
        obs_visited = set()
        for o_node in obs_nodes.keys():
            obs = (o_node[0], o_node[1])
            if obs not in obs_visited:
                support_WS = observation_map_WS[o_node]
                file_dot.write('"'+str(obs)+'"' +
                                '->' + '"' + str(support_WS) + '"' + ';\n')
            obs_visited.add(obs)
        file_dot.write('}\n')
        file_dot.close()
        print("-------observation_state.dot generated-------")

        #----
        state_observation_map = dict()
        state_observation_map_WS = dict()
        for fnode in prod_nodes_list: 
            fx = fnode[0][0]
            fy = fnode[0][1]
            fz = fnode[2] 
            support_obs = set()
            support_obs_WS = set()   
            for o_node in obs_nodes.keys():
                ox = o_node[0][0]
                oy = o_node[0][1]
                oz = o_node[1]  
                obs_WS = (ox, oy)  
                if (abs(fx-ox) <= 2) and (abs(fy-oy) <= 2) and (fz == oz):
                    support_obs.add(o_node)
                    support_obs_WS.add(obs_WS)
            state_observation_map[fnode] = support_obs
            state_observation_map_WS[fnode] = support_obs_WS

        self.state_observation_map = state_observation_map
        #----

        file_dot = open('state_observation.dot', 'w')
        file_dot.write('state -> observation { \n')
        node_visited = set()
        for fnode in prod_nodes_list:
            node = fnode[0]
            if node not in node_visited:
                support_obs_WS = state_observation_map_WS[fnode]
                file_dot.write('"'+str(node)+'"' +
                                '->' + '"' + str(support_obs_WS) + '"' + ';\n')
            node_visited.add(node)
        self.state_observation_map_WS = state_observation_map_WS

        file_dot.write('}\n')
        file_dot.close()
        print("-------observation_state.dot generated-------")

        #----calculate H-step reachable set------------
        H = 5
        observation_successor_map = dict()
        for o_node in obs_nodes.keys():
            init_obs = set()
            init_obs.add(o_node)
            observation_successor_map[o_node, 0] = init_obs

        for o_node in obs_nodes.keys():
            succ_obs = set()
            support_set = observation_map[o_node]
            for fnode in support_set:
                for tnode in successor_map[fnode]:
                    support_obs = state_observation_map[tnode]
                    for obs in support_obs:
                        if obs not in succ_obs:
                            succ_obs.add(obs)
            observation_successor_map[o_node, 1] = succ_obs

        for o_node in obs_nodes.keys():
            for i in range(2, H+1):
                succ_obs = observation_successor_map[o_node, i-1]
                succ_step = set()
                for oo_node in succ_obs:
                    for ooo_node in observation_successor_map[oo_node, 1]:
                        if ooo_node not in succ_step:
                            succ_step.add(ooo_node)
                observation_successor_map[o_node, i] = succ_step

        t5 = time.time()

        #--------------ONLINE-------------------------
        # Build the N-step reachable support belief MDP, the target set for the support belief MDP is given by AccStates (which is computed offline)
        ACP = dict()
        ACP_step = dict()
        for i in range(1, H+1):
            ACP_step[i] = []  #adaptive conformal prediction constraints
            ACP[i] = ACP_step[i]
        obstacle_static = set(obstacle)
        obstacle_new = dict()
        for i in range(H):
            obstacle_new[i+1] = obstacle_static.union(ACP[i+1])

        #----
        obs_initial_node = ((2, 2), 1)
        obs_initial_node_count = ((2, 2, 0), 1)
        obs_initial_label = frozenset()

        H_step_obs = observation_successor_map[obs_initial_node, H]
        obs_nodes_reachable = dict()
        obs_nodes_reachable[obs_initial_node_count] = {frozenset(): 1.0}
        for o_node, prop in obs_nodes.items():
            ox = o_node[0][0]
            oy = o_node[0][1]
            oz = o_node[1] 
            for oc in range(1, H+1):
                if o_node in H_step_obs:
                    obs_nodes_reachable[((ox, oy, oc), oz)] = prop

        SS = dict()
        for o_node in obs_nodes_reachable.keys():
            ox = o_node[0][0]
            oy = o_node[0][1]
            oc = o_node[0][2]
            oz = o_node[1]
            support_set = observation_map[((ox, oy), oz)]
            support_set_WS = observation_map_WS[((ox, oy), oz)]
            if support_set.issubset(AccStates):
                obs_nodes_reachable[(o_node)] = {frozenset(['target']): 1.0}
            for i in range(1, H+1):
                SS[i] = support_set_WS.intersection(obstacle_new[i])
                if oc == i and len(SS[i]) > 0:
                    obs_nodes_reachable[(o_node)] = {frozenset(['obstacle']): 1.0}
            
        #----
        with open('pickle_data/observation_nodes_reachable.pkl', 'wb') as pickle_file:
            pickle.dump(obs_nodes_reachable, pickle_file)
        #----
        self.observation_nodes_reachable = obs_nodes_reachable

        obs_edges = dict()
        for fnode, prop in obs_nodes_reachable.items():
            ox = fnode[0][0]
            oy = fnode[0][1]
            oc = fnode[0][2]
            ok = fnode[1]
            support_set = observation_map[((ox, oy), ok)]
            for node in support_set:  
                for k, u in enumerate(U):
                    tnode_set = prod_state_action_successor_map[(node, u)]
                    for ttnode in tnode_set.keys():
                        t_obs_set = state_observation_map[ttnode]
                        for t_obs in t_obs_set:
                            if oc < H: 
                                tnode = ((t_obs[0][0], t_obs[0][1], oc+1), t_obs[1])
                            else:
                                tnode = ((t_obs[0][0], t_obs[0][1], oc), t_obs[1]) 
                            if tnode in list(obs_nodes_reachable.keys()):  
                                obs_edges[(fnode, u, tnode)] = (1, C[k])

        obs_mdp = Motion_MDP(obs_nodes_reachable, obs_edges, U,
                                obs_initial_node_count, obs_initial_label)

        #----
        successor_obs_mdp = dict()
        for node in obs_mdp:
            successor_obs_mdp[node]= obs_mdp.successors(node)
        #----

        with open('pickle_data/observation_nodes_reachable_successor.pkl', 'wb') as pickle_file:
            pickle.dump(successor_obs_mdp, pickle_file)
        #----

        # important to transform string names to indexs
        ts = obs_mdp
        ts_nodes_list = ts.nodes()
        f_ts_node = open('data/ts_node.dat','w')
        for nd_id, nd in enumerate(ts_nodes_list):
            # ts_node_id, ts_node_x, ts_node_y, ts_node_d
            f_ts_node.write('%d,%s,%d\n' %(nd_id, nd[0], nd[1]))
            ts.nodes[nd]['index'] = nd_id
        f_ts_node.close()
        # save edges, node name swapped by index
        f_ts_edge = open('data/ts_edge.dat','w')
        for e in ts.edges():
            id_ef = ts.nodes[e[0]]['index']
            id_et = ts.nodes[e[1]]['index']
            f_ts_edge.write('%d,%d\n' %(id_ef, id_et)) 
        f_ts_edge.close()

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

        #----
        with open('pickle_data/dfa_nodes.pkl', 'wb') as pickle_file:
            pickle.dump(dfa.nodes(), pickle_file)
        #----

        prod_dfa_obs = Product_Dfa(obs_mdp, dfa)
        t6 = time.time()
        print('Product DFA done, time: %s' % str(t6-t5))
        # ----
        prod_dfa_list = prod_dfa_obs.nodes()
        #----
        with open('pickle_data/prod_dfa_nodes.pkl', 'wb') as pickle_file:
            pickle.dump(prod_dfa_list, pickle_file)
        #----

        successor_map_obs = dict()
        for node in prod_dfa_list:
            successor_map_obs[node]= prod_dfa_obs.successors(node)
        #----
        with open('pickle_data/product_dfa_nodes_successor.pkl', 'wb') as pickle_file:
            pickle.dump(successor_map_obs, pickle_file)
        #----

        AccStates_obs = set()
        Obs_Sf = prod_dfa_obs.graph['accept']
        for S_f in Obs_Sf:
            for MEC in S_f:
                for sf in MEC:
                    AccStates_obs.add(sf) # sf = (((x_axis, y_axis, time_step), DRA state), observation_label, DFA state)

                index, v_new = syn_plan_prefix_dfa(prod_dfa_obs, MEC)
                for vf in v_new.keys():
                    if v_new[vf] >= 0.999 and vf not in AccStates_obs: #can change the threshold               
                        AccStates_obs.add(vf)
        print('Number of winning states in observation space: %s' % len(AccStates_obs))

        with open('pickle_data/winning_observation.pkl', 'wb') as file:
            pickle.dump(AccStates_obs, file)

        f_accept_observation = open('data/accept_observation.dat','w')
        for nd_id, nd in enumerate(AccStates_obs):
            # ts_node_id, ts_node_x, ts_node_y, ts_node_d
            f_accept_observation.write('%s,%s,%s\n' %(nd[0], nd[1], nd[2]))
        f_accept_observation.close()

    def compute_winning_region(self, ACP = {}):
        t5 = time.time()

        """
        this can be improved
        """
        obstacle = [(5, 1), (7, 3), (17, 7)]
        H = 5
        U = actions = ['N', 'S', 'E', 'W', 'ST']
        C = cost = [3, 3, 3, 3, 1]
        with open('pickle_data/data.pkl', 'rb') as pickle_file1:
            observation_successor_map, obs_nodes, observation_map, \
            observation_map_WS, AccStates, prod_state_action_successor_map, \
            state_observation_map, obstacle, H, U  = pickle.load(pickle_file1)


        ACP = dict()
        ACP_step = dict()
        for i in range(1, H+1):
            ACP_step[i] = []  #adaptive conformal prediction constraints
            ACP[i] = ACP_step[i]
        obstacle_static = set(obstacle)
        obstacle_new = dict()
        for i in range(H):
            obstacle_new[i+1] = obstacle_static.union(ACP[i+1])

        #----
        obs_initial_node = ((2, 2), 1)
        obs_initial_node_count = ((2, 2, 0), 1)
        obs_initial_label = frozenset()

        H_step_obs = observation_successor_map[obs_initial_node, H]
        obs_nodes_reachable = dict()
        obs_nodes_reachable[obs_initial_node_count] = {frozenset(): 1.0}
        for o_node, prop in obs_nodes.items():
            ox = o_node[0][0]
            oy = o_node[0][1]
            oz = o_node[1] 
            for oc in range(1, H+1):
                if o_node in H_step_obs:
                    obs_nodes_reachable[((ox, oy, oc), oz)] = prop

        SS = dict()
        for o_node in obs_nodes_reachable.keys():
            ox = o_node[0][0]
            oy = o_node[0][1]
            oc = o_node[0][2]
            oz = o_node[1]
            support_set = observation_map[((ox, oy), oz)]
            support_set_WS = observation_map_WS[((ox, oy), oz)]
            if support_set.issubset(AccStates):
                obs_nodes_reachable[(o_node)] = {frozenset(['target']): 1.0}
            for i in range(1, H+1):
                SS[i] = support_set_WS.intersection(obstacle_new[i])
                if oc == i and len(SS[i]) > 0:
                    obs_nodes_reachable[(o_node)] = {frozenset(['obstacle']): 1.0}
            
        #----
        with open('pickle_data/observation_nodes_reachable.pkl', 'wb') as pickle_file:
            pickle.dump(obs_nodes_reachable, pickle_file)
        #----

        obs_edges = dict()
        for fnode, prop in obs_nodes_reachable.items():
            ox = fnode[0][0]
            oy = fnode[0][1]
            oc = fnode[0][2]
            ok = fnode[1]
            support_set = observation_map[((ox, oy), ok)]
            for node in support_set:  
                for k, u in enumerate(U):
                    tnode_set = prod_state_action_successor_map[(node, u)]
                    for ttnode in tnode_set.keys():
                        t_obs_set = state_observation_map[ttnode]
                        for t_obs in t_obs_set:
                            if oc < H: 
                                tnode = ((t_obs[0][0], t_obs[0][1], oc+1), t_obs[1])
                            else:
                                tnode = ((t_obs[0][0], t_obs[0][1], oc), t_obs[1]) 
                            if tnode in list(obs_nodes_reachable.keys()):  
                                obs_edges[(fnode, u, tnode)] = (1, C[k])

        obs_mdp = Motion_MDP(obs_nodes_reachable, obs_edges, U,
                                obs_initial_node_count, obs_initial_label)

        #----
        successor_obs_mdp = dict()
        for node in obs_mdp:
            successor_obs_mdp[node]= obs_mdp.successors(node)
        #----

        with open('pickle_data/observation_nodes_reachable_successor.pkl', 'wb') as pickle_file:
            pickle.dump(successor_obs_mdp, pickle_file)
        #----

        # important to transform string names to indexs
        ts = obs_mdp
        ts_nodes_list = ts.nodes()
        f_ts_node = open('data/ts_node.dat','w')
        for nd_id, nd in enumerate(ts_nodes_list):
            # ts_node_id, ts_node_x, ts_node_y, ts_node_d
            f_ts_node.write('%d,%s,%d\n' %(nd_id, nd[0], nd[1]))
            ts.nodes[nd]['index'] = nd_id
        f_ts_node.close()
        # save edges, node name swapped by index
        f_ts_edge = open('data/ts_edge.dat','w')
        for e in ts.edges():
            id_ef = ts.nodes[e[0]]['index']
            id_et = ts.nodes[e[1]]['index']
            f_ts_edge.write('%d,%d\n' %(id_ef, id_et)) 
        f_ts_edge.close()

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

        #----
        with open('pickle_data/dfa_nodes.pkl', 'wb') as pickle_file:
            pickle.dump(dfa.nodes(), pickle_file)
        #----

        prod_dfa_obs = Product_Dfa(obs_mdp, dfa)
        t6 = time.time()
        print('Product DFA done, time: %s' % str(t6-t5))
        # ----
        prod_dfa_list = prod_dfa_obs.nodes()
        #----
        with open('pickle_data/prod_dfa_nodes.pkl', 'wb') as pickle_file:
            pickle.dump(prod_dfa_list, pickle_file)
        #----

        successor_map_obs = dict()
        for node in prod_dfa_list:
            successor_map_obs[node]= prod_dfa_obs.successors(node)
        #----
        with open('pickle_data/product_dfa_nodes_successor.pkl', 'wb') as pickle_file:
            pickle.dump(successor_map_obs, pickle_file)
        #----

        AccStates_obs = set()
        Obs_Sf = prod_dfa_obs.graph['accept']
        for S_f in Obs_Sf:
            for MEC in S_f:
                for sf in MEC:
                    AccStates_obs.add(sf) # sf = (((x_axis, y_axis, time_step), DRA state), observation_label, DFA state)

                index, v_new = syn_plan_prefix_dfa(prod_dfa_obs, MEC)
                for vf in v_new.keys():
                    if v_new[vf] >= 0.999 and vf not in AccStates_obs: #can change the threshold               
                        AccStates_obs.add(vf)
        print('Number of winning states in observation space: %s' % len(AccStates_obs))

        with open('pickle_data/winning_observation.pkl', 'wb') as file:
            pickle.dump(AccStates_obs, file)
        self.winning_observation = AccStates_obs

        f_accept_observation = open('data/accept_observation.dat','w')
        for nd_id, nd in enumerate(AccStates_obs):
            # ts_node_id, ts_node_x, ts_node_y, ts_node_d
            f_accept_observation.write('%s,%s,%s\n' %(nd[0], nd[1], nd[2]))
        f_accept_observation.close()

##########
    def get_observation_from_belief(self, support_belief = []): # TODO @piany to review
        # ret observation = (((ox, oy, oc), dra_state), label, dfa_state)
        # how to get observation from belief support if dra is not always the same
        # return (((6, 6, 1), 1), frozenset(), 2)
        new_obs = set()
        for support in support_belief:
            # ((x, y, oc), dra_state), label, dfa_state = support
            (stateWS_time, dra_state), label, dfa_state = support
            state_WS, oc = stateWS_time[:-1], stateWS_time[-1]
            state = state_WS, label, dra_state
            for obsWS in self.state_observation_map_WS[state]:
                obsWS_time = (*obsWS, oc)
                new_obs.add(((obsWS_time, dra_state), label, dfa_state))
                # new_obs.add((((ox, oy, oc), odra), label, dfa_state)) is it the same ?
        observation = list(new_obs)[0]
        print("observation", observation)
        return observation

    def get_next_possible_states(self, stateWS, actionIndex): #TODO
        if stateWS not in self.robot_state_action_map or actionIndex not in self.robot_state_action_map[stateWS]:
            print("erorr")
            return set()
        return self.robot_state_action_map[stateWS][actionIndex]

    def get_next_belief_support(self, support_belief = [], actionIndex = -1, observation = -1): # TODO
        # return
        # b, a, o => b'
        next_support_beleif = set()
        for key in support_belief:
            # ((x, y, oc), dra_state), label, dfa_state = key
            (stateWS_time, dra_state), label, dfa_state = key
            stateWS, oc = stateWS_time[:-1], stateWS_time[-1]
            next_possible_states = self.get_next_possible_states(stateWS, actionIndex, observation)

            # get next dra

            oc += 1

            # get next dfa
            next_state = ((nx, ny, oc), nxt_dra, nxt_dfa)
            next_support_beleif.add(next_state)
        return list(next_support_beleif)

    def step(self, state, actionIndex):
            probabilities = self.robot_state_action_map[state][actionIndex]
            states, probs = zip(*probabilities.items())
            next_state = random.choices(states, weights=probs, k=1)[0]
            return next_state

    def check_winning(self, support_belief = [], actionIndex = 0, current_state = -1):
        dra = self.dra
        dfa = self.dfa

        #----Randomly choose the last step belief state-------------
        belief = (1/4, 1/4, 1/4, 1/4)

        #---The support states and the corresponding observation are-----
        if not support_belief:
            # why is support belief defined as [((x, y, oc), dra_state), label, dfa_state]
            support_belief = [(((5, 5, 1), 1), frozenset(), 2), 
                            (((5, 7, 1), 1), frozenset(), 2),
                            (((7, 5, 1), 1), frozenset(), 2),
                            (((7, 7, 1), 1), frozenset(), 2),
                            ]
        observation = self.get_observation_from_belief(support_belief) # how to get observation of belief support
        print(observation)
        obs_time = observation[0][0][2]
        obs_dra = observation[0][1]
        obs_label = observation[1]
        obs_dfa = observation[2]

        #----Randomly choose an action---------
        action = self.actions[actionIndex]

        #----Make an observation in robot workspace------
        # how to get next obs        
        # next_stateWS = self.step(current_state, actionIndex)
        # observation_WS_next = self.state_observation_map[next_stateWS]
        observation_WS_next = (10, 6)
        
        ox_next = observation_WS_next[0]
        oy_next = observation_WS_next[1]
        oc_next = obs_time+1

        #----Compute corresponding DRA and DFA states-------
        label_dict = dict()
        for state in support_belief:
            fnode_WS = (state[0][0][0], state[0][0][1])
            label_dict.update(robot_nodes[fnode_WS])

        f_dra_label = set()
        for label, prob in label_dict.items(): 
            if label not in f_dra_label:
                f_dra_label.add(label)

        dra_next = set()
        for label in f_dra_label:
            for dra_node in dra.successors(obs_dra):
                truth = dra.check_label_for_dra_edge(
                    label, obs_dra, dra_node)
                if truth:
                    dra_next.add(dra_node)

        label_next_dict = self.observation_nodes_reachable[((ox_next, oy_next, oc_next), 1)]
        for l, p in label_next_dict.items():
            label_next = l

        dfa_next = None
        for dfa_node in dfa.successors(obs_dfa):
            truth = dfa.check_label_for_dra_edge(
                    obs_label, obs_dfa, dfa_node)
            if truth:
                dfa_next = dfa_node
                break

        observation_next = set()
        for dra in dra_next:
            obs_next = (((ox_next, oy_next, oc_next), dra), label_next, dfa_next)
            observation_next.add(obs_next)

        print(observation_next)

        #----Check winning-------
        if observation_next.issubset(self.winning_observation):
            print('The belief support is a winning state!')
        else:
            print('The belief support is a failure!')


    def init_pomcp(self):
        constant = 1000
        max_depth = 200
        initial_belief_support = [(((5, 5, 1), 1), frozenset(), 2), 
                        (((5, 7, 1), 1), frozenset(), 2),
                        (((7, 5, 1), 1), frozenset(), 2),
                        (((7, 7, 1), 1), frozenset(), 2),
                        ]
        initial_belief = {}
        for state in initial_belief_support:
            initial_belief[state] = 1 / len(initial_belief_support)
        self.initial_belief = initial_belief_support
        self.pomcp = POMCP(self.initial_belief, self.actions, self.robot_state_action_map, self.state_observation_map, 
                           self.state_action_reward_map, self.end_states, constant, max_depth)
        # PartiallyObservableMonteCarloPlanning pomcp = new PartiallyObservableMonteCarloPlanning(, , target, minMax, statesOfInterest, endStates, constant, maxDepth);
if __name__ == "__main__":
    U = actions = ['N', 'S', 'E', 'W', 'ST']
    C = cost = [3, 3, 3, 3, 1]

    transition_prob = [[] for _ in range(len(actions))]
    transition_prob[0] = [0.1, 0.8, 0.1] # S
    transition_prob[1] = [0.1, 0.8, 0.1] # N
    transition_prob[2] = [0.1, 0.8, 0.1] # E
    transition_prob[3] = [0.1, 0.8, 0.1] # W
    transition_prob[4] = [1]             # ST

    transition = [[] for _ in range(len(actions))]
    transition[0] = [(-2, 2), (0, 2), (2, 2)]       # S
    transition[1] = [(-2, -2), (0, -2), (2, -2)]    # N
    transition[2] = [(2, -2), (2, 0), (2, 2)]       # E
    transition[3] = [(-2, -2), (-2, 0), (-2, 2)]    # W
    transition[4] = [(0,0)]                         # ST

    obstacle =  [(5, 1), (7, 3), (17, 7)]
    base1 = [(19, 19)]
    base2 = [(19, 1)]
    end_states = set([(19,1)])

    WS_d = 1  # ?
    WS_node_dict = dict()
    for i in range(1, 20, 2):
        for j in range(1, 20, 2):
            WS_node_dict[(i, j)] = {frozenset(): 1.0}
    robot_nodes = dict()
    for loc, prop in WS_node_dict.items():
        if (loc[0], loc[1]) in obstacle:
            robot_nodes[(loc[0], loc[1])] = {frozenset(['obstacle']): 1.0}
        elif (loc[0], loc[1]) in base1:
            robot_nodes[(loc[0], loc[1])] = {frozenset(['base1']): 1.0}
        elif (loc[0], loc[1]) in base2:
            robot_nodes[(loc[0], loc[1])] = {frozenset(['base2']): 1.0}
        else:
            robot_nodes[(loc[0], loc[1])] = prop

    all_base = '& F base1 & F base2 G ! obstacle'
    
    initial_belief_support = [(((5, 5, 1), 1), frozenset(), 2), 
                            (((5, 7, 1), 1), frozenset(), 2),
                            (((7, 5, 1), 1), frozenset(), 2),
                            (((7, 7, 1), 1), frozenset(), 2),
                            ]
    initial_belief = {}
    for state in initial_belief_support:
        initial_belief[state] = 1 / len(initial_belief_support)

    pomdp = Model(robot_nodes, actions, cost, transition, transition_prob, obstacle, base1, base2, end_states)
    # pomdp.display_state_transiton()
    pomdp.compute_accepting_states(all_base)
    pomdp.compute_winning_region()
    # pomdp.check_winning()
    pomdp.init_pomcp()
    """
        # online planning
        
        pomdp = model()
        pomdp.compute_accepting_states(allbase)

        while step < 1000:
            step += 1
            acp  = acp(observation)
            winning_region = compute_wining_region(acp)
            action = pomcp.select_action(winning_region)
                # check_wining(belif_support)
            new_belief, observation <= exectue(action)
    """
