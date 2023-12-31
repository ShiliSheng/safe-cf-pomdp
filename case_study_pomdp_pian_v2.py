from MDP_TG.mdp import Motion_MDP
from MDP_TG.dra import Dra, Dfa, Product_Dra, Product_Dfa
from MDP_TG.vi4wr import syn_plan_prefix, syn_plan_prefix_dfa
from networkx.classes.digraph import DiGraph
import pickle
import time

t0 = time.time()

# -------- real example -------
# Given a POMDP and an LTL formula, it is required that 
# the probability of satisfying the LTL formula is >= gamma.
# In addition, there are conformal prediction constraints.

# This algorithm contains 2 stages:
#  Stage 1: Offline compute 1) a set of states from which the the probability of satisfying the LTL formula is >= gamma;
#                           2) H step reachable support belief states given the inital support belief state
#  Stage 2: Online compute the winning region for satisying the conformal prediction constraints.

#-------------------------
#--State of the POMDP (x, y)---
#--State of the product POMDP-LTL ((x, y),label, a), where a is the DRA state---
#--State of the support belief MDP ((ox, oy, oc), a), where (ox, oy) is the observation state, oc is a time counter, a is the DRA state
#--State in the final winning region (((ox, oy, oc), a), dl, f), where dl is the DFA (translated from !obstacle U target) label, f is the DFA state

#--------------OFFLINE 1)-------------------------
#--Compte a set of states from which the the probability of satisfying the LTL formula is >= gamma, this set is the target set for online planning 
WS_d = 1
WS_node_dict = dict()
for i in range(1, 20, 2):
    for j in range(1, 20, 2):
        WS_node_dict[(i, j)] = {frozenset(): 1.0}

obstacle = [(5, 1), (7, 3), (17, 7)]
base1 = [(19, 19)]
base2 = [(19, 1)]

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

#----
with open('pickle_data/robot_nodes.pkl', 'wb') as pickle_file:
    pickle.dump(robot_nodes, pickle_file)
#----

initial_node = (1, 1)
initial_label = frozenset()

U = [tuple('N'), tuple('S'), tuple('E'), tuple('W'), tuple('ST')]
C = [3, 3, 3, 3, 1]
P_FR = [0.1, 0.8, 0.1]
P_BK = [0.1, 0.8, 0.1]
P_TR = [0.1, 0.8, 0.1]
P_TL = [0.1, 0.8, 0.1]
P_ST = [1]
# ----
robot_edges = dict()
robot_state_action_map = dict()
for fnode in robot_nodes.keys(): 
    fx = fnode[0]
    fy = fnode[1]
    # action N
    u = U[0]
    c = C[0]
    t_nodes = [(fx-2, fy+2), (fx, fy+2), (fx+2, fy+2)]
    for k, tnode in enumerate(t_nodes):
        succ_set = dict()
        if tnode in list(robot_nodes.keys()):
            robot_edges[(fnode, u, tnode)] = (P_FR[k], c)
            succ_prop = {tnode: P_FR[k]}
            succ_set.update(succ_prop)
    robot_state_action_map[(fnode, u)] = succ_set
    
    # action S
    u = U[1]
    c = C[1]
    t_nodes = [(fx-2, fy-2), (fx, fy-2), (fx+2, fy-2)]
    for k, tnode in enumerate(t_nodes):
        succ_set = dict()
        if tnode in list(robot_nodes.keys()):
            robot_edges[(fnode, u, tnode)] = (P_BK[k], c)
            succ_prop = {tnode: P_FR[k]}
            succ_set.update(succ_prop)
    robot_state_action_map[(fnode, u)] = succ_set

    # action E
    u = U[2]
    c = C[2]
    t_nodes = [(fx+2, fy-2), (fx+2, fy), (fx+2, fy+2)]
    for k, tnode in enumerate(t_nodes):
        succ_set = dict()
        if tnode in list(robot_nodes.keys()):
            robot_edges[(fnode, u, tnode)] = (P_TR[k], c)
            succ_prop = {tnode: P_FR[k]}
            succ_set.update(succ_prop)
    robot_state_action_map[(fnode, u)] = succ_set

    # action W
    u = U[3]
    c = C[3]
    t_nodes = [(fx-2, fy-2), (fx-2, fy), (fx-2, fy+2)]
    for k, tnode in enumerate(t_nodes):
        succ_set = dict()
        if tnode in list(robot_nodes.keys()):
            robot_edges[(fnode, u, tnode)] = (P_TL[k], c)
            succ_prop = {tnode: P_FR[k]}
            succ_set.update(succ_prop)
    robot_state_action_map[(fnode, u)] = succ_set

    # action ST
    u = U[4]
    c = C[4]
    t_nodes = [(fx, fy)]
    for k, tnode in enumerate(t_nodes):
        succ_set = dict()
        if tnode in list(robot_nodes.keys()):
            robot_edges[(fnode, u, tnode)] = (P_ST[k], c)
            succ_prop = {tnode: P_FR[k]}
            succ_set.update(succ_prop)
    robot_state_action_map[(fnode, u)] = succ_set
# ----
motion_mdp = Motion_MDP(robot_nodes, robot_edges, U,
                        initial_node, initial_label)
motion_mdp.dotify()
t2 = time.time()
print('MDP done, time: %s' % str(t2-t0))

# ----
mdp_nodes_list = motion_mdp.nodes()
successor_mdp = dict()
for node in mdp_nodes_list:
    successor_mdp[node]= motion_mdp.successors(node)
#----
with open('pickle_data/robot_nodes_successor.pkl', 'wb') as pickle_file:
    pickle.dump(successor_mdp, pickle_file)

#----
with open('pickle_data/robot_state_action_map.pkl', 'wb') as pickle_file:
    pickle.dump(robot_state_action_map, pickle_file)
#----
#----

#all_base = '& G F base1 & G F base2 & G F base3 G ! obstacle'
#all_base = '& F target G ! obstacle'
all_base = '& F base1 & F base2 G ! obstacle'
# all_base= '(F G base1) & (G ! obstacle)'
dra = Dra(all_base)
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
    for act in U:
        succ_set = dict()
        for tnode in prod_dra.successors(fnode): 
            tnode_set_dict = robot_state_action_map[(fnode[0], act)]
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

f_accept_observation = open('data/accept_observation.dat','w')
for nd_id, nd in enumerate(AccStates_obs):
    # ts_node_id, ts_node_x, ts_node_y, ts_node_d
    f_accept_observation.write('%s,%s,%s\n' %(nd[0], nd[1], nd[2]))
f_accept_observation.close()

   