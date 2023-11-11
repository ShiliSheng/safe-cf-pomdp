from MDP_TG.dra import Dra, Dfa
import random
import pickle

#--Input: i) the last step belief state 'b' and its corresponding observation '(ox, oy, oc)', the DRA state 'a', the label of the observation 'l', and the DFA state 'f'
#         ii) the current step observation in robot workspace '(ox_next, oy_next)' 
#--Ouput: The next step support belief state and its corresponding DRA and DFA states

#-----------------------------
# Load the dictionary from the pickle file
with open('pickle_data/robot_nodes.pkl', 'rb') as pickle_file1:
    robot_nodes = pickle.load(pickle_file1)

with open('pickle_data/observation_nodes_reachable.pkl', 'rb') as pickle_file2:
    observation_nodes_reachable = pickle.load(pickle_file2)

with open('pickle_data/winning_observation.pkl', 'rb') as pickle_file3:
    winning_observation = pickle.load(pickle_file3)

with open('pickle_data/robot_state_action_map.pkl', 'rb') as pickle_file4:
    robot_state_action_map = pickle.load(pickle_file4)
#----
all_base = '& F base1 & F base2 G ! obstacle'
dra = Dra(all_base)

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


#----Randomly choose the last step belief state-------------
belief = (1/4, 1/4, 1/4, 1/4)

#---The support states and the corresponding observation are-----
support_belief = [(((5, 5, 1), 1), frozenset(), 2), 
                (((5, 7, 1), 1), frozenset(), 2),
                (((7, 5, 1), 1), frozenset(), 2),
                (((7, 7, 1), 1), frozenset(), 2),
                ]
observation = (((6, 6, 1), 1), frozenset(), 2)
print(observation)

obs_time = observation[0][0][2]
obs_dra = observation[0][1]
obs_label = observation[1]
obs_dfa = observation[2]

#----Randomly choose an action---------
action = tuple('E')

#----Make an observation in robot workspace------
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

label_next_dict = observation_nodes_reachable[((ox_next, oy_next, oc_next), 1)]
for l, p in label_next_dict.items():
    label_next = l

for dfa_node in dfa.successors(obs_dfa):
    truth = dfa.check_label_for_dra_edge(
            obs_label, obs_dfa, dfa_node)
    if truth:
       dfa_next = dfa_node

observation_next = set()
for dra in dra_next:
    obs_next = (((ox_next, oy_next, oc_next), dra), label_next, dfa_next)
    observation_next.add(obs_next)

print(observation_next)

#----Check winning-------
if observation_next.issubset(winning_observation):
    print('The belief support is a winning state!')
else:
    print('The belief support is a failure!')
