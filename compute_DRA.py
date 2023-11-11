from MDP_TG.dra import Dra
import random
import pickle

#-----------------------------
# Load the dictionary from the pickle file
with open('pickle_data/robot_nodes.pkl', 'rb') as pickle_file1:
    robot_nodes = pickle.load(pickle_file1)

with open('pickle_data/robot_nodes_successor.pkl', 'rb') as pickle_file2:
    robot_nodes_successor = pickle.load(pickle_file2)

with open('pickle_data/prod_dra_nodes.pkl', 'rb') as pickle_file3:
    product_dra_nodes = pickle.load(pickle_file3)

with open('pickle_data/product_dra_nodes_successor.pkl', 'rb') as pickle_file4:
    product_dra_nodes_successor = pickle.load(pickle_file4)

all_base = '& F base1 & F base2 G ! obstacle'
dra = Dra(all_base)

#------------------------------------------------------
#----------choose an intial state randomly-------------
prod_nodes_list = product_dra_nodes
fnode = random.choice(list(prod_nodes_list))
fnode_WS = (fnode[0][0], fnode[0][1])
fnode_WS_successor = robot_nodes_successor[fnode_WS]
t_node_WS = random.choice(list(fnode_WS_successor))

#---compute the DRA state for node t_node_WS-----------
label_dict = robot_nodes[t_node_WS]
for label, prob in label_dict.items(): 
    t_node_label = label

fk = fnode[2]
f_mdp_label = fnode[1]
for t_dra_node in dra.successors(fk):
    truth = dra.check_label_for_dra_edge(
            f_mdp_label, fk, t_dra_node)
    if truth:
        t_dra = t_dra_node
tnode = (t_node_WS, t_node_label, t_dra)
        
print(fnode)
print(t_node_WS)
print(tnode)

