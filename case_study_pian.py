from MDP_TG.mdp import Motion_MDP
from MDP_TG.dra import Dra, Product_Dra
from MDP_TG.lp import syn_full_plan

import time

t0 = time.time()

# -------- real example -------
WS_d = 1

WS_node_dict = {
    # base stations
    (9.0, 1.0): {frozenset(): 1.0, },
    (9.0, 9.0): {frozenset(): 1.0, },
    (1.0, 9.0): {frozenset(): 1.0, },
    (3.0, 3.0): {frozenset(['obstacle']): 1.0, },
    (3.0, 5.0): {frozenset(): 1.0, },
    (7.0, 5.0): {frozenset(): 1.0 },
    (3.0, 9.0): {frozenset(): 1.0, },
    (3.0, 7.0): {frozenset(): 1.0, },
    (1.0, 5.0): {frozenset(): 1.0, },
    (5.0, 3.0): {frozenset(): 1.0, },
    (9.0, 5.0): {frozenset(): 1.0, },
    (5.0, 9.0): {frozenset(): 1.0, },
    (1.0, 3.0): {frozenset(): 1.0, },
    (1.0, 1.0): {frozenset(): 1.0, },
    (1.0, 7.0): {frozenset(): 1.0, },
    (3.0, 1.0): {frozenset(['obstacle']): 1.0, },
    (7.0, 1.0): {frozenset(): 1.0, },
    (5.0, 1.0): {frozenset(): 1.0, },
    (7.0, 3.0): {frozenset(): 1.0, },
    (5.0, 5.0): {frozenset(): 1.0, },
    (5.0, 7.0): {frozenset(): 1.0, },
    (7.0, 7.0): {frozenset(): 1.0, },
    (7.0, 9.0): {frozenset(): 1.0, },
    (9.0, 3.0): {frozenset(): 1.0, },
    (9.0, 7.0): {frozenset(['obstacle']): 1.0, },
    (1.0, 11.0): {frozenset(['base1']): 1.0, },
    (3.0, 11.0): {frozenset(): 1.0, },
    (5.0, 11.0): {frozenset(): 1.0, },
    (7.0, 11.0): {frozenset(): 1.0, },
    (9.0, 11.0): {frozenset(): 1.0, },
    (11.0, 11.0): {frozenset(['base2']): 1.0, },
    (11.0, 9.0): {frozenset(): 1.0, },
    (11.0, 7.0): {frozenset(): 1.0, },
    (11.0, 5.0): {frozenset(): 1.0, },
    (11.0, 3.0): {frozenset(): 1.0, },
    (11.0, 1.0): {frozenset(['base3']): 1.0, },
}

# ------------------------------------
robot_nodes = dict()
for loc, prop in WS_node_dict.items():
    # for d in ['N', 'S', 'E', 'W']:
    for d in [1, 2, 3, 4]:
        robot_nodes[(loc[0], loc[1], d)] = prop

observation = [tuple('O1'), tuple('O2'), tuple('O3'), tuple('O4'), tuple('O5'), tuple('O6'), tuple('O7'), tuple('O8'), tuple('O9')]
observation_state_map = []

# ------------------------------------
initial_node = (1.0, 1.0, 1)
initial_label = frozenset()

U = [tuple('FR'), tuple('BK'), tuple('TR'), tuple('TL'), tuple('ST')]
C = [2, 4, 3, 3, 1]
P_FR = [0.1, 0.8, 0.1]
P_BK = [0.15, 0.7, 0.15]
P_TR = [0.05, 0.9, 0.05]
P_TL = [0.05, 0.9, 0.05]
P_ST = [0.005, 0.99, 0.005]
# -------------
robot_edges = dict()
for fnode in robot_nodes.keys():
    fx = fnode[0]
    fy = fnode[1]
    fd = fnode[2]
    # action FR
    u = U[0]
    c = C[0]
    if fd == 1:
        t_nodes = [(fx-2, fy+2, fd), (fx, fy+2, fd), (fx+2, fy+2, fd)]
    if fd == 2:
        t_nodes = [(fx-2, fy-2, fd), (fx, fy-2, fd), (fx+2, fy-2, fd)]
    if fd == 3:
        t_nodes = [(fx+2, fy-2, fd), (fx+2, fy, fd), (fx+2, fy+2, fd)]
    if fd == 4:
        t_nodes = [(fx-2, fy-2, fd), (fx-2, fy, fd), (fx-2, fy+2, fd)]
    for k, tnode in enumerate(t_nodes):
        if tnode in list(robot_nodes.keys()):
            robot_edges[(fnode, u, tnode)] = (P_FR[k], c)
    # action BK
    u = U[1]
    c = C[1]
    if fd == 1:
        t_nodes = [(fx-2, fy-2, fd), (fx, fy-2, fd), (fx+2, fy-2, fd)]
    if fd == 2:
        t_nodes = [(fx-2, fy+2, fd), (fx, fy+2, fd), (fx+2, fy+2, fd)]
    if fd == 3:
        t_nodes = [(fx-2, fy-2, fd), (fx-2, fy, fd), (fx-2, fy+2, fd)]
    if fd == 4:
        t_nodes = [(fx+2, fy-2, fd), (fx+2, fy, fd), (fx+2, fy+2, fd)]
    for k, tnode in enumerate(t_nodes):
        if tnode in list(robot_nodes.keys()):
            robot_edges[(fnode, u, tnode)] = (P_BK[k], c)
    # action TR
    u = U[2]
    c = C[2]
    if fd == 1:
        t_nodes = [(fx, fy, 1), (fx, fy, 3), (fx, fy, 2)]
    if fd == 2:
        t_nodes = [(fx, fy, 2), (fx, fy, 4), (fx, fy, 1)]
    if fd == 3:
        t_nodes = [(fx, fy, 3), (fx, fy, 2), (fx, fy, 4)]
    if fd == 4:
        t_nodes = [(fx, fy, 4), (fx, fy, 1), (fx, fy, 3)]
    for k, tnode in enumerate(t_nodes):
        if tnode in list(robot_nodes.keys()):
            robot_edges[(fnode, u, tnode)] = (P_TR[k], c)
    # action TL
    u = U[3]
    c = C[3]
    if fd == 2:
        t_nodes = [(fx, fy, 2), (fx, fy, 3), (fx, fy, 1)]
    if fd == 1:
        t_nodes = [(fx, fy, 1), (fx, fy, 4), (fx, fy, 2)]
    if fd == 4:
        t_nodes = [(fx, fy, 4), (fx, fy, 2), (fx, fy, 3)]
    if fd == 3:
        t_nodes = [(fx, fy, 3), (fx, fy, 1), (fx, fy, 4)]
    for k, tnode in enumerate(t_nodes):
        if tnode in list(robot_nodes.keys()):
            robot_edges[(fnode, u, tnode)] = (P_TL[k], c)
    # action ST
    u = U[4]
    c = C[4]
    if fd == 2:
        t_nodes = [(fx, fy, 4), (fx, fy, 2), (fx, fy, 3)]
    if fd == 1:
        t_nodes = [(fx, fy, 4), (fx, fy, 1), (fx, fy, 3)]
    if fd == 4:
        t_nodes = [(fx, fy, 2), (fx, fy, 4), (fx, fy, 1)]
    if fd == 3:
        t_nodes = [(fx, fy, 1), (fx, fy, 3), (fx, fy, 2)]
    for k, tnode in enumerate(t_nodes):
        if tnode in list(robot_nodes.keys()):
            robot_edges[(fnode, u, tnode)] = (P_ST[k], c)
# ----
motion_mdp = Motion_MDP(robot_nodes, robot_edges, U,
                        initial_node, initial_label)
motion_mdp.dotify()
t2 = time.time()
print('MDP done, time: %s' % str(t2-t0))

#all_base = '(G F base1) & (G F base2) & (G F base3)'
#all_base = '(G F base1) & (G F base2) & (G F base3) & (G ! obstacle)'
# all_base= '(F G base1) & (G ! obstacle)'
all_base = '& G F base1 & G F base2 & G F base3 G ! obstacle'
# all_base= '(F G base1) & (G ! obstacle)'
dra = Dra(all_base)
t3 = time.time()
print('DRA done, time: %s' % str(t3-t2))

# ----
prod_dra = Product_Dra(motion_mdp, dra)
prod_dra.dotify()
t41 = time.time()
print('Product DRA done, time: %s' % str(t41-t3))
# ----
prod_dra.compute_S_f()
t42 = time.time()
print('Compute accepting MEC done, time: %s' % str(t42-t41))

# ----
AccStates = []
for S_fi in prod_dra.Sf:
    for MEC in S_fi:
        for sf in MEC[0]:
            AccStates.append(sf)
print('Number of accepting states: %s' % len(AccStates))

# # ------
# gamma = 0.2
# best_all_plan = syn_full_plan(prod_dra, gamma)
# t5 = time.time()
# print('Plan synthesis done, time: %s' % str(t5-t42))


