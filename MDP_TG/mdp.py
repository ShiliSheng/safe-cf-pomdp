# -*- coding: utf-8 -*-

from math import sqrt
from networkx.classes.digraph import DiGraph
from networkx import single_source_shortest_path
from networkx import strongly_connected_components_recursive,strongly_connected_components
from networkx.algorithms import shortest_path
import random

# ----------------------------------
def print(*args, **kwargs):
    return

new_var = DiGraph
class Motion_MDP_label(new_var):
    # ----construct probabilistic-labeled MDP----
    def __init__(self, node_dict, edge_dict, U, initial_node, initial_label):
        DiGraph.__init__(self, name='motion_mdp',
                         init_state=initial_node, init_label=initial_label)
        for (n, prob_label) in node_dict.items():
            self.add_node(n, label=prob_label, act=set())
        print("-------Motion MDP Initialized-------")
        self.add_edges(edge_dict, U)
        print("%s states and %s edges" %
              (str(len(self.nodes())), str(len(self.edges()))))
        self.unify_mdp()

    def add_edges(self, edge_dict, U):
        self.graph['U'] = set()
        for u in U:
            self.graph['U'].add(tuple(u))
        for edge, attri in edge_dict.items():
            f_node = edge[0]
            u = edge[1]
            t_node = edge[2]
            if (f_node, t_node) in self.edges():
                prop = self[f_node][t_node]['prop']
                prop[tuple(u)] = [attri[0], attri[1]]
            else:
                prob_cost = dict()
                #--- prob, cost
                prob_cost[tuple(u)] = [attri[0], attri[1]]
                self.add_edge(f_node, t_node, prop=prob_cost)
        # ----
        for f_node in self.nodes():
            Us = set()
            for t_node in self.successors(f_node):
                prop = self[f_node][t_node]['prop']
                Us.update(set(prop.keys()))
            if Us:
                self.nodes[f_node]['act'] = Us.copy()
            else:
                print('Isolated state')
        print("-------Motion MDP Constructed-------")

    #--------------Pian added
    def dotify(self):
        # ----generate dot diagram for motion_mdp----
        file_dot = open('mdp.dot', 'w')
        file_dot.write('digraph mdp { \n')
        file_dot.write(
            'graph[rankdir=LR, center=true, margin=0.2, nodesep=0.1, ranksep=0.3]\n')
        file_dot.write(
            'node[shape=circle, fontname="Courier-Bold", fontsize=10, width=0.4, height=0.4, fixedsize=false]\n')
        file_dot.write('edge[arrowsize=0.6, arrowhead=vee]\n')
        for edge in self.edges:
            file_dot.write('"'+str(edge[0])+'"' +
                           '->' + '"' + str(edge[1]) + '"' + ';\n')
        file_dot.write('}\n')
        file_dot.close()
        print("-------motion_mdp.dot generated-------")
        print("Run 'dot -Tpdf motion_mdp.dot > mdp.pdf'")

    def unify_mdp(self):
        # ----verify the probability sums up to 1----
        for f_node in self.nodes():
            for u in self.nodes[f_node]['act']:
                sum_prob = 0
                N = 0
                for t_node in self.successors(f_node):
                    prop = self[f_node][t_node]['prop']
                    if u in list(prop.keys()):
                        sum_prob += prop[u][0]
                        N += 1
                if sum_prob < 1.0:
                    to_add = (1.0-sum_prob)/N
                    for t_node in self.successors(f_node):
                        prop = self[f_node][t_node]['prop']
                        if u in list(prop.keys()):
                            prop[u][0] += to_add
                if sum_prob > 1.0:
                    for t_node in self.successors(f_node):
                        prop = self[f_node][t_node]['prop']
                        if u in list(prop.keys()):
                            prop[u][0] = prop[u][0]/sum_prob
        print('Unify MDP Done')

# --------------------------------

class Motion_MDP(new_var):
    # ----construct MDP----
    def __init__(self, node_set, edge_dict, U):
        DiGraph.__init__(self, name='motion_mdp')
        for node in node_set:
            self.add_node(node, act=set())
        print("-------Motion MDP Initialized-------")
        self.add_edges(edge_dict, U)
        print("%s states and %s edges" %
              (str(len(self.nodes())), str(len(self.edges()))))
        self.unify_mdp()

    def add_edges(self, edge_dict, U):
        self.graph['U'] = set()
        for u in U:
            self.graph['U'].add(tuple(u))
        for edge, attri in edge_dict.items():
            f_node = edge[0]
            u = edge[1]
            t_node = edge[2]
            if (f_node, t_node) in self.edges():
                prop = self[f_node][t_node]['prop']
                prop[tuple(u)] = [attri[0], attri[1]]
            else:
                prob_cost = dict()
                #--- prob, cost
                prob_cost[tuple(u)] = [attri[0], attri[1]]
                self.add_edge(f_node, t_node, prop=prob_cost)
        # ----
        for f_node in self.nodes():
            Us = set()
            for t_node in self.successors(f_node):
                prop = self[f_node][t_node]['prop']
                Us.update(set(prop.keys()))
            if Us:
                self.nodes[f_node]['act'] = Us.copy()
            else:
                print('Isolated state')
        print("-------Motion MDP Constructed-------")

    def dotify(self):
        # ----generate dot diagram for motion_mdp----
        file_dot = open('mdp.dot', 'w')
        file_dot.write('digraph mdp { \n')
        file_dot.write(
            'graph[rankdir=LR, center=true, margin=0.2, nodesep=0.1, ranksep=0.3]\n')
        file_dot.write(
            'node[shape=circle, fontname="Courier-Bold", fontsize=10, width=0.4, height=0.4, fixedsize=false]\n')
        file_dot.write('edge[arrowsize=0.6, arrowhead=vee]\n')
        for edge in self.edges:
            file_dot.write('"'+str(edge[0])+'"' +
                           '->' + '"' + str(edge[1]) + '"' + ';\n')
        file_dot.write('}\n')
        file_dot.close()
        print("-------motion_mdp.dot generated-------")
        print("Run 'dot -Tpdf motion_mdp.dot > mdp.pdf'")

    def unify_mdp(self):
        # ----verify the probability sums up to 1----
        for f_node in self.nodes():
            for u in self.nodes[f_node]['act']:
                sum_prob = 0
                N = 0
                for t_node in self.successors(f_node):
                    prop = self[f_node][t_node]['prop']
                    if u in list(prop.keys()):
                        sum_prob += prop[u][0]
                        N += 1
                if sum_prob < 1.0:
                    to_add = (1.0-sum_prob)/N
                    for t_node in self.successors(f_node):
                        prop = self[f_node][t_node]['prop']
                        if u in list(prop.keys()):
                            prop[u][0] += to_add
                if sum_prob > 1.0:
                    for t_node in self.successors(f_node):
                        prop = self[f_node][t_node]['prop']
                        if u in list(prop.keys()):
                            prop[u][0] = prop[u][0]/sum_prob
        print('Unify MDP Done')

# --------------------------------

def find_MECs(mdp, Sneg):
    # ----implementation of Alg.47 P866 of Baier08----
    print('Remaining states size', len(Sneg))
    U = mdp.graph['U']
    A = dict()
    for s in Sneg:
        A[s] = mdp.nodes[s]['act'].copy()
        if not A[s]:
            print("Isolated state")
    MEC = set()
    MECnew = set()
    MECnew.add(frozenset(Sneg))
    # ----
    k = 0
    while MEC != MECnew:
        print("<============iteration %s============>" % k)
        k += 1
        MEC = MECnew
        MECnew = set()
        print("MEC size: %s" % len(MEC))
        print("MECnew size: %s" % len(MECnew))
        for T in MEC:
            R = set()
            T_temp = set(T)
            simple_digraph = DiGraph()
            for s_f in T_temp:
                if s_f not in simple_digraph:
                    simple_digraph.add_node(s_f)
                for s_t in mdp.successors(s_f):
                    if s_t in T_temp:
                        simple_digraph.add_edge(s_f, s_t)
            print("SubGraph of one MEC: %s states and %s edges" % (
                str(len(simple_digraph.nodes())), str(len(simple_digraph.edges()))))
            i = 0
            
            for Scc in strongly_connected_components(simple_digraph):
                i += 1
                if (len(Scc) >= 1):
                    for s in Scc:
                        U_to_remove = set()
                        for u in A[s]:
                            for t in mdp.successors(s):
                                if ((u in list(mdp[s][t]['prop'].keys())) and (t not in Scc)):
                                    U_to_remove.add(u)
                        A[s].difference_update(U_to_remove)
                        if not A[s]:
                            R.add(s)
                # print(i,'number of sccs',len(Scc))
            while R:
                s = R.pop()
                T_temp.remove(s)
                for f in mdp.predecessors(s):
                    if f in T_temp:
                        A[f].difference_update(
                            set(mdp[f][s]['prop'].keys()))
                        if not A[f]:
                            R.add(f)
            j = 0
            for Scc in strongly_connected_components(simple_digraph):
                j += 1
                if (len(Scc) >= 1):
                    common = set(Scc).intersection(T_temp)
                    if common:
                        MECnew.add(frozenset(common))
    # ---------------
    print('Final MEC and MECnew size:', len(MEC))
    return MEC, A


def find_SCCs(mdp, Sneg):
    # ----simply find strongly connected components----
    print('Remaining states size', len(Sneg))
    SCC = set()
    simple_digraph = DiGraph()
    A = dict()
    for s in mdp.nodes():
        A[s] = mdp.nodes[s]['act'].copy()
    for s_f in Sneg:
        if s_f not in simple_digraph:
            simple_digraph.add_node(s_f)
        for s_t in mdp.successors(s_f):
            if s_t in Sneg:
                simple_digraph.add_edge(s_f, s_t)
    print("SubGraph of one Sf: %s states and %s edges" %
          (str(len(simple_digraph.nodes())), str(len(simple_digraph.edges()))))
    for scc in strongly_connected_components_recursive(simple_digraph):
        SCC.add(frozenset(scc))
    return SCC, A


def compute_accept_states(mdp, obstacle, target):
    S = set(mdp.nodes())
    simple_digraph = DiGraph()
    simple_digraph.add_edges_from(((v, u) for u, v in mdp.edges()))
    path = single_source_shortest_path(
        simple_digraph, random.sample(sorted(target), 1)[0])
    reachable_set = set(path.keys())
    print('States that can reach target set, size: %s' % str(len(reachable_set)))
    Sn = reachable_set.difference(obstacle)
    S_f = []
    S_fi = []
    MEC, Act = find_SCCs(mdp, Sn)
    for T in MEC:
        common = set(T.intersection(target))
        if common:
            if len(T) > 1:
                S_fi.append([T, common, Act])
                print('S_fii added to S_fi!!, size: %s' % len(T))
            if len(T) == 1:  # self-loop
                common_cp = common.copy()
                s = common_cp.pop()
                if not mdp[s][s]['prop'].keys():
                    loop_act_set = set(mdp[s][s]['prop'].keys())
                else:
                    loop_act_set = set()
                loop_act = dict()
                loop_act[s] = loop_act_set
                S_fi.append([T, common, loop_act])
                    
    if len(S_fi) > 0:
        S_f.append(S_fi)
        print("****S_fi added to S_f!!!, size: %s******" % len(S_fi))
    return S_f




