# -*- Coding: utf-8 -*-

from networkx.classes.digraph import DiGraph
from networkx import single_source_shortest_path
from ortools.linear_solver import pywraplp

from collections import defaultdict
import numpy as np
import random
import struct

def syn_plan_prefix(prod_mdp, MEC):
    # ----Synthesize optimal plan prefix to reach accepting MEC or SCC----
    print("===========[plan prefix synthesis starts]===========")
    sf = MEC[0]
    ip = MEC[1]  # force convergence to ip
    Sn = set()
    for node in prod_mdp.nodes:
        if node not in sf:
            Sn.add(node)
    v_old = dict()
    v_new = dict()
    # ----find bad states that can not reach MEC
    simple_digraph = DiGraph()
    simple_digraph.add_edges_from(((v, u) for u, v in prod_mdp.edges()))
    path = single_source_shortest_path(
        simple_digraph, random.sample(sorted(ip), 1)[0])
    reachable_set = set(path.keys())
    print('States that can reach sf, size: %s' % str(len(reachable_set)))
    Sd = Sn.difference(reachable_set)
    Sk = Sn.difference(Sd)
    print('Sn size: %s; Sd inside size: %s; Sk inside size: %s' % (len(Sn), len(Sd), len(Sk)))

    # ---------solve vi------------
    print('-----')
    print('Value iteration for prefix starts now')
    print('-----')
    for s in prod_mdp.nodes:
        if s in sf:
            v_old[s] = 1
            v_new[s] = 1
        elif s in Sd:
            v_old[s] = 0
            v_new[s] = 0
        else:
            v_old[s] = 0

    num_iteration = 0
    num_num = 0
    delta_old = 1
    for num_iteration in range(200):
        if delta_old > 10^-3:
            v_new, index, delta_new = value_iteration(prod_mdp, Sk, v_old)
            for s in Sk:                    
                v_old[s] = v_new[s]
            num_iteration += 1
            delta_old = delta_new
            num_num += 1
        else:   
            num_iteration += 1 
                
    print("Prefix Value iteration completed in interations: %s" %num_num)
    return index, v_new

def value_iteration(prod_mdp, Sr, v_old):
    num1 = len(Sr)
    U = prod_mdp.graph['U']
    num2 = len(U)
    vlist = [[0] * num2 for _ in range(num1)]
    v_new = dict()
    index = dict()
    delta = 0
    for idx, s in enumerate(Sr):
        for t in prod_mdp.successors(s):
            prop = prod_mdp[s][t]['prop'].copy()
            for u in prop.keys():
                j = list(U).index(u)
                pe = prop[u][0]
                vlist[idx][j] += pe*v_old[t]                  

        v_new[s], index[s] =  max((value, index) for index, value in enumerate(vlist[idx]))
        error = abs(v_new[s] - v_old[s])
        if error > delta:
            delta = error

    return  v_new, index, delta


def syn_plan_prefix_dfa(prod_mdp, MEC):
    # ----Synthesize optimal plan prefix to reach accepting MEC or SCC----
    print("===========[plan prefix synthesis starts]===========")
    sf = MEC
    ip = sf  # force convergence to ip
    Sn = set()
    for node in prod_mdp.nodes:
        if node not in sf:
            Sn.add(node)
    v_old = dict()
    v_new = dict()
    # ----find bad states that can not reach MEC
    simple_digraph = DiGraph()
    simple_digraph.add_edges_from(((v, u) for u, v in prod_mdp.edges()))
    path = single_source_shortest_path(
        simple_digraph, random.sample(sorted(ip), 1)[0])
    reachable_set = set(path.keys())
    print('States that can reach sf, size: %s' % str(len(reachable_set)))
    Sd = Sn.difference(reachable_set)
    Sk = Sn.difference(Sd)
    print('Sn size: %s; Sd inside size: %s; Sk inside size: %s' % (len(Sn), len(Sd), len(Sk)))

    # ---------solve vi------------
    print('-----')
    print('Value iteration for prefix starts now')
    print('-----')
    for s in prod_mdp.nodes:
        if s in sf:
            v_old[s] = 1
            v_new[s] = 1
        elif s in Sd:
            v_old[s] = 0
            v_new[s] = 0
        else:
            v_old[s] = 0

    num_iteration = 0
    num_num = 0
    delta_old = 1
    for num_iteration in range(200):
        if delta_old > 10^-3:
            v_new, index, delta_new = value_iteration(prod_mdp, Sk, v_old)
            for s in Sk:                    
                v_old[s] = v_new[s]
            num_iteration += 1
            delta_old = delta_new
            num_num += 1
        else:   
            num_iteration += 1 
                
    print("Prefix Value iteration completed in interations: %s" %num_num)
    return index, v_new

def value_iteration(prod_mdp, Sr, v_old):
    num1 = len(Sr)
    U = prod_mdp.graph['U']
    num2 = len(U)
    vlist = [[0] * num2 for _ in range(num1)]
    v_new = dict()
    index = dict()
    delta = 0
    for idx, s in enumerate(Sr):
        for t in prod_mdp.successors(s):
            prop = prod_mdp[s][t]['prop'].copy()
            for u in prop.keys():
                j = list(U).index(u)
                pe = prop[u][0]
                vlist[idx][j] += pe*v_old[t]                  

        v_new[s], index[s] =  max((value, index) for index, value in enumerate(vlist[idx]))
        error = abs(v_new[s] - v_old[s])
        if error > delta:
            delta = error

    return  v_new, index, delta
