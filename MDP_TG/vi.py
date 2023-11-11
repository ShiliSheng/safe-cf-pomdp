# -*- Coding: utf-8 -*-

from networkx.classes.digraph import DiGraph
from networkx import single_source_shortest_path
from ortools.linear_solver import pywraplp

from collections import defaultdict
import numpy as np
import random
import struct

def syn_full_plan(prod_mdp, gamma):
    # ----Optimal plan synthesis, total cost over plan prefix and suffix----
    f_plan_prefix_dict = open('to_matlab/data/plan_prefix.dat','w')
    f_plan_probability_dict = open('to_matlab/data/plan_probability.dat','w')
    f_plan_cost_dict = open('to_matlab/data/plan_cost.dat','w')
    f_plan_total_dict = open('to_matlab/data/plan_total.dat','w')
    f_plan_suffix_dict = open('to_matlab/data/plan_suffix.dat','w')
    f_plan_suffix_cost_dict = open('to_matlab/data/plan_suffix_cost.dat','w')
    print("==========[Optimal full plan synthesis start]==========")
    Plan = []
    U = prod_mdp.graph['U']
    for l, S_fi in enumerate(prod_mdp.Sf):
        print("---for one S_fi---")
        plan = []
        beta = 1
        inf_cost = 10^4
        for k, MEC in enumerate(S_fi):
            # suffix_cost, plan_suffix = syn_plan_suffix(prod_mdp, MEC, inf_cost)
            # for s in plan_suffix.keys():
            #     index = plan_suffix[s]
            #     f_plan_suffix_dict.write('%s, %s\n' %(s, index)) 
            sf = MEC[0]
            suffix_cost = dict()
            risk = dict()
            plan_suffix = dict()
            for s0 in sf:
                suffix_cost[s0], risk[s0], plan_suffix[s0] = syn_plan_suffix_lp(prod_mdp, MEC, s0)
            for s in suffix_cost.keys():
                index = suffix_cost[s]
                f_plan_suffix_cost_dict.write('%s, %s\n' %(s, index)) 

            plan_prefix, vtotal, v, vcost, Sr, Sd = syn_plan_prefix(
                prod_mdp, MEC, beta, suffix_cost, inf_cost)
            for s in plan_prefix.keys():
                index = plan_prefix[s]
                prob = v[s]
                cost = vcost[s]
                total = vtotal[s]
                f_plan_prefix_dict.write('%s, %s\n' %(s, index))
                f_plan_probability_dict.write('%s, %s\n' %(s, prob))
                f_plan_cost_dict.write('%s, %s\n' %(s, cost))
                f_plan_total_dict.write('%s, %s\n' %(s, total))
            for init_node in prod_mdp.graph['initial']:
                print("Best plan obtained, cost: %s, risk %s" %
                    (str(vcost[init_node]), 1-str(v[init_node])))
                plan.append([[plan_prefix, vtotal[init_node], v[init_node], vcost[init_node]], [suffix_cost]])
            
        if plan:
            best_k_plan = max(plan, key=lambda p: p[0][1])
            Plan.append(best_k_plan)
        else:
            print("No valid found!")
    if Plan:
        print("=========================")
        print(" || Final compilation  ||")
        print("=========================")
        best_all_plan = max(Plan, key=lambda p: p[0][1])
        print('cost: %s; risk: %s ' %
              (best_all_plan[0][3], 1-best_all_plan[0][2]))
        return best_all_plan
    else:
        print("No valid plan found")
        return None

def syn_plan_prefix(prod_mdp, MEC, beta, suffix_cost, inf_cost):
    # ----Synthesize optimal plan prefix to reach accepting MEC or SCC----
    # ----with bounded risk and minimal expected total cost----
    f_prod_Sr = open('to_matlab/data/prod_prefix_Sr.dat','w')
    f_prod_Sf = open('to_matlab/data/prod_prefix_Sf.dat','w')
    print("===========[plan prefix synthesis starts]===========")
    sf = MEC[0]
    ip = MEC[1]  # force convergence to ip
    Sk = set()
    for node in prod_mdp.nodes:
        if node not in sf:
            Sk.add(node)
    delta = 0.01
    v_old = dict()
    vcost_old = dict()
    vtotal_old = dict()
    v_new = dict()
    vcost_new = dict()
    vtotal_new = dict()
    for init_node in prod_mdp.graph['initial']:
        path_init = single_source_shortest_path(prod_mdp, init_node)
        print('Reachable from init size: %s' % len(list(path_init.keys())))
        if not set(path_init.keys()).intersection(sf):
            print("Initial node can not reach sf")
            return None, None, None, None, None
        Sn = set(path_init.keys()).difference(sf)
        # ----find bad states that can not reach MEC
        simple_digraph = DiGraph()
        simple_digraph.add_edges_from(((v, u) for u, v in prod_mdp.edges()))
        path = single_source_shortest_path(
            simple_digraph, random.sample(sorted(ip), 1)[0])
        reachable_set = set(path.keys())
        print('States that can reach sf, size: %s' % str(len(reachable_set)))
        Sd = Sn.difference(reachable_set)
        Sr = Sn.intersection(reachable_set)

        # #--------------
        print('Sn size: %s; Sd inside size: %s; Sr inside size: %s' %
              (len(Sn), len(Sd), len(Sr)))
        for np in list(Sr):
            f_prod_Sr.write('%s, %s\n' %(np[0], np[2]))
        for hp in list(sf):
            f_prod_Sf.write('%s, %s\n' %(hp[0], hp[2])) 
        # ---------solve vi------------
        print('-----')
        print('Value iteration for prefix starts now')
        print('-----')
        for s in prod_mdp.nodes:
            if s in sf:
                v_old[s] = 1
                v_new[s] = 1
                if s in suffix_cost.keys():
                    vcost_old[s] = suffix_cost[s]
                    vcost_new[s] = suffix_cost[s]
                vtotal_old[s] = beta*v_old[s] - (1-beta)*vcost_old[s]
                vtotal_new[s] = beta*v_new[s] - (1-beta)*vcost_new[s]
            else:
                v_old[s] = 0
                vcost_old[s] = inf_cost
                vtotal_old[s] = beta*v_old[s] - (1-beta)*vcost_old[s]

        num_iteration = 0
        num_num = 0
        delta_old = 1
        for num_iteration in range(100):
            if delta_old > 10^-5:
                vtotal_new, v_new, vcost_new, index_prefix, delta_new, delta_cost = value_iteration(prod_mdp, Sk, v_old, vcost_old, beta, inf_cost)
                for s in prod_mdp.nodes:
                    if s in Sk:                    
                        v_old[s] = v_new[s]
                        vcost_old[s] = vcost_new[s]
                        vtotal_old[s] = vtotal_new[s]
                num_iteration += 1
                delta_old = delta_new
                num_num += 1
            else:   
                num_iteration += 1 
                
        print("Prefix Value iteration completed in interations: %s" %num_num)
        return index_prefix, vtotal_new, v_new, vcost_new, Sr, Sd

def value_iteration(prod_mdp, Sr, v_old, vcost_old, beta, inf_cost):
    num1 = len(Sr)
    U = prod_mdp.graph['U']
    num2 = len(U)
    vlist = [[0] * num2 for _ in range(num1)]
    vlist_cost = [[inf_cost] * num2 for _ in range(num1)]
    vlist_total = [[beta*0-(1-beta)*inf_cost] * num2 for _ in range(num1)]
    v_new = dict()
    vcost_new =dict()
    vtotal_new =dict()
    index = dict()
    delta = 0
    delta_cost = 0
    for idx, s in enumerate(Sr):
        for idu, u in enumerate(U):
            vlist_cost[idx][idu] = ce

        for t in prod_mdp.successors(s):
            prop = prod_mdp[s][t]['prop'].copy()
            for u in prop.keys():
                j = list(U).index(u)
                pe = prop[u][0]
                ce = prop[u][1]
                vlist[idx][j] += pe*v_old[t]
                vlist_cost[idx][j] += pe*vcost_old[t]                    
            else:
                print(f"{u} is not in the set U.")

    for idx, s in enumerate(Sr):
        for idu, u in enumerate(U): 
            vlist_total[idx][idu] = beta*vlist[idx][idu] - (1-beta)*vlist_cost[idx][idu]

        vtotal_new[s], index[s] =  max((value, index) for index, value in enumerate(vlist_total[idx]))
        v_new[s] = vlist[idx][index[s]]
        vcost_new[s] = vlist_cost[idx][index[s]]

        error = abs(v_new[s] - v_old[s])
        error_cost = abs(vcost_new[s] - vcost_old[s])
        if error > delta:
            delta = error
        if error_cost > delta_cost:
            delta_cost = error_cost

    return vtotal_new, v_new, vcost_new, index, delta, delta_cost

def syn_plan_suffix_lp(prod_mdp, MEC, s0):
    # ----Synthesize optimal plan suffix to stay within the accepting MEC----
    # ----with minimal expected total cost of accepting cyclic paths----
    print("===========[plan suffix synthesis starts]")
    sf = MEC[0]
    ip = MEC[1]
    act = MEC[2].copy()
    delta = 0.01
    gamma = 0.00
    for init_node in prod_mdp.graph['initial']:
        paths = single_source_shortest_path(prod_mdp, init_node)
        Sn = set(paths.keys()).intersection(sf)
        print('Sf size: %s' % len(sf))
        print('reachable sf size: %s' % len(Sn))
        print('Ip size: %s' % len(ip))
        print('Ip and sf intersection size: %s' % len(Sn.intersection(ip)))
        # ---------solve lp------------
        print('------')
        print('ORtools for suffix starts now')
        print('------')
        if 1:
            Y = defaultdict(float)
            suffix_solver = pywraplp.Solver.CreateSolver('GLOP')
            # create variables
            for s in Sn:
                for u in act[s]:
                    Y[(s, u)] = suffix_solver.NumVar(0, 1000, 'y[(%s, %s)]' % (s, u))
            print('Variables added: %d' % len(Y))
            # set objective
            obj = 0
            for s in Sn:
                for u in act[s]:
                    for t in prod_mdp.successors(s):
                        prop = prod_mdp[s][t]['prop'].copy()
                        if u in list(prop.keys()):
                            pe = prop[u][0]
                            ce = prop[u][1]
                            obj += Y[(s, u)]*pe*ce
            suffix_solver.Minimize(obj)
            print('Objective added')
            # add constraints
            # --------------------
            for s in Sn:
                constr3 = 0
                constr4 = 0
                for u in act[s]:
                    constr3 += Y[(s, u)]
                for f in prod_mdp.predecessors(s):
                    if (f in Sn) and (s not in ip):
                        prop = prod_mdp[f][s]['prop'].copy()
                        for uf in act[f]:
                            if uf in list(prop.keys()):
                                constr4 += Y[(f, uf)]*prop[uf][0]
                            else:
                                constr4 += Y[(f, uf)]*0.00
                    if (f in Sn) and (s in ip) and (f != s):
                        prop = prod_mdp[f][s]['prop'].copy()
                        for uf in act[f]:
                            if uf in list(prop.keys()):
                                constr4 += Y[(f, uf)]*prop[uf][0]
                            else:
                                constr4 += Y[(f, uf)]*0.00
                if (s in s0) and (s not in ip):
                    suffix_solver.Add(constr3 == constr4 + 1)
                if (s in s0) and (s in ip):
                    suffix_solver.Add(constr3 == 1)
                if (s not in s0) and (s not in ip):
                    suffix_solver.Add(constr3 == constr4)
            print('Balance condition added')
            print('Initial sf condition added')
            # --------------------
            y_to_ip = 0.0
            y_out = 0.0
            for s in Sn:
                for t in prod_mdp.successors(s):
                    if t not in Sn:
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                pe = prop[u][0]
                                y_out += Y[(s, u)]*pe
                    elif t in ip:
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                pe = prop[u][0]
                                y_to_ip += Y[(s, u)]*pe
            # suffix_solver.Add(y_to_ip+y_out >= delta)
            suffix_solver.Add(y_to_ip >= (1.0-gamma-delta)*(y_to_ip+y_out))
            print('Risk constraint added')
            # ------------------------------
            # solve
            print('--optimization for suffix starts--')
            status = suffix_solver.Solve()
            if status == pywraplp.Solver.OPTIMAL:
                print('Solution:')
                print('Objective value =', suffix_solver.Objective().Value())
                print('Advanced usage:')
                print('Problem solved in %f milliseconds' %
                      suffix_solver.wall_time())
                print('Problem solved in %d iterations' %
                      suffix_solver.iterations())
            else:
                print('The problem does not have an optimal solution.')
                return None, None, None
            
            plan = []
            norm = 0
            U = []
            P = []
            for u in act[s0]:
                norm += Y[(s0, u)].solution_value()
            for u in act[s0]:
                U.append(u)
                if norm > 0.01:
                    P.append(Y[(s0, u)].solution_value()/norm)
                else:
                    P.append(1.0/len(act[s0]))
            plan = [U, P]
            print("----Suffix plan added")
            cost = suffix_solver.Objective().Value()
            print("----Suffix cost computed")
            # compute risk given the plan suffix
            risk = 0.0
            y_to_ip = 0.0
            y_out = 0.0
            for s in Sn:
                for t in prod_mdp.successors(s):
                    if t not in Sn:
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                pe = prop[u][0]
                                y_out += Y[(s, u)].solution_value()*pe
                    elif t in ip:
                        prop = prod_mdp[s][t]['prop'].copy()
                        for u in prop.keys():
                            if u in act[s]:
                                pe = prop[u][0]
                                y_to_ip += Y[(s, u)].solution_value()*pe
            if (y_to_ip+y_out) > 0:
                risk = y_out/(y_to_ip+y_out)
            print('y_out: %s; y_to_ip+y_out: %s' % (y_out, y_to_ip+y_out))
            print("----Suffix risk computed")
            return cost, risk, plan

def syn_plan_suffix(prod_mdp, MEC, inf_cost):
    # ----Synthesize optimal plan suffix to stay within the accepting MEC----
    # ----with minimal expected total cost of accepting cyclic paths----
    f_prod_suffix_Ip = open('to_matlab/data/prod_suffix_Ip.dat','w')

    print("===========[plan suffix synthesis starts]")
    cost_old = dict() 
    cost_new = dict()
    sf = MEC[0]
    ip = MEC[1]
    for lip in list(ip):
        f_prod_suffix_Ip.write('%s, %s\n' %(lip[0], lip[2]))
    # ip_in = []
    # for s in ip:
    #     ip_in.append([s, 'in'])
    act = MEC[2].copy()
    # ---------solve vi------------
    print('-----')
    print('Value iteration for suffix starts now')
    print('-----')
    for s in sf:
        cost_old[s] = inf_cost
    num_iteration = 0
    num_num = 0
    delta_cost_old = 1
    for num_iteration in range(5):
        if delta_cost_old > 10^-3:
            cost_new, delta_cost_new, index_suffix = value_iteration_cost(prod_mdp, sf, act, ip, cost_old, inf_cost)
            for s in sf:
                cost_old[s] = cost_new[s]
            num_iteration += 1
            delta_cost_old = delta_cost_new
            num_num += 1
        else:
            num_iteration += 1

    print("Suffix cost Value iteration completed in interations: %s" %num_num)
    return cost_new, index_suffix


def value_iteration_cost(prod_mdp, Sf, act, ip, cost_old, inf_cost):
    C = [2, 4, 3, 3, 1]
    num1 = len(Sf)
    U = prod_mdp.graph['U']
    num2 = len(U)
    vlist_cost = [[inf_cost] * num2 for _ in range(num1)]
    vcost_old =dict()
    cost_new =dict()
    index = dict()
    delta = 0
    Sn = Sf.difference(ip)
    for s in Sf:
        for t in prod_mdp.successors(s):
            if s in Sn and t in Sn:
                vcost_old[t] = cost_old[t]
            elif s in Sn and t in ip:
                vcost_old[t] = 0
            elif s in ip and t in Sn:
                vcost_old[t] = cost_old[t]
            else:
                vcost_old[t] = 0

    for idx, s in enumerate(Sf):
        for u in act[s]:
            j = list(U).index(u)
            ce = C[j]
            vlist_cost[idx][j] = ce

        for t in prod_mdp.successors(s):
            prop = prod_mdp[s][t]['prop'].copy()
            for u in prop.keys():
                if u in act[s]:
                    j = list(U).index(u)
                    pe = prop[u][0]
                    ce = prop[u][1]
                    vlist_cost[idx][j] += pe*vcost_old[t]
                else:
                    print(f"{u} is not in the set U.")          
                
        index[s], cost_new[s] =  min((index, value) for index, value in enumerate(vlist_cost[idx]))
        #print(cost_new)
        error = abs(cost_new[s] - cost_old[s])
        if error > delta:
            delta = error
        print(delta)

    return cost_new, delta, index

def syn_plan_bad(prod_mdp, state_types):
    f_state_bad_Sd = open('to_matlab/data/state_bad_Sn.dat','w')
    Sf = state_types[0]
    Sr = state_types[2]
    Sd = state_types[3]
    plan_bad = dict()
    for sd in Sd:
        # print 'Sd size',len(Sd)
        # print 'Sf size',len(Sf)
        # print 'Sr size',len(Sr)
        f_state_bad_Sd.write('%s, %s, %s, %s\n' %(sd[0][0], sd[0][1], sd[0][2], sd[2])) 
        (xf, lf, qf) = sd
        Ud = prod_mdp.nodes[sd]['act'].copy()
        proj_cost = dict()
        postqf = prod_mdp.graph['dra'].successors(qf)
        # print 'postqf size',len(postqf)
        for xt in prod_mdp.graph['mdp'].successors(xf):
            if xt != xf:
                prop = prod_mdp.graph['mdp'][xf][xt]['prop']
                for u in prop.keys():
                    prob_edge = prop[u][0]
                    cost = prop[u][1]
                    label = prod_mdp.graph['mdp'].nodes[xt]['label']
                    for lt in label.keys():
                        prob_label = label[lt]
                        dist = dict()
                        for qt in postqf:
                            if (xt, lt, qt) in Sf.union(Sr):
                                dist[qt] = prod_mdp.graph['dra'].check_distance_for_dra_edge(
                                    lf, qf, qt)
                        if list(dist.keys()):
                            qt = min(list(dist.keys()), key=lambda q: dist[q])
                            if u not in list(proj_cost.keys()):
                                proj_cost[u] = 0
                            else:
                                proj_cost[u] += prob_edge*prob_label*dist[qt]
        # policy for bad states
        U = []
        P = []
        if list(proj_cost.keys()):
            # print 'sd',sd
            u_star = min(list(proj_cost.keys()), key=lambda u: proj_cost[u])
            for u in Ud:
                U.append(u)
                if u == u_star:
                    P.append(1)
                else:
                    P.append(0)
        else:
            for u in Ud:
                U.append(u)
                P.append(1.0/len(Ud))
        plan_bad[sd] = [U, P]
    return plan_bad