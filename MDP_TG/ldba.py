# -*- coding: utf-8 -*-

from networkx.classes.digraph import DiGraph

from numpy import random

from .mdp import find_MECs, find_SCCs
from .automaton import OmegaAutomaton


# from .lp import act_by_plan, rd_act_by_plan

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------


class Dra(DiGraph):
    def __init__(self, formula):
        # ----call ltl2dra executable----
        automaton = OmegaAutomaton(formula)
        self.auto=automaton

        # ----parse the output----
        statenum, init,  acc = self.auto.shape[1],self.auto.q0,self.auto.acc
        # ------,
        # print(self.auto.delta[3][('base3',)],'obstacle transition?')

        DiGraph.__init__(self, type='LDBA', initial=set([init, ]), accept=acc, delta=self.auto.delta)
        print("-------LDBA Initialized-------")
        for state in range(0, statenum):
            self.add_node(state)
        self.graph['accept'] = acc
        print("-------LDBA Constructed-------")
        print("%s states and %s accepting states" %
              (str(len(self.nodes())),  str(len(acc))))

    def check_label_for_dra_edge(self, label, f_dra_node, t_dra_node):
        # ----check if a label satisfies the guards on one dra edge----
        guard_string_list = self[f_dra_node][t_dra_node]['guard_string']
        guard_int_list = []
        for st in guard_string_list:
            int_st = []
            for l in st:
                int_st.append(int(l))
            guard_int_list.append(int_st)
        for guard_list in guard_int_list:
            valid = True
            for k, ap in enumerate(self.graph['symbols']):
                if (guard_list[k] == 1) and (ap not in label):
                    valid = False
                if (guard_list[k] == 0) and (ap in label):
                    valid = False
            if valid:
                return True
        return False

    def check_distance_for_dra_edge(self, label, f_dra_node, t_dra_node):
        # ----check the distance between a label and the guards on one dra edge----
        guard_string_list = self[f_dra_node][t_dra_node]['guard_string']
        guard_int_list = []
        for st in guard_string_list:
            int_st = []
            for l in st:
                int_st.append(int(l))
            guard_int_list.append(int_st)
        Dist = []
        for guard_list in guard_int_list:
            dist = 0
            for k, ap in enumerate(self.graph['symbols']):
                if (guard_list[k] == 1) and (ap not in label):
                    dist += 1
                if (guard_list[k] == 0) and (ap in label):
                    dist += 1
            Dist.append(dist)
        return min(Dist)


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
class Product_Dra(DiGraph):
    def __init__(self, mdp, ldba):
        DiGraph.__init__(self, mdp=mdp, ldba=ldba, initial=set(),
                         accept=[], name='Product_Ldba')
        self.ldba=ldba
        self.graph['U'] = mdp.graph['U']
        print("-------Prod LDBA Initialized-------")
        self.build_full()

    def build_full(self):
        # ----construct full product----
        for f_mdp_node in self.graph['mdp']:
            for f_ldba_node in self.graph['ldba']:
                for f_mdp_label, f_label_prob in self.graph['mdp'].nodes[f_mdp_node]['label'].items():
                    f_prod_node = self.composition(
                        f_mdp_node, f_mdp_label, f_ldba_node)
                    for t_mdp_node in self.graph['mdp'].successors(f_mdp_node):
                        mdp_edge = self.graph['mdp'][f_mdp_node][t_mdp_node]
                        for t_mdp_label, t_label_prob in self.graph['mdp'].nodes[t_mdp_node]['label'].items():
                            t_ldba_node=self.ldba.auto.delta[f_ldba_node][tuple(f_mdp_label)]
                            t_prod_node = self.composition(
                                t_mdp_node, t_mdp_label, t_ldba_node)
                            prob_cost = dict()
                            for u, attri in mdp_edge['prop'].items():
                                if t_label_prob*attri[0] != 0:
                                    prob_cost[u] = (
                                        t_label_prob*attri[0], attri[1])
                            if list(prob_cost.keys()):
                                self.add_edge(
                                    f_prod_node, t_prod_node, prop=prob_cost)
                                        
        self.build_acc()
        print("-------Prod LDBA Constructed-------")
        print("%s states, %s edges and %s accepting states" % (
            str(len(self.nodes())), str(len(self.edges())), str(len(self.graph['accept']))))

    def composition(self, mdp_node, mdp_label, ldba_node):
        prod_node = (mdp_node, mdp_label, ldba_node)
        if not self.has_node(prod_node):
            Us = self.graph['mdp'].nodes[mdp_node]['act'].copy()
            self.add_node(prod_node, mdp=mdp_node,
                          label=mdp_label, ldba=ldba_node, act=Us)
            if ((mdp_node == self.graph['mdp'].graph['init_state']) and
                    (ldba_node in self.graph['ldba'].graph['initial'])):
                self.graph['initial'].add(prod_node)
                print("Initial node added:")
                print(self.graph['initial'])
        return prod_node

    def build_acc(self):
        acc_product=set([prod_n for prod_n in self.nodes() if self.ldba.auto.acc[prod_n[2]][tuple(prod_n[1])][0]])
        self.graph['accept'] = acc_product


    def compute_S_f(self):
        # ----find all accepting End components----
        S = set(self.nodes())
        Ip = self.graph['accept']
        S_f = []
        k = 1
        f_prod_accept = open('./prod_node_accept.dat','w')
        f_prod_accept_noact = open('./prod_node_accept_noact.dat','w')
        f_prod_common = open('./prod_node_common.dat','w')
        f_prod_Ip = open('./prod_node_Ip.dat','w')
        # ---for each accepting states
        print("+++++++++++++++++++++++++++++++++++++")
        print("++++++++++++ acc_pair %s ++++++++++++" % k)
        print("+++++++++++++++++++++++++++++++++++++")
        S_fi = []
        print("Ip size: %s" % len(Ip))
        for np in list(Ip):
            f_prod_Ip.write('%s, %s, %s, %s\n' %(np[0][0], np[0][1], np[0][2], np[2]))
        MEC, Act = find_MECs(self, S)
        print(len(MEC))
        # ---find accepting ones
        for T in MEC:
            common = set(T.intersection(Ip))
            for j in list(common):
                f_prod_common.write('%s, %s, %s, %s\n' %(j[0][0], j[0][1], j[0][2], j[2]))
            if common:
                if len(T) > 1:
                    S_fi.append([T, common, Act])
                    print('S_fii added to S_fi!!, size: %s' % len(T))
                    for i in list(T):
                        f_prod_accept.write('%s, %s, %s, %s, %s\n' %(i[0][0], i[0][1], i[0][2], i[2], Act[i]))
                        f_prod_accept_noact.write('%s, %s, %s, %s\n' %(i[0][0], i[0][1], i[0][2], i[2]))
                if len(T) == 1:  # self-loop
                    common_cp = common.copy()
                    s = common_cp.pop()
                    loop_act_set = set(self[s][s]['prop'].keys())
                    loop_act = dict()
                    loop_act[s] = loop_act_set
                    S_fi.append([T, common, loop_act])
                    for i in list(T):
                        f_prod_accept.write('%s, %s, %s, %s, %s\n' %(i[0][0], i[0][1], i[0][2], i[2], Act[i]))
                        f_prod_accept_noact.write('%s, %s, %s, %s\n' %(i[0][0], i[0][1], i[0][2], i[2]))

        if len(S_fi) > 0:
            S_f.append(S_fi)
            print("****S_fi added to S_f!!!, size: %s******" % len(S_fi))
        k += 1

        self.Sf = S_f
        f_prod_accept.close()
        f_prod_Ip.close()
        f_prod_common.close()
        f_prod_accept_noact.close()

        if S_f:
            print("-------Accepting MEC for Prod LDBA Computed-------")
            print("acc number: %s" % str(k-1))
            print("Sf AMEC number: %s" % len(S_f))
        else:
            print("No accepting ECs found!")
            print("Check your MDP and Task formulation")
            print("Or try the relaxed plan")
        

    def compute_S_f_rex(self):
        # ----find accepting SCC for rex plans----
        S = set(self.nodes())
        acc_pairs = self.graph['accept']
        S_f = []
        k = 1
        for pair in acc_pairs:
            print("+++++++++++++++++++++++++++++++++++++")
            print("++++++++++++ acc_pair %s ++++++++++++" % k)
            print("+++++++++++++++++++++++++++++++++++++")
            S_fi = []
            Ip = pair[0]
            Hp = pair[1]
            print("Ip size: %s" % len(Ip))
            print("Hp size: %s" % len(Hp))
            MEC, Act = find_SCCs(self, S.difference(Hp))
            for T in MEC:
                common = set(T.intersection(Ip))
                if common:
                    if len(T) > 1:
                        S_fi.append([T, common, Act])
                        print('S_fii added to S_fi!!, size: %s' % len(T))
                    if len(T) == 1:  # self-loop
                        common_cp = common.copy()
                        s = common_cp.pop()
                        if s in self.successors(s):
                            loop_act_set = set(self[s][s]['prop'].keys())
                            loop_act = dict()
                            loop_act[s] = loop_act_set
                            S_fi.append([T, common, loop_act])
                            print('S_fii added to S_fi!!, size: %s' % len(T))
            if len(S_fi) > 0:
                S_f.append(S_fi)
                print("****S_fi added to S_f!!!, size: %s******" % len(S_fi))
            k += 1
        self.Sf = S_f
        if S_f:
            print("-------Accepting SCC for Prod DRA Computed-------")
            print("acc_pair number: %s" % str(k-1))
            print("Sf number: %s" % len(S_f))
        else:
            print("No accepting SCC found")
            print("Check your MDP and Task formulation")

    

    def dotify(self):
        # ----generate dot diagram for the product automaton----
        file_dot = open('product_dra.dot', 'w')
        file_dot.write('digraph prodDRA { \n')
        file_dot.write(
            'graph[rankdir=LR, center=true, margin=0.2, nodesep=0.1, ranksep=0.3]\n')
        file_dot.write(
            'node[shape=circle, fontname="Courier-Bold", fontsize=10, width=0.4, height=0.4, fixedsize=false]\n')
        file_dot.write('edge[arrowsize=0.6, arrowhead=vee]\n')
        for edge in self.edges:
            file_dot.write('"'+str(edge[0])+'"' +
                           '->' + '"' + str(edge[1]) + '"' + ';\n')
        for acc_pairs in self.graph['accept']:
            I = acc_pairs[0]
            H = acc_pairs[1]
            for i in I:
                file_dot.write(
                    '"'+str(i)+'"'+'[style=filled, fillcolor=green]'+';\n')
            for h in H:
                file_dot.write(
                    '"'+str(h)+'"'+'[style=filled, fillcolor=red]'+';\n')
        file_dot.write('}\n')
        file_dot.close()
        print("-------produc_dra.dot generated-------")
        print("Run 'dot -Tpdf product_dra.dot > prod.pdf'")

    def execution(self, best_all_plan, total_T, state_seq, label_seq):
        # ----plan execution with or without given observation----
        t = 0
        X = []
        L = []
        U = []
        M = []
        PX = []
        m = 0
        # ----
        while (t <= total_T):
            if (t == 0):
                # print '---initial run----'
                mdp_state = state_seq[0]
                label = label_seq[0]
                initial_set = self.graph['initial'].copy()
                current_state = initial_set.pop()
            elif (t >= 1) and (len(state_seq) > t):
                # print '---observation given---'
                mdp_state = state_seq[t]
                label = label_seq[t]
                prev_state = tuple(current_state)
                error = True
                for next_state in self.successors(prev_state):
                    if((self.nodes[next_state]['mdp'] == mdp_state) and (self.nodes[next_state]['label'] == label) and (u in list(self[prev_state][next_state]['prop'].keys()))):
                        current_state = tuple(next_state)
                        error = False
                        break
                if error:
                    print(
                        'Error: The provided state and label sequences do NOT match the mdp structure!')
                    break
            else:
                # print '---random observation---'
                prev_state = tuple(current_state)
                S = []
                P = []
                if m != 2:  # in prefix or suffix
                    for next_state in self.successors(prev_state):
                        prop = self[prev_state][next_state]['prop']
                        if (u in list(prop.keys())):
                            S.append(next_state)
                            P.append(prop[u][0])
                if m == 2:  # in bad states
                    # print 'in bad states'
                    Sd = best_all_plan[2][3]
                    Sf = best_all_plan[2][0]
                    Sr = best_all_plan[2][2]
                    (xf, lf, qf) = prev_state
                    postqf = self.graph['dra'].successors(qf)
                    for xt in self.graph['mdp'].successors(xf):
                        if xt != xf:
                            prop = self.graph['mdp'][xf][xt]['prop']
                            if u in list(prop.keys()):
                                prob_edge = prop[u][0]
                                label = self.graph['mdp'].nodes[xt]['label']
                                for lt in label.keys():
                                    prob_label = label[lt]
                                    dist = dict()
                                    for qt in postqf:
                                        if (xt, lt, qt) in Sf.union(Sr):
                                            dist[qt] = self.graph['dra'].check_distance_for_dra_edge(
                                                lf, qf, qt)
                                    if list(dist.keys()):
                                        qt = min(list(dist.keys()),
                                                 key=lambda q: dist[q])
                                        S.append((xt, lt, qt))
                                        P.append(prob_edge*prob_label)
                rdn = random.random()
                pc = 0
                for k, p in enumerate(P):
                    pc += p
                    if pc > rdn:
                        break
                if len(S)>=k+1:
                    current_state = tuple(S[k])
                    mdp_state = self.nodes[current_state]['mdp']
                    label = self.nodes[current_state]['label']
            # ----
            u, m = act_by_plan(self, best_all_plan, current_state)
            X.append(mdp_state)
            PX.append(current_state)
            L.append(set(label))
            U.append(u)
            M.append(m)
            t += 1
        return X, L, U, M, PX

    def rd_execution(self, best_all_plan, total_T, state_seq, label_seq):
        # ----plan execution with or without given observation----
        # ----Round-robin policy as the plan suffix----
        print('Round-robin policy for suffix')
        t = 0
        X = []
        L = []
        U = []
        M = []
        PX = []
        # memory needed for round-robin
        I = dict()
        for s in self.nodes():
            I[s] = 0
        # ----
        while (t <= total_T):
            if (t == 0):
                # print '---initial run----'
                mdp_state = state_seq[0]
                label = label_seq[0]
                initial_set = self.graph['initial'].copy()
                current_state = initial_set.pop()
            elif (t >= 1) and (len(state_seq) > t):
                # print '---observation given---'
                mdp_state = state_seq[t]
                label = label_seq[t]
                prev_state = tuple(current_state)
                error = True
                for next_state in self.successors(prev_state):
                    if((self.nodes[next_state]['mdp'] == mdp_state) and (self.nodes[next_state]['label'] == label) and (u in list(self[prev_state][next_state]['prop'].keys()))):
                        current_state = tuple(next_state)
                        error = False
                        break
                if error:
                    print(
                        'Error: The provided state and label sequences do NOT match the mdp structure!')
                    break
            else:
                # print '---random observation---'
                prev_state = tuple(current_state)
                S = []
                P = []
                for next_state in self.successors(prev_state):
                    prop = self[prev_state][next_state]['prop']
                    if (u in list(prop.keys())):
                        S.append(next_state)
                        P.append(prop[u][0])
                rdn = random.random()
                pc = 0
                for k, p in enumerate(P):
                    pc += p
                    if pc > rdn:
                        break
                current_state = tuple(S[k])
                mdp_state = self.nodes[current_state]['mdp']
                label = self.nodes[current_state]['label']
            # ----
            u, m, I = rd_act_by_plan(self, best_all_plan, current_state, I)
            X.append(mdp_state)
            PX.append(current_state)
            L.append(set(label))
            U.append(u)
            M.append(m)
            t += 1
        return X, L, U, M, PX
