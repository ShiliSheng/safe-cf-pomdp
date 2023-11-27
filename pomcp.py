from MDP_TG.mdp import Motion_MDP_label, Motion_MDP, compute_accept_states
from MDP_TG.dra import Dra, Dfa, Product_Dra, Product_Dfa
from MDP_TG.vi4wr import syn_plan_prefix, syn_plan_prefix_dfa
from networkx.classes.digraph import DiGraph
import pickle
import time
import math
import random
from model import Model
from collections import defaultdict
# class POMCPBelief:
#     def __init__(self) -> None:
#         self.particles = []
#         self.uniqueStates = set()
    
#     def copy(self, orignal):
#         self.particles = [x for x in orignal.particles]
#         self.uniqueStates = set(self.particles)
    
#     def empty(self):
#         return not self.particles

#     def is_depleted(self):
#         return not self.particles
    
#     def add_particle(self, state):
#         self.particles.append(state)
#         self.uniqueStates.add(state)

#     def is_state_in_belief_support(self, state):
#         return state in self.uniqueStates
    
#     def update_belief_suport(self, state):
#         self.uniqueStates.add(state)

#     def size(self):
#         return len(self.particles)
ON_THE_FLY = 1
class POMCPNode:
    def __init__(self) -> None:
        self.belief = defaultdict(int)
        self.belief_particles = []
        self.clear()

    def clear(self):
        self.parent = None
        self.children = {}
        self.h = -1
        self.isQNode = False
        self.v = 0
        self.n = 0
        self.time = 0
        self.illegalActionIndexes = set()
    
    def add_particle(self, state):
        self.belief[state] = self.belief.get(state, 0) + 1;

    def add_illegal_action_index(self, actionIndex):
        self.remove_child(actionIndex)
        self.illegalActionIndexes.add(actionIndex)
    
    def is_action_index_illegal(self, actionIndex):
        return actionIndex in self.illegalActionIndexes

    def set_h_action(self, isAction):
        self.isQNode = isAction

    def getH(self):
        return self.h
    
    def getParent(self):
        return self.parent
    
    def getV(self):
        return self.v / self.n
    
    def increaseV(self, value):
        self.v += value
        self.n += 1

    def add_child(self, node, index):
        self.children[index] = node
    
    def remove_child(self, index):
        if index not in self.children: return
        qchild = self.children[index]
        if qchild:
            self.v -= qchild.v
            self.n -= qchild.n
        del self.children[index]
        if len(self.children) == 0 and self.parent :
            qParent = self.parent
            vparent = qParent.parent
            vparent.remove_child(qParent.h)
        
    def get_child_by_action_index(self, index):
        if index in self.children: return self.children[index]
        child = POMCPNode()
        child.h = index
        child.parent = self
        self.add_child(child, index)
        return child

    def check_child_by_observation_index(self, index):
        return index in self.children
    
    def get_child_by_observation_index(self, index):
        if index not in self.children:
            child = POMCPNode()
            child.h = index
            child.parent = self
            self.add_child(child, index)
        return self.children[index]
    
    def sample_state_from_belief(self):
        # return random.choice(self.belief_particles)
        states, counts = zip(*self.belief.items())
        total_counts = sum(cnt for cnt in counts)
        probs = (cnt / total_counts for cnt in counts)
        return random.choices(states, weights=probs, k=1)[0]

    def get_belief_suport(self):
        return self.belief.keys()
    
    def get_time(self):
        return self.time

    def have_state_in_belief_support(self, state):
        return state in self.belief
    
class POMCP:
    def __init__(self, pomdp, constant = 1000, maxDepth = 100, end_states = set(), target = set(), horizon = 5):

        # def __init__(self, initial_belief, actions, robot_state_action_map, state_to_observation, state_action_reward_map, 
        #              end_states, constant = 1000, maxDepth = 100, targets = set()):
        #e (float): Threshold value below which the expected sum of discounted rewards for the POMDP is considered 0. Default value is 0.005.
        # c (float): Parameter that controls the importance of exploration in the UCB heuristic. Default value is 1.
        # no_particles (int): Controls the maximum number of particles that will be kept at each node 
        #                       and the number of particles that will be sampled from the posterior belief after an action is taken.
        self.numSimulations = 2 ** 6
        self.gamma = 0.95
        self.e = 0.05
        self.noParticles = 1200
        self.K = 10000
        self.TreeDepth = 0
        self.PeakTreeDepth = 0
        self.c = constant
        self.maxDepth = maxDepth
        self.verbose = 1
        self.pomdp = pomdp
        self.target = target
        self.end_states = end_states

        self.root = None
        self.is_min = False
        self.stateOfInteste = None
        self.varNames = []
        self.variabelIndexX = None
        self.variableIndexY = None
        self.stateSuccessorsHashSet = {}
        self.stateSuccessorArryList = {}
        self.stateSuccessorCumProb = {}
        self.shiledLevel = 1
        self.horizon = 5
        self.initializePOMCP()

    def initializePOMCP(self):
        self.TreeDepth = 0
        self.PeakTreeDepth = 0
        self.initialUCB(1000, 100)
        # this.shieldLevel = NO_SHIELD; 
        # 		this.useLocalShields = false;

    def fastUCB(self, N, n, logN):
        if N < 1000 and n < 100: return self.UCB[N][n]
        if n == 0: return float("inf")
        return (logN / n) ** 0.5 * self.c
    
    def initialUCB(self, UCB_N, UCB_n):
        self.UCB = [[0] * UCB_n for _ in range(UCB_N)]
        for N in range(UCB_N):
            for n in range(UCB_n):
                if n == 0: self.UCB[N][n] = float("inf")
                else: self.UCB[N][n] = math.log(N + 1) / n
    
    def set_num_simulations(self, n):
        self.num_simulations = n

    def set_verbose(self, v):
        self.verbose = v

    def set_root(self, node):
        self.root = node

    def reset_root(self):
        self.root = POMCPNode()
        for key, prob in self.pomdp.initial_belief.items():
            self.root.belief[key] = prob * self.K
        # for state, state_fre in self.root.belief.items():
        #     for _ in range(state_fre):
        #         self.root.belief_particles.append(state)

    def draw_from_probabilities(self, probabilities):
        states, probs = zip(*probabilities.items())
        next_state = random.choices(states, weights=probs, k=1)[0]
        return next_state
    
    def step(self, state, actionIndex):
        probabilities = self.pomdp.robot_state_action_map[state][actionIndex]
        states, probs = zip(*probabilities.items())
        next_state = random.choices(states, weights=probs, k=1)[0]
        return next_state
    
    def get_observation(self, state):
        if (state not in self.pomdp.state_observation_map):
            print("error")
        return self.pomdp.state_observation_map[state]
    
    def invigorate_belief(self, parent, child, action_index, obs):
        # fill child belief with particles
        child_belief_size = sum(cnt for cnt in child.belief.values())

        states, counts = zip(*self.root.belief.items())
        sum_count = sum(counts)
        probs = [cnt / sum_count for cnt in counts]

        while child_belief_size < self.K:
            # s = parent.sample_state_from_belief() # sample
            s = random.choices(states, weights=probs, k=1)[0]

            next_state = self.step(s, action_index)
            obs_sample = self.get_observation(next_state)
            if obs_sample == obs:
                child.belief[next_state] = child.belief.get(next_state, 0) + 1
                child_belief_size += 1

    def update(self, actionIndex, obs):
        qnode = self.root.get_child_by_action_index(actionIndex)
        vnode = qnode.get_child_by_action_index(obs)
        self.invigorate_belief(self.root, vnode, actionIndex, obs)
        vnode.clear()
        self.root = vnode

    # def Update(self, actionIndex, obs):
    #     beliefs = defaultdict(int)
    #     qnode = self.root.get_child_by_action_index(actionIndex)
    #     isVnode = qnode.check_child_by_observation_index(obs)
    #     if isVnode:
    #         vnode = qnode.get_child_by_observation_index(obs)
    #         beliefs = vnode.belief.copy()
    #     if not beliefs and (not isVnode or not qnode.get_child_by_observation_index(obs).belief):
    #         return False
        
    #     if isVnode and qnode.get_child_by_observation_index(obs).belief:
    #         vnode = qnode.get_child_by_observation_index(obs)
    #         if vnode.belief:
    #             state = vnode.belief.keys()[0]
    #     else:
    #         state = beliefs.keys()[0]
        
    #     newRoot = POMCPNode()
    #     self.expand(newRoot, state)
    #     newRoot.belief = beliefs
    #     self.invigorate_belief(self.root, newRoot, actionIndex, obs)
    #     self.root = newRoot
    #     return True

    def get_default_action(self):
        return actions[0]

    def select_action(self):
        distableTrue = False
        if distableTrue: return -1
        self.UCT_search()
        actionIndex = self.greedyUCB(self.root, False)
        return actionIndex    

    def UCT_search(self):
        states, counts = zip(*self.root.belief.items())
        sum_count = sum(counts)
        probs = [cnt / sum_count for cnt in counts]
        for n in range(self.numSimulations):
            # state = self.root.sample_state_from_belief() # sample
            state = random.choices(states, weights=probs, k=1)[0]
            if self.verbose >= 2: print("====Start UCT search with sample state", state, "nums Search", n)
            self.TreeDepth = 0
            self.PeakTreeDepth = 0
            reward = self.simulateV(state, self.root)
            if (self.verbose >= 2):
                print("==MCTSMCT after num simulation", n)
        if self.verbose >= 1:
            print("finishing all simulations", self.numSimulations)

    def simulateV(self, state, vnode):
        self.PeakTreeDepth = self.TreeDepth
        if not vnode.children:
            self.expand(vnode, state)

        if (self.TreeDepth >= self.maxDepth): 
            return 0
        
        actionIndex = self.greedyUCB(vnode, True)
        if self.TreeDepth <= self.horizon and self.shiledLevel == ON_THE_FLY:
            if not vnode.have_state_in_belief_support(state): 
                print("checking", self.TreeDepth)
                vnode.add_particle(state)
                if not self.is_current_belief_winning(vnode, self.TreeDepth): #TODO
                    qparent = vnode.getParent() 
                    parentActionIndex = qparent.getH()
                    vparent = qparent.getParent() 
                    vparent.add_illegal_action_index(parentActionIndex)
                    print("pruning", vparent.belief.keys(), self.get_observation_from_beleif(vnode.belief), parentActionIndex)
                    for x in self.pomdp.winning_obs:
                        print(x)
        if self.TreeDepth >= 1:
            vnode.add_particle(state)

        qnode = vnode.get_child_by_action_index(actionIndex)
        total_reward = self.simulateQ(state, qnode, actionIndex)
        vnode.increaseV(total_reward)
        return total_reward
        
    
    def get_observation_from_beleif(self, belief):
        for state in belief:
            return self.get_observation(state)
        
    def simulateQ(self, state, qnode, actionIndex):
        delayed_reward = 0
        nextState = self.step(state, actionIndex)
        observation = self.get_observation(nextState)
        done = nextState in self.end_states
        immediate_reward = self.step_reward(state, actionIndex)
        total_reward = 0

        if self.verbose >= 3:
            print("uct action = ", self.pomdp.actions[actionIndex], "reward=", immediate_reward, "state", nextState)

        state = nextState
        vnode = None
        if qnode.check_child_by_observation_index(observation):
            vnode = qnode.get_child_by_observation_index(observation)
        
        para_expand_count = 1
        if (not vnode and (not done) and (qnode.n >= para_expand_count)):
            vnode = POMCPNode()
            vnode = self.expand_node(state)
            vnode.h = observation
            vnode.parent = qnode
            qnode.add_child(vnode, observation)

        if not done:
            self.TreeDepth += 1
            if vnode:
                delayed_reward += self.simulateV(state, vnode)
            else:
                delayed_reward += self.rollout(state)
            self.TreeDepth -= 1
        else:
            total_reward += self.get_state_reward(state)
        total_reward += immediate_reward + self.gamma * delayed_reward
        qnode.increaseV(total_reward)
        return total_reward
    
    def is_current_belief_winning(self, vnode, time):
        if vnode == self.root:
            for state in self.root.belief:
                obs = self.get_observation(state)
                if not self.is_winning((obs, time)):
                    return False
        obs = vnode.getH()
        return self.is_winning((obs, time))  


    def expand(self, parent, state):
        availableActions = self.get_legal_actions(state)
        for actionIndex in availableActions:
            newChild = POMCPNode()
            newChild.set_h_action(True)
            newChild.h = actionIndex
            newChild.parent = parent
            parent.add_child(newChild, actionIndex)
            #TODO
            # if self.shieldLevel == 2 and self.TreeDepth == 0 and parent.is_action_index_illegal(actionIndex):
            #     continue
            # if self.shieldLevel == 1 and self.TreeDepth == 0 and self.isActionShieldedForNode(parent, actionIndex):
            #     parent.add_illegal_action_index(actionIndex)
        
        if not parent.children:
            print("add default available actions")
            for actionIndex in availableActions:
                newChild = POMCPNode()
                newChild.set_h_action(True)
                newChild.h = actionIndex
                newChild.parent = parent
                parent.add_child(newChild, actionIndex)
    
    def is_action_index_shielded_for_node(self, parent, actionIndex): 
        if not self.pomdp.winning_obs:
            return False
        for state in parent.get_belief_support():
            for nxt_state in self.get_next_states(state, actionIndex):
                nxt_obs = self.get_observation(nxt_state)
                if not self.is_winning((nxt_obs, time+1)):
                    return True
        return False

    def get_next_states(self, state, actionIndex):
        if state not in self.pomdp.robot_state_action_map: return set()
        if actionIndex not in self.pomdp.robot_state_action_map[state]: return set()
        nxts =  self.pomdp.robot_state_action_map[state][actionIndex].keys()
        return nxts

    def is_winning(self, obs_time):
        return obs_time in self.pomdp.winning_obs

    def get_legal_actions(self, state):
        return set(self.pomdp.robot_state_action_map[state].keys())

    def expand_node(self, state):
        vnode = POMCPNode()
        vnode.belief[state] += 1
        available_actions = self.get_legal_actions(state)
        for actionIndex, action in enumerate(self.pomdp.actions):
            if action not in available_actions: continue
            qnode = POMCPNode()
            qnode.h = actionIndex
            qnode.set_h_action(True)
            qnode.parent = vnode
            vnode.add_child(qnode, actionIndex)
        return vnode
    
    def step_reward(self, state, actionIndex):
        if state not in self.pomdp.state_action_reward_map:
            return float("-inf")
        if actionIndex not in self.pomdp.state_action_reward_map[state]:
            return float("-inf")
        return self.pomdp.state_action_reward_map[state][actionIndex]
    
    def get_state_reward(self, state):
        return 0
    
    def get_random_action_index(self, state): # to be improved
        available_action_index = self.pomdp.robot_state_action_map[state].keys()
        return random.choice(list(available_action_index))
    
    def rollout(self, state):
        total_reward = 0
        discount = 1
        done = False
        if self.verbose >= 3: print("starting rollout")
        numStep = 0
        remainTree = self.maxDepth - self.TreeDepth

        while (not done and numStep < remainTree):
            actionIndex = self.get_random_action_index(state)
            next_state = self.step(state, actionIndex)
            reward = self.step_reward(state, actionIndex)
            done = next_state in self.end_states
            if self.verbose >= 4:
                print("state", state, "action", self.actions[actionIndex], "reward", reward, "depth", numStep, "totalR", total_reward)
            total_reward += reward * discount
            discount *= self.gamma
            numStep += 1
            state = next_state
        """
        if done: print("Done")
        """
        total_reward += self.get_state_reward(state) * discount 
        return total_reward

    def greedyUCB(self, vnode, ucb):
        besta = []
        bestq = float("-inf")
        N = vnode.n
        logN = math.log(N + 1)
        children = vnode.children
        action_index_candidates = []
        for i in children:
            action_index_candidates.append(i)
            # 			if (shieldLevel == ON_THE_FLY_SHIELD && vnode.isActionIndexIllegal(i)) {
            # //				System.out.println("shield level" + shieldLevel + " known illegal action " + allActions.get(i) +" for node " + vnode.getID() + " belief support" 	+ vnode.getBelief().getUniqueStatesInt());
            # 				continue;
            # 			}
            qnode = children[i]
            # print("-------",i, qnode)
            n = qnode.n
            if n == 0: return i
            q = qnode.getV()
            if ucb:
                q += self.fastUCB(N, n, logN)
            if q >= bestq:
                if q > bestq:
                    besta = []
                bestq = q
                besta.append(i)
#                   #//			if ( !ucb  && shieldLevel == 1  && isActionShieldedForNode(vnode, action) ) { // shiled only apply to the most up level
                # ////				System.out.println("shield Level = "+shieldLevel+ " Shielded Action = "  + action);
                # //				continue;
                # //			}
                # //			if (shieldLevel == 3 && vnode.isActionIllegal(action)) {
                # ////				System.out.println("shield level" + shieldLevel + " known illegal action" 
                # ////									+ action +" for node " + vnode.getID() + " belief support" 
                # ////									+ vnode.getBelief().getUniqueStatesInt());
                # //				continue;
                # //			}
                # //			if (shieldLevel == 3 && isActionShieldedForNode(vnode, action)) {
                # ////				System.out.println("shield level" + shieldLevel +" shielded action: " + action 
                # ////									+ "\n adding to illegal actions for node " + vnode.getID() 
                # ////									+ " belief support" +  vnode.getBelief().getUniqueStatesInt());
                # //				vnode.addIllegalActions(action);
                # //				continue;
                # //			}
        if besta:
            return random.choice(besta)
        else:
            actionIndex = random.choice(action_index_candidates)
            qParent = vnode.parent
            vParent = qParent.parent
            vParent.add_illegal_action_index(qParent.h)
            return actionIndex

    def get_action_index(self, action):
        if not self.action2Index:
            self.action2Index = {}
            for i, action in enumerate(self.actions):
                self.action2Index[action] = i
        return self.action2Index.get(action, -1)

if __name__ == "__main__":
    U = actions = ['N', 'S', 'E', 'W', 'ST']
    C = cost = [3300, 3, 3, 3, 1]

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

    obstacles =  [(5, 1), (7, 3), (17, 7), (7, 9)]
    target = [(19, 19)]
    end_states = set([(19,1)])

    robot_nodes = set()
    for i in range(1, 20, 2):
        for j in range(1, 20, 2):
            node = (i, j)
            robot_nodes.add(node) 

    initial_belief_support = [(5,5), (5,7), (7,5), (7,7)]
    initial_belief = {}
    for state in initial_belief_support:
        initial_belief[state] = 1 / len(initial_belief_support)

    pomdp = Model(robot_nodes, actions, cost, WS_transition, transition_prob,
                     initial_belief, obstacles, target, end_states)

    pomcp = POMCP(pomdp)

    motion_mdp, AccStates = pomcp.pomdp.compute_accepting_states() 
    H = 5
    observation_successor_map = pomcp.pomdp.compute_H_step_space(H)
    step = 0
    discounted_reward = 0
    undiscounted_redward = 0
    num_episodes = 1
    max_steps = 3

    # initial_delta = 0.95
    # acp_gamma = 0.95
    # delta = [[0] * max_steps for _ in range(H)]
    # e = [[0] * max_steps for _ in range(H)]
    # for tau in range(1, H+1):
    #     delta[tau][0] = initial_delta
            # // delta[tau][0] = initial_delta

    for _ in range(num_episodes):
        pomcp.reset_root()
        state_ground_truth = pomcp.root.sample_state_from_belief()
        state_ground_truth = (7,7)
        print(state_ground_truth, "current state")
        obs_current_node = pomcp.get_observation(state_ground_truth)
        while step < max_steps:

            # Y = [1] #observe()
            # ACP_step = [] # comformal_prediction() #TODO
            # for tau in range(1, H+1):
            #     delta[tau][step + 1] = delta[tau][step] + acp_gamma * (initial_delta * e[tau][step])
            #     R[tau][step] =  dist(Y, estimated_Y[tau][step-tau])
            #     q = ceil((step+1) * (1 - delta[tau][step+1]))
            #     C[tau][step+1] = sorted([R[tau][k] for k in range(tau, step+1)])[q]
            # ACP


            # self.compute_winning_region() # is winning region indepent of current belief ? @pian
            obs_mdp, Winning_observation = pomcp.pomdp.online_compute_winning_region(obs_current_node, AccStates, observation_successor_map, H, ACP_step)
            actionIndex = pomcp.select_action()
            next_state_ground_truth = pomcp.step(state_ground_truth, actionIndex)
            reward = pomcp.step_reward(state_ground_truth, actionIndex)
            obs_current_node = pomcp.get_observation(next_state_ground_truth)
            print("===================step", step, "s", "action", actions[actionIndex], state_ground_truth, "s'", next_state_ground_truth, "observation", obs_current_node)
            pomcp.update(actionIndex, obs_current_node)
            state_ground_truth = next_state_ground_truth
            step += 1
            discounted_reward += pomcp.gamma * reward
            undiscounted_redward += reward
