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

class POMCPNode:
    def __init__(self) -> None:
        self.belief = defaultdict(int)
        self.clear()

    def clear(self):
        self.parent = None
        self.children = {}
        self.h = -1
        self.isQNode = False
        self.v = 0
        self.n = 0
        self.illegalActionIndexes = set()
    
    def add_illegal_action_index(self, actionIndex):
        self.remove_child(actionIndex)
        self.illegalActionIndexes.add(actionIndex)
    
    def is_action_index_illegal(self, actionIndex):
        return actionIndex in self.illegalActionIndexes

    def set_h_action(self, isAction):
        self.isQNode = isAction

    def getV(self):
        return self.v / self.n
    
    def increaseV(self, value):
        self.v += value
        self.n += 1

    def add_child(self, node, index):
        self.children[node] = index
    
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
        states, counts = zip(*self.belief.items())
        total_counts = sum(cnt for cnt in counts)
        probs = (cnt / total_counts for cnt in counts)
        return random.choices(states, weights=probs, k=1)[0]


class POMCP:
    def __init__(self, pomdp, constant = 1000, maxDepth = 100, end_states = set(), target = set()):

        # def __init__(self, initial_belief, actions, robot_state_action_map, state_to_observation, state_action_reward_map, 
        #              end_states, constant = 1000, maxDepth = 100, targets = set()):
        #e (float): Threshold value below which the expected sum of discounted rewards for the POMDP is considered 0. Default value is 0.005.
        # c (float): Parameter that controls the importance of exploration in the UCB heuristic. Default value is 1.
        # no_particles (int): Controls the maximum number of particles that will be kept at each node 
        #                       and the number of particles that will be sampled from the posterior belief after an action is taken.
        self.numSimulations = 2 ** 15
        self.gamma = 0.95
        self.e = 0.05
        self.noParticles = 1200
        self.K = 10000
        self.TreeDepth = 0
        self.PeakTreeDepth = 0
        self.c = constant
        self.maxDepth = maxDepth

        # private double[] initialBeliefDistribution;
        # private double [][] UCB;
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
        self.shiledLevel = 0

        # self.initial_belief = self.pomdp.initial_belief
        # self.actions = self.pomdp.actions
        # self.mdpRewards = self.pomdp.state_action_reward_map
        # self.rewardFunction = self.pomdp.state_action_reward_map

        # self.robot_state_action_map = self.pomdp.robot_state_action_map
        # self.state_action_reward_map = self.pomdp.state_action_reward_map
        # self.state_to_observation = self.pomdp.state_to_observation

    def initializePOMCP(self):
        self.TreeDepth = 0
        self.PeakTreeDepth = 0
        if not self.UCB: self.initialUCB(1000, 100)
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
        while child_belief_size < self.K:
            # s = self.draw_from_probabilities(parent.belief)
            s = parent.sample_state_from_belief()
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
        for n in range(self.numSimulations):
            state = self.root.sample_state_from_belief()
            if self.verbose >= 2: print("====Start UCT search with sample state", state, "nums Search", n)
            self.TreeDepth = 0
            self.PeakTreeDepth = 0
            reward = self.simulateV(state, self.root)
            if (self.verbose >= 2):
                print("==MCTSMCT after num simulation", n)
        if self.verbose >= 1:
            print("finishing all simulations" + self.numSimulations)

    def simulateV(self, state, vnode):
        self.PeakTreeDepth = self.TreeDepth
        if not vnode.children:
            self.expand(vnode, state)
        if (self.TreeDepth >= self.maxDepth): return 0

        # 		// TODO check later for shielding logic
        # 		if (TreeDepth == 1 && shieldLevel != ON_THE_FLY_SHIELD) {
        # 			vnode.getBelief().addParticle(state); //add sample for only first layer
        # 			vnode.getBelief().updateBeliefSupport(state);
        # 		}
        actionIndex = self.greedUCB(vnode, True)
        # 		if (shieldLevel == ON_THE_FLY_SHIELD) {
        # 			vnode.getBelief().addParticle(state); //add sample for every layer
        # 			if (!vnode.getBelief().isStateInBeliefSupport(state)) { // only check when a new unique particle is to be added
        # 				vnode.getBelief().updateBeliefSupport(state);
        # 				if (!isSetOfStatesWinning(vnode.getBelief().getUniqueStatesInt())) {
        # 	//				System.out.println("\n" + vnode.getBelief().getUniqueStatesInt() + "is not winning. TreeDepth = " + TreeDepth);
        # 					POMCPNode qparent = vnode.getParent();
        # 					int parentActionIndex = qparent.getH();
        # 					POMCPNode vparent = qparent.getParent();
        # 					vparent.addIllegalActionIndex(parentActionIndex);
        # 					if (verbose >= 5) {
        # 						System.out.println("Currnet Node=" + vnode.getID() + " Current belief support" + vnode.getBelief().getUniqueStatesInt()  );
        # 						System.out.println("Currenting belief support is not winning. ");
        # 						System.out.println("shield level" + shieldLevel +" shielded action: " + allActions.get(parentActionIndex)
        # 								+ "\n adding to illegal actions for it parent node " + vparent.getID() 
        # 								+ " parent belief support" +  vparent.getBelief().getUniqueStatesInt());
        # 					}
        # 	//				System.out.println("after" + vparent.getIllegalActions());
        # 				} else {
        # 					if (verbose >= 5) {
        # 						System.out.println("safe. Current Belief Support" + vnode.getBelief().getUniqueStatesInt() + " vnode ID="+ vnode.getID() );
        # 					}
        # 				}
        # 			}		
        # 		}
        qnode = vnode.get_child_by_action_index(actionIndex)
        total_reward = self.simulateQ(state, qnode, actionIndex)
        vnode.increaseV(total_reward)
        return total_reward


    def simulateQ(self, state, qnode, actionIndex):
        delayed_reward = 0
        nextState = self.step(state, actionIndex)
        observation = self.get_observatoin(nextState)
        done = nextState in self.end_states
        immediate_reward = self.step_reward(state, actionIndex)
        total_reward = 0

        if self.verbose >= 3:
            print("uct action = ", self.actions[actionIndex], "reward=", immediate_reward, "state", nextState)

        state = nextState
        vnode = None
        if qnode.check_child_by_observation_index(observation):
            vnode = qnode.get_child_by_obseravation(observation)
        
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

    def get_legal_actions(self, state):
        return set(self.robot_state_action_map[state].keys())

    def expand_node(self, state):
        vnode = POMCPNode()
        vnode.belief[state] += 1
        available_actions = self.get_legal_actions(state)
        for actionIndex, action in enumerate(self.actions):
            if action not in available_actions: continue
            qnode = POMCPNode()
            qnode.h = actionIndex
            qnode.set_h_action(True)
            qnode.parent = vnode
            vnode.add_child(qnode, actionIndex)
        return vnode
    
    def step_reward(self, state, actionIndex):
        if state not in self.state_action_reward_map:
            return float("-inf")
        if actionIndex not in self.state_action_reward_map[state]:
            return float("-inf")
        return self.state_action_reward_map[state][actionIndex]
    
    def get_state_reward(self, state):
        return 0
    
    def get_random_actoin_index(self, state):
        available_action_index = self.robot_state_action_map[state].keys()
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
            n = qnode.n
            q = qnode.getV()
            if n == 0: return i
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
        
    # 	public void initializeStates() 
    # 	{
    # 		stateSuccessorsHashSet = new HashMap<Integer, HashSet<Integer>>();
    # 		stateSuccessorArrayList = new HashMap<Integer, ArrayList<Integer>> ();
    # 		stateSuccessorCumProb= new HashMap<Integer, ArrayList<Double>> ();
    # 		for (int state = 0; state < pomdp.getNumStates(); state++) {
    # 			for (int actionIndex = 0; actionIndex < allActions.size(); actionIndex++) {
    # 				int key = getKeyByStateActionIndex(state, actionIndex);
    # 				Object action = allActions.get(actionIndex);
    # 				if(!pomdp.getAvailableActions(state).contains(action)) {
    # 					continue;
    # 				}
    # 				int choice = pomdp.getChoiceByAction(state, action);
    # 				Iterator<Entry<Integer, Double>> iter = pomdp.getTransitionsIterator(state, choice);
    # 				ArrayList<Integer> nextStates = new ArrayList<Integer> ();
    # 				ArrayList<Double> nextStatesProbs = new ArrayList<Double> ();
    # 				ArrayList<Double> nextStatesCumProbs = new ArrayList<Double> ();
    # 				int count = 0;
    # 				while (iter.hasNext()) {
    # 					Map.Entry<Integer, Double> trans = iter.next();
    # 					nextStates.add(trans.getKey());
    # 					nextStatesProbs.add(trans.getValue());
    # 					if (count > 0) {
    # 						nextStatesCumProbs.add(nextStatesCumProbs.get(count - 1) + trans.getValue());
    # 					} else {
    # 						nextStatesCumProbs.add(trans.getValue());
    # 					}
    # 					count += 1;
    # 				}
    # 				stateSuccessorsHashSet.put(key, new HashSet<> (nextStates));
    # 				stateSuccessorArrayList.put(key, nextStates);
    # 				stateSuccessorCumProb.put(key, nextStatesCumProbs);
    # 				stepReward(state, actionIndex);
    # 			}
    # 		}
    # //		Collections.binarySearch(nextStatesProbs, 0.4);
    # 	}
        
    # 	public double stepReward(int state, int actionIndex) {
    # 		int key = getKeyByStateActionIndex(state, actionIndex);
    # 		double reward;
    # 		if (rewardFunction.containsKey(key)) {
    # 			reward = rewardFunction.get(key);
    # 		} else {
    # 			Object action = allActions.get(actionIndex);
    # 			int choice = pomdp.getChoiceByAction(state, action);		
    # 			if (!stateSuccessorArrayList.containsKey(key)) {
    # 				System.out.println("state = "+ state + allActions.get(actionIndex));
    # 			}
    # 			reward = mdpRewards.getTransitionReward(state, choice) + mdpRewards.getStateReward(state); // to check if no shield what is the cost function
    # 			if (min) {
    # 				reward *= -1;
    # 			}
    # 			rewardFunction.put(key,reward);
    # 		}
    # 		return reward;
    # 	}
        
        
    # 	public void initializeVariables() 
    # 	{
    # 		varNames = new ArrayList<String>();
    # 		for (int i = 0; i < pomdp.getVarList().getNumVars(); i++) {
    # 			varNames.add(pomdp.getVarList().getName(i));
    # 		}
    # 		variableIndexX = varNames.indexOf("ax") > 0 ? varNames.indexOf("ax") : varNames.indexOf("x");
    # 		variableIndexY = varNames.indexOf("ay") > 0 ? varNames.indexOf("ay") : varNames.indexOf("y");
    # 	}
        
    # 	public int getAX(int state) {
    # 		return (int) pomdp.getStatesList().get(state).getValueByIndex(variableIndexX);
    # 	}
    # 	public int getAY(int state) {
    # 		return (int) pomdp.getStatesList().get(state).getValueByIndex(variableIndexY);
    # 	}
        
    # 	public List<String> getVarNames()
    # 	{
    # 		return varNames;
    # 	}
    # 	public String getStateMeaning(int state) 
    # 	{	
    # 		return pomdp.getStatesList().get(state).toString(varNames);
    # 	}
    # 	public void displayState(int state) 
    # 	{
    # 		List<String> varNames = getVarNames();
    # 		System.out.println("s=" + state + pomdp.getStatesList().get(state).toString( varNames));
    # //		List<Object> availableActions = pomdp.getAvailableActions(state);
    # //		System.out.println("Available actions");
    # //		for (Object a: availableActions) {
    # //			System.out.print(a + " ");
    # //		}
    # //		System.out.println();
    # 	}
    # 	public int getStompyState(int PrismState) {
    # 		return mainShield.getStompyState(PrismState);
    # 	}
    # 	public HashSet<Integer> getStompyBeliefSupport(HashSet<Integer> PrismBeliefSupport){
    # 		HashSet<Integer> StompyBeliefSupport = new HashSet<Integer> ();
    # 		for (int PrismState : PrismBeliefSupport) {
    # 			StompyBeliefSupport.add(mainShield.getStompyState(PrismState));
    # 		}
    # 		return StompyBeliefSupport;
    # 	}
    # 	public HashSet<Integer> getRootBeliefSupportPrism() {
    # 		return root.getBelief().getUniqueStatesInt();
    # 	}
    # 	public HashSet<Integer> getRootBeliefSupportStompy(){
    # 		return getStompyBeliefSupport(getRootBeliefSupportPrism());
    # 	}
    # 	public HashSet<Object> getRootIllegaActions()
    # 	{
    # 		HashSet<Integer> illegalActionIndexes = root.getIllegalActionIndexes();
    # 		HashSet<Object> illegalActions = new HashSet<Object> ();
    # 		for (int index: illegalActionIndexes) {
    # 			illegalActions.add(allActions.get(index));
    # 		}
    # 		return illegalActions;
    # 	}
    # 	public void displayValue(int depth)
    # 	{
    # 		Queue<POMCPNode> queue = new LinkedList<POMCPNode>();
    # 		queue.offer(root);
    # 		int d = 0;
    # 		while(!queue.isEmpty()) {
    # 			if (d >= depth) {
    # 				System.out.println("reach tree print depth "+ depth);
    # 				break;
    # 			}
    # 			d++;
    # 			int size = queue.size();
    # //			if ( d % 2 == 0) {
    # 				System.out.println("MCTS layer" + d);
    # //			}
    # 			for (int i =0; i < size; i++) {
    # 				POMCPNode node = queue.poll();
    # 				displayNode(node, d);
    # 				displayNodeActions(node, d);
    # 				System.out.println("");
    # 				HashMap<Integer, POMCPNode> children = node.getChildren();
    # 				if (children != null) {
    # 					for(POMCPNode child : children.values()) {
    # 						if (child.getN() > 0) {
    # 							queue.offer(child);
    # 						}
    # 					}
    # 				}
    # 			}
    # 		}
    # 	}
    # 	public void displayNode(POMCPNode node, int depth)
    # 	{
    # 		String info = "";
    # 		if (!node.isQNode()){
    # 			info +="Id=" + node.getID()+ "depth" + depth  + " o=" + node.getH() + " vmean=" + (node.getV()) + " vall=" + (node.getV() * node.getN()) + " n=" + node.getN() +" Belief Support=" + node.getBelief().getUniqueStatesInt();
    # 		}
    # 		else {
    # 			info +="Id=" + node.getID() + "depth" + depth + " a=" + allActions.get(node.getH() ) + " vmean=" + (node.getV()) + " vall=" + (node.getV() * node.getN()) + " n=" + node.getN() ;
    # 		}
    # 		POMCPNode parent = node.getParent();
    # 		if (parent == null) {
    # 			System.out.println(info);
    # 		}
    # 		else {
    # 			System.out.print(info + " | ");			
    # 			displayNode(parent, depth - 1);
    # 		}
    # 	}
    # 	public void displayNodeActions(POMCPNode node, int depth)
    # 	{
    # 		String info = "";
    # 		if (!node.isQNode()){
    # //			info +="depth" + depth + " obs=" + node.getH()  ;
    # 		}
    # 		else {
    # 			info +="depth" + depth + " a=" + allActions.get(node.getH() ) ;
    # 		}
    # 		POMCPNode parent = node.getParent();
    # 		if (parent == null) {
    # 			System.out.println(info);
    # 		}
    # 		else {
    # 			System.out.print(info + " | ");			
    # 			displayNodeActions(parent, depth - 1);
    # 		}
    # 	}

    # 	public HashSet<Integer> getNextBeliefSupport(HashSet<Integer> beliefSupport, Object action)
    # 	{
    # 		HashSet<Integer> nextBeliefSupport = new HashSet<Integer> ();
    # 		for (int state: beliefSupport) { // for every state in current belief support
    # 			HashSet<Integer> nextStates = getNextStates(state, action);			// get its successor states
    # 			nextBeliefSupport.addAll(nextStates);			// add these states into next belief support
    # 		}
    # 		return nextBeliefSupport;
    # 	}
        
    # 	public HashSet<Integer> getNextStates(int state, Object action)
    # 	{
    # 		int actionIndex = actionToIndex.get(action);
    # 		int key = getKeyByStateActionIndex(state, actionIndex);
            
    # 		if (stateSuccessorsHashSet.get(key) == null) {
    # 			HashSet<Integer> nextStates = new HashSet<Integer>();
    # 			int choice = pomdp.getChoiceByAction(state, action);
    # 			Iterator<Entry<Integer, Double>> iter = pomdp.getTransitionsIterator(state, choice);
    # 			while (iter.hasNext()) {
    # 				Map.Entry<Integer, Double> trans = iter.next();
    # 				int nextState = trans.getKey();
    # 				nextStates.add(nextState);
    # 			}
    # 			stateSuccessorsHashSet.put(key, nextStates);
    # 		}
    # 		return stateSuccessorsHashSet.get(key);
    # 	}
        
    # 	public boolean isActionShieldedForNode(POMCPNode node, Object action) // Main interface checking if action should be shielded
    # 	{
    # 		HashSet<Integer> currentBeliefSupport = node.getBelief().getUniqueStatesInt();
    # 		return isActionShieldedForStates(currentBeliefSupport, action);
    # 	}
        
    # 	public boolean isActionShieldedForStates(HashSet<Integer> beliefSupport, Object action) 
    # 	{
    # 		if (useLocalShields) {
    # 			return isActionShieldedForStatesByLocalShileds(beliefSupport, action);
    # 		} else {
    # 			return isActionShieldedForStatesByMainShield(beliefSupport, action);
    # 		}
    # 	}
        
    # 	public int getStateIndex(int x, int y) 
    # 	{
    # 		return x + y * gridSize;
            
    # 	}
    # 	public int getShieldIndex(int state)
    # 	{
            
    # 		int x = getAX(state);
    # 		int y = getAY(state);
    # //		int shieldIndex = (x / shieldSize)* (gridSize / shieldSize) + (y / shieldSize) ;
    # 		return stateToLocalShieldIndex.get(getStateIndex(x, y));
    # //		System.out.println(state + " get " + getStateMeaning(state) + "x " + x + ", y= " + y + "sheild index" + shieldIndex);
    # //		return shieldIndex; 
    # 	}
    # 	public boolean isActionShieldedForStatesByLocalShileds(HashSet<Integer> beliefSupport, Object action) 
    # 	{
    # 		if (localShields == null) {
    # 			return false; // no local shields available
    # 		}
    # 		HashSet<Integer> nextBeliefSupport = getNextBeliefSupport(beliefSupport, action);
    # 		if (verbose > 0) {
    # 			System.out.println("considering if to shield action "  + action + " next suport" + nextBeliefSupport);
    # 		}
    # 		if (isSetOfStatesWinningByLocalShields(nextBeliefSupport)) {
    # 			return false;
    # 		} else {
    # 			return true;
    # 		}
    # 	}
  
    # 	public boolean isActionShieldedForStatesByMainShield(HashSet<Integer> beliefSupport, Object action) 
    # 	{
    # 		if (mainShield == null) {
    # 			return false; // no shield available
    # 		}
    # 		HashSet<Integer> nextBeliefSupport = getNextBeliefSupport(beliefSupport, action);
    # 		if (!mainShield.isSetOfStatesWinning(nextBeliefSupport)) {
    # //			System.out.print("action shield for " + beliefSupport);
    # 			return true; // action should be shielded because next belief support is not winning
    # 		}
    # 		return false;
    # 	}

    # 	public boolean isSetOfStatesWinning(HashSet<Integer> beliefSupport) {
    # 		if (!useLocalShields) {
    # 			return isSetOfStatesWinningByMainShield(beliefSupport);
    # 		}else {
    # 			return isSetOfStatesWinningByLocalShields(beliefSupport);
    # 		}
    # 	}
    # 	public boolean isSetOfStatesWinningByMainShield(HashSet<Integer> beliefSupport) {
    # 		if (mainShield == null) {
    # 			return true; // no shield available
    # 		}
    # 		return mainShield.isSetOfStatesWinning(beliefSupport);
    # 	}
    # 	public boolean isSetOfStatesWinningByLocalShields(HashSet<Integer> beliefSupport) {
    # 		if (localShields == null) {
    # 			return true; // no shield available
    # 		}
    # 		HashMap<Integer, HashSet<Integer>> shieldIndexToBeliefSupport = new HashMap<Integer, HashSet<Integer>> ();
    # 		for (int state: beliefSupport) {
    # 			int shieldIndex = getShieldIndex(state);
    # 			if (!shieldIndexToBeliefSupport.containsKey(shieldIndex)) {
    # 				HashSet<Integer> belief = new HashSet<Integer>();
    # 				shieldIndexToBeliefSupport.put(shieldIndex, belief);
    # 			}
    # 			shieldIndexToBeliefSupport.get(shieldIndex).add(state);
    # //			System.out.println("x" + getAX(state)+ "y " + getAY(state) + getStateIndex(getAX(state),getAY(state)) + " " + " " + shieldIndex );
    # 		}
    # 		for (int shieldIndex: shieldIndexToBeliefSupport.keySet()) {
    # 			POMDPShield localShield = localShields.get(shieldIndex);
    # 			HashSet<Integer> beliefSupportToCheck = shieldIndexToBeliefSupport.get(shieldIndex);
    # 			if(!localShield.isSetOfStatesWinning(beliefSupportToCheck)) {
    # 				return false;
    # 			}
    # 		}
    # 		return true;
    # 	}

    # 	public void loadMainShield(String shieldDir) {
    # 		File files = new File(shieldDir);
    # 		File[] array = files.listFiles();
    # 		for (int i = 0; i < array.length; i++) {
    # 			if (!array[i].isFile() ) {
    # 				continue;
    # 			}
    # 			File file = array[i];
    # 			String fileName = array[i].getName(); 
    # 			if (!fileName.contains("centralized")) {
    # 				continue;
    # 			}
    # 			if (fileName.contains("._")) {
    # 				continue;
    # 			}
    # 			System.out.println("++++Initialize main shield " );
    # 			System.out.println(fileName);
    # 			this.isMainShieldAvailable = true;
    # 			String[] parameters = fileName.split("-");
    # 			int n = parameters.length;
    # //			gridSize = Integer.parseInt(parameters[1]);
    # //			shieldSize = gridSize;
    # 			int[] pStates = {0, 0, Integer.parseInt(parameters[n-3]), Integer.parseInt(parameters[n-2])}; 
                
    # 			String winning = file.toString();
                
                
    # 			mainShield = new POMDPShield(pomdp, winning,  varNames, endStates, pStates);
    # 			break;
    # 		}
    # 	}
        
    # 	public void loadShiled() 
    # 	{
    # 		File files = new File(".");
    # 		File[] array = files.listFiles();
    # 		for (int i = 0; i < array.length; i++) {
    # 			String path = array[i].getPath();
    # 			if (path.contains("winningregion")){
    # 				loadMainShield(path);
    # 				loadLocalShield(path);
    # 				return;
    # 			}
    # 		}
    # 		System.out.println("Fail to find directory winningregion");
    # 	}
        
    # 	public void setUseLocalShields(boolean use) 
    # 	{
    # 		useLocalShields = use;
    # 	}
    # 	public void setShieldLevel(String level) 
    # 	{
    # 		shieldLevel = level;
    # 	}
    # 	public void loadLocalShield(String shieldDir) 
    # 	{
    # 		localShields = new ArrayList<POMDPShield> ();
    # 		stateToLocalShieldIndex = new HashMap<Integer, Integer> ();
    # 		gridSize = 0;
    # 		File files = new File(shieldDir);
    # 		File[] array = files.listFiles();
    # 		ArrayList<String> fileNames = new ArrayList<String> ();
    # 		for (int i = 0; i < array.length; i++) {
    # 			if (!array[i].isFile()) {
    # 				continue;
    # 			}
    # 			String fileName = array[i].getName(); 
    # 			if (!fileName.contains("factor")) {
    # 				continue;
    # 			}
    # 			if (fileName.contains("._")) {
    # 				continue;
    # 			}
    # 			fileNames.add(fileName);
    # 		}
    # 		Collections.sort(fileNames);
            
    # 		for (int i = 0; i < fileNames.size(); i++) {
    # 			String fileName = fileNames.get(i);
    # 			String[] parameters = fileName.split("-");
    # //			shieldSize = Integer.parseInt(parameters[1]);
    # 			int n= parameters.length;
    # 			gridSize = Math.max(gridSize, Integer.parseInt(parameters[n - 2]) - Integer.parseInt(parameters[n - 5]) + 1);
    # 		}
            
    # 		for (int i = 0; i < fileNames.size(); i++) {
    # 			String fileName = fileNames.get(i);
    # 			System.out.println("++++Initialize shield index = " + localShields.size() + ", shieldName = "+ fileName);
    # 			this.isLocalShieldAvailable = true;
    # 			String winning = shieldDir + System.getProperties().getProperty("file.separator") + fileName;
                
                
                
    # 			String[] parameters = fileName.split("-");
    # 			int n= parameters.length;
    # 			int[] pStates = {Integer.parseInt(parameters[n-5]), Integer.parseInt(parameters[n-4]), Integer.parseInt(parameters[n-3]), Integer.parseInt(parameters[n-2])}; 
                
    # //			System.out.println(Arrays.toString(pStates));
                
    # 			POMDPShield localShield = new POMDPShield(pomdp, winning, varNames, endStates, pStates);
    # 			for (int x = pStates[0]; x < pStates[2] + 1; x ++ ) {
    # 				for (int y = pStates[1]; y < pStates[3] + 1; y ++) {
    # 					int xy = getStateIndex(x, y);
    # 					int shieldIndex = localShields.size();
    # 					stateToLocalShieldIndex.put(xy, shieldIndex);
    # //					System.out.println(x + " " + y+ " " + localShields.size() + xy + " "  + Integer.parseInt(parameters[2])+ " " + Integer.parseInt(parameters[3])+ " " 
    # //							+ Integer.parseInt(parameters[4])+ " " + Integer.parseInt(parameters[5] ));
    # 				}
    # 			}
    # 			localShields.add(localShield);
    # 		}
    # 	}
    # 	public boolean hasMainShield() 
    # 	{
    # 		return this.isMainShieldAvailable;
    # 	}
    # 	public boolean hasLocalShield() 
    # 	{
    # 		return this.isLocalShieldAvailable;
    # 	}	
        
    # //	public int getDistanceToEndState(int state) {
    # //		for(int endState: endStates) {
    # //			int ax = getAX(state);
    # //			int ay = getAY(state);
    # //			int x = getAX(endState);
    # //			int y = getAY(endState);
    # //			return Math.abs(ax -x) + Math.abs(ay - y);
    # //		} 
    # //		return -1;
    # //	}
    # } 


if __name__ == "__main__":
    U = actions = ['N', 'S', 'E', 'W', 'ST']
    C = cost = [3, 3, 3, 3, 1]

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

    obstacles =  [(5, 1), (7, 3), (17, 7)]
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
    for _ in range(num_episodes):
        pomcp.reset_root()
        state_ground_truth = pomcp.root.sample_state_from_belief()
        print(state_ground_truth, "current state")
        obs_current_node = pomcp.get_observation(state_ground_truth)
        max_steps = 1000
        while step < max_steps:
            # environment_observation = observe()
            ACP_step = {} # comformal_prediction() #TODO
            # self.compute_winning_region() # is winning region indepent of current belief ? @pian
            obs_mdp, Winning_observation = pomcp.pomdp.online_compute_winning_region(obs_current_node, AccStates, observation_successor_map, H, ACP_step)
            actionIndex = pomcp.select_action()
            next_state_ground_truth = pomcp.step(state_ground_truth, actionIndex)
            reward = pomcp.step_reward(state_ground_truth, actionIndex)
            obs_current_node = pomcp.get_observation(next_state_ground_truth)
            pomcp.update(actionIndex, obs_current_node)
            state_ground_truth = next_state_ground_truth
            step += 1
            discounted_reward += pomcp.gamma * reward
            undiscounted_redward += reward
