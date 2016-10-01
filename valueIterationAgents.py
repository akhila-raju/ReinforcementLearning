# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def runValueIteration(self):
        # Write value iteration code here
        """
          Return max Q value.
        """
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        # V_0 of all states starts at 0. with each iteration, we want to update the values for each state.
        for i in range(self.iterations):
          values = util.Counter()
          for state in allStates:
            if self.mdp.isTerminal(state):
              values[state] = 0.0
            else:
              possibleActions = self.mdp.getPossibleActions(state)
              q_vals = []
              for action in possibleActions:
                q_vals.append(self.computeQValueFromValues(state, action))
              values[state] =  max(q_vals)
          self.values = values

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # Qk+1(s, a) = T(s, a, s') * (R(s, a, s') + Y*max Qk(s', a'))

        # list of (nextState, prob) pairs representing reachable states and transition probabilities
        nextStateList = self.mdp.getTransitionStatesAndProbs(state, action)
        q_sum = 0
        for nextState in nextStateList:
          transition = nextState[1]
          reward = self.mdp.getReward(state, action, nextState[0])
          q_sum += transition * (reward + self.discount * self.getValue(nextState[0]))
        return q_sum


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
          return None
        def getKey(item):
          return item[0]
        possibleActions = self.mdp.getPossibleActions(state)
        q_vals = util.Counter()
        for action in possibleActions:
          q_vals[action] = self.computeQValueFromValues(state, action)
        # return the action with the highest q-val
        return q_vals.argMax()


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        # V_0 of all states starts at 0. with each iteration, we want to update the values for each state.
        for i in range(self.iterations):
          state = allStates[i % len(allStates)]
          if self.mdp.isTerminal(state):
            self.values[state] = 0.0
          else:
            possibleActions = self.mdp.getPossibleActions(state)
            q_vals = []
            for action in possibleActions:
              q_vals.append(self.computeQValueFromValues(state, action))
            self.values[state] =  max(q_vals)



class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        # Compute predecessors of all states.
        " ADD CODE HERE "
        predecessors = {}
        # Initialize dictionary with empty set for predecessors.
        for state in allStates:
          predecessors[state] = set()
        for state in allStates:
          possibleActions = self.mdp.getPossibleActions(state)
          for action in possibleActions:
            nextStateList = self.mdp.getTransitionStatesAndProbs(state, action)
            for nextState in nextStateList:
              if nextState[1] != 0:
                predecessors[nextState[0]].add(state)

        # Initialize an empty priority queue.
        queue = util.PriorityQueue()
        for state in allStates:
          # For each non-terminal state s, do:
          if not self.mdp.isTerminal(state):
            possibleActions = self.mdp.getPossibleActions(state)
            q_vals = []
            for action in possibleActions:
              q_vals.append(self.computeQValueFromValues(state, action))
            highest_q_val =  max(q_vals)
            # Find the absolute value of the difference between the current value of s in self.values and the highest Q-value across all possible actions from s (this represents what the value should be); call this number diff. Do NOT update self.values[s] in this step.
            diff = abs(self.values[state] - highest_q_val)
            # Push s into the priority queue with priority -diff (note that this is negative). We use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
            queue.update(state, -diff)

        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        for i in range(self.iterations):
          # If the priority queue is empty, then terminate.
          if not queue.isEmpty():
            # Pop a state s off the priority queue.
            state = queue.pop()
            # Update s's value (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(state):
              possibleActions = self.mdp.getPossibleActions(state)
              q_vals = []
              for action in possibleActions:
                q_vals.append(self.computeQValueFromValues(state, action))
              highest_q_val =  max(q_vals)
              self.values[state] = highest_q_val

              state_predecessors = predecessors[state]
              # For each predecessor p of s, do:
              for predecessor in state_predecessors:
                # Find the absolute value of the difference between the current value of p in self.values and the highest Q-value across all possible actions from p (this represents what the value should be); call this number diff. Do NOT update self.values[p] in this step.
                possibleActions = self.mdp.getPossibleActions(predecessor)
                q_vals = []
                for action in possibleActions:
                  q_vals.append(self.computeQValueFromValues(predecessor, action))
                highest_q_val =  max(q_vals)

                diff = abs(self.values[predecessor] - highest_q_val)
                # If diff > theta, push p into the priority queue with priority -diff (note that this is negative), as long as it does not already exist in the priority queue with equal or lower priority. As before, we use a negative because the priority queue is a min heap, but we want to prioritize updating states that have a higher error.
                if diff > self.theta:
                  queue.update(predecessor, -diff)


