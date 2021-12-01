from Game.environment import *
import random
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt


class SARSA_Agent():
    def __init__(self, actions, epsilon=0.0, discount=1, alpha=0.9):
        self.actions = actions
        self.game = Game()
        self.Q = defaultdict(float)
        self.first_epsilon = epsilon
        self.gamma = discount
        self.alpha = alpha

    def select_action(self, state):
        """implement epsilon greedy to pick next action"""
        #Choose an action randomly if less than epsilon - exploration
        if random.random() < self.epsilon:
            return np.random.choice(self.game.action_space.n)

        qList = [self.Q.get((state, action), 0) for action in self.actions]

        #choose best option - exploitation
        if qList[0] < qList[1]:
            return 1
        elif qList[0] > qList[1]:
            return 0
        else:
            return np.random.choice(self.game.action_space.n)

    def update_Q(self, state, action, reward, next_state, next_action):
        """Based on Sarsa policy, update the Q values"""

        self.Q[(state, action)] = (1 - self.alpha) * \
            self.Q.get((state, action), 0) + self.alpha * \
            (reward + self.gamma * self.Q.get((next_state, next_action), 0))

    def train(self, num_iters, n_iters_eval):
        """ Train the agent"""

        done = False
        max_score = 0
        max_reward = 0
        self.game.seed(random.randint(0, 100))
        test_scores = []

        for i in range(num_iters):

            self.epsilon = self.first_epsilon
            curr_score = 0
            total_reward = 0
            ob = self.game.reset()
            sarsa_policy = []
            #initialize game state
            state = self.game.getGameState()
            # get the next best action using epsilon greedy approach
            action = self.select_action(state)

            while True:
                # take the action and find the next step
                next_state, reward, done, _ = self.game.step(action)
                next_action = self.select_action(next_state)
                #add state action pairs to list
                sarsa_policy.append((state, action, reward, next_state,
                                   next_action))

                #update the states and actions                  
                state = next_state
                action = next_action

                #update reward
                total_reward += reward
                if reward >= 1:
                    curr_score += 1
                if done:
                    break

            if curr_score > max_score:
                max_score = curr_score
            if total_reward > max_reward:
                max_reward = total_reward

            for (state, action, reward, next_state, next_action) in sarsa_policy:
                self.update_Q(state, action, reward, next_state, next_action)

            #print current iteration every 250 iterations
            if i % 250 == 0:
                print("Iter: ", i)

            # after every 500 iterations, evaluate the model
            if (i + 1) % 500 == 0:
                max_score = self.evaluate(n_iter=n_iters_eval)
                test_scores.append(max_score)

        #write scores to a csv file so they can be plotted later
        save_score = pd.DataFrame(test_scores, columns=['scores'])
        save_score.to_csv("sarsa.csv")
        self.game.close()

    def evaluate(self, n_iter):
        """evaluate the agent"""

        self.epsilon = 0
        self.game.seed(0)

        done = False
        max_score = 0
        max_reward = 0
        output = defaultdict(int)

        for i in range(n_iter):
            curr_score = 0
            total_reward = 0
            ob = self.game.reset()
            state = self.game.getGameState()

            while True:
                action = self.select_action(state)
                state, reward, done, _ = self.game.step(action)
                total_reward += reward
                if reward >= 1:
                    curr_score += 1
                if done:
                    break

            #keep track of maximum score
            output[curr_score] += 1
            if curr_score > max_score:
                max_score = curr_score
            if total_reward > max_reward:
                max_reward = total_reward

        self.game.close()
        print("Max Score on Evaluation: ", max_score)

        return max_score

if __name__ == "__main__":
    agent = SARSA_Agent(actions = [0,1])
    agent.train(75000, 100)

    #read in scores from csv file 
    sarsa_scores = pd.read_csv("/Users/renee/Downloads/FlappyBird_using_RL-master/Scores/sarsa.csv", index_col=0)

    i = list(range(0, 75001, 500))[1:]
    sarsa_scores.index = i
    
    #create sarsa plot
    plt.plot(sarsa_scores['scores'][:80])
    plt.legend(['SARSA'])

    plt.show()


