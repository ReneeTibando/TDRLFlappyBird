from Game.environment import *
import random
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

class Baseline_Agent():
    def __init__(self, actions):

        self.actions = actions
        self.game = Game()

    def select_action(self, state):
        """Select an action in a random manner"""
        if random.random() < 0.5:
            return 0
        return 1

    def train(self, n_iters, n_iters_eval):
        """ Train the agent"""

        done = False
        max_score = 0
        max_reward = 0
        self.game.seed(random.randint(0, 100))
        test_scores = []

        for i in range(n_iters):

            if i % 250 == 0:
                print("Iter: ", i)

            if (i + 1) % 500 == 0:
                max_score = self.evaluate(n_iter=n_iters_eval)
                test_scores.append(max_score)

        df = pd.DataFrame(test_scores, columns=['scores'])
        df.to_csv("baseline.csv")
        self.game.close()

    def evaluate(self, n_iter):
        """evaluates the agent"""

        self.epsilon = 0
        self.game.seed(0)

        done = False
        max_score = 0
        max_reward = 0
        output = defaultdict(int)

        for i in range(n_iter):
            score = 0
            total_reward = 0
            ob = self.game.reset()
            state = self.game.getGameState()

            while True:
                action = self.select_action(state)
                state, reward, done, _ = self.game.step(action)
                total_reward += reward
                if reward >= 1:
                    score += 1
                if done:
                    break

            output[score] += 1
            if score > max_score:
                max_score = score
            if total_reward > max_reward:
                max_reward = total_reward

        self.game.close()
        print("Max Score on Evaluation: ", max_score)

        return max_score

if __name__ == "__main__":
    agent = Baseline_Agent(actions=[0, 1])
    agent.train(50000, 100)

    baseline_scores = pd.read_csv("/Users/renee/Downloads/FlappyBird_using_RL-master/Scores/baseline.csv", index_col=0)

    i = list(range(0, 50001, 500))[1:]
    baseline_scores.index = i

    plt.plot(baseline_scores['scores'][:80])
    plt.legend(['Baseline'])

    plt.show()


    # this is the code that plots the graph against all three algorithms
    sarsa_scores = pd.read_csv("/Users/renee/Downloads/FlappyBird_using_RL-master/Scores/sarsa.csv", index_col=0)
    q_scores = pd.read_csv("/Users/renee/Downloads/FlappyBird_using_RL-master/Scores/qlearning.csv", index_col=0)

    i = list(range(0, 50001, 500))[1:]
    baseline_scores.index = i
    sarsa_scores.index = i
    q_scores.index = i[:80]

    plt.plot(baseline_scores['scores'][:80])
    plt.plot(sarsa_scores['scores'][:80])
    plt.plot(q_scores['scores'][:80])
    plt.legend(['Baseline', 'SARSA', 'Q-learning'])

    plt.show()


