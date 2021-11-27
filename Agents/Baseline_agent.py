# import all the required libraries and the game environment that is in the game folder
from Game.environment import *
import random
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

class Baseline_Agent():
    """this class includes the function that allow for the baseline agent to make decisions and play """
    def __init__(self, actions):
        # initialize the actions and the game space
        self.actions = actions
        self.game = Game()

    def select_action(self, state):
        """the baseline agent chooses/selects a random action"""
        if random.random() < 0.5:
            return 0
        return 1

    def train(self, iterations, iteration_eval):
        """
        Train the baseline agent
        :param iterations: the number of times the agent runs through flappy bird to learn
        :param iteration_eval: after this number of iterations evaluate the agents learning
        """
        done = False
        max_score = 0
        max_reward = 0
        self.game.seed(random.randint(0, 100))
        # store the results
        test_scores = []

        for i in range(iterations):
            # print number of iterations
            if i % 250 == 0:
                print("Iter: ", i)
            # evaluate how much flappy has learned after i iterations
            if (i + 1) % 500 == 0:
                max_score = self.evaluate(iterations=iteration_eval)
                test_scores.append(max_score)

        # save the results to an excel file
        df = pd.DataFrame(test_scores, columns=['scores'])
        df.to_csv("baseline.csv")
        self.game.close()

    def evaluate(self, iterations):
        """
        This function evaluates flappy bird learning
        :param iterations: the number of times the agent runs through flappy bird to learn
        :return: max_score of flappy bird
        """

        self.epsilon = 0
        self.game.seed(0)

        # initialize the max score and reward
        done = False
        max_score = 0
        max_reward = 0
        # create a container to store the output
        output = defaultdict(int)

        for i in range(iterations):
            score = 0
            total_reward = 0
            # reset the state space
            ob = self.game.reset()
            state = self.game.getGameState()

            while True:
                action = self.select_action(state)
                # calculate the state, reward and if we finished
                state, reward, done, _ = self.game.step(action)
                total_reward += reward
                # if reward is greater than 1 we add the the score
                if reward >= 1:
                    score += 1
                # if done is true, break
                if done:
                    break
            # increment the container of our output
            output[score] += 1
            if score > max_score:
                max_score = score
            if total_reward > max_reward:
                max_reward = total_reward

        self.game.close()
        print("Max Score on Evaluation: ", max_score)

        return max_score

if __name__ == "__main__":
    # this is the training and output for the baseline agent
    agent = Baseline_Agent(actions=[0, 1])
    agent.train(50000, 100)

    # read the excel file that has our training results
    baseline_scores = pd.read_csv("/Users/renee/Downloads/FlappyBird_using_RL-master/Scores/baseline.csv", index_col=0)

    i = list(range(0, 50001, 500))[1:]
    baseline_scores.index = i

    # plot the baseline agent results
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

    # plot the results for baseline, SARSA, and Q-learning
    plt.plot(baseline_scores['scores'][:80])
    plt.plot(sarsa_scores['scores'][:80])
    plt.plot(q_scores['scores'][:80])
    plt.legend(['Baseline', 'SARSA', 'Q-learning'])

    plt.show()


