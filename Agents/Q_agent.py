from Game.environment import *
import random
import numpy as np
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt

class Q_Agent():
    def __init__(self, actions, epsilon=0.0, discount=1, alpha=0.9):
        self.actions = actions
        self.game = Game()
        self.Q = defaultdict(float)
        self.first_epsilon = epsilon
        self.gamma = discount
        self.learning_rate = alpha

    def select_action(self, state):
        """Select action corresponding to maximum reward"""

        if random.random() < self.epsilon:
            return np.random.choice(self.game.action_space.n)

        # Get rewards for all possible actions in the current state
        qList = [self.Q.get((state, action), 0) for action in self.actions]

        # Select action yielding maximum reward
        if qList[0] < qList[1]:
            return 1
        elif qList[0] > qList[1]:
            return 0
        else:
            return np.random.choice(self.game.action_space.n)

    def update_Q(self, state, action, reward, next_state):
        """Based on the Q-learning algorithm, update the Q-value"""

        next_Q_values = [self.Q.get((next_state, a), 0) for a in self.actions]
        # Select maximum value
        max_value = max(next_Q_values)
        # Update Q-value
        self.Q[(state, action)] = (1 - self.learning_rate) * \
            self.Q.get((state, action), 0) + self.learning_rate * \
            (reward + self.gamma * max_value)

    def train(self, num_iter, num_iter_eval):
        """ Train the Q-learning agent"""

        complete = False
        maximum_score = 0
        maximum_reward = 0
        self.game.seed(random.randint(0, 100))
        scores_test = []

        for i in range(num_iter):

            # initialize score, total reward sum, current game state, etcetera
            self.epsilon = self.first_epsilon
            curr_score = 0
            reward_sum = 0
            ob = self.game.reset()
            q_learning_list = []
            state = self.game.getGameState()

            while True:
                # identify best future action based on achieving maximum reward
                future_action = self.select_action(state)
                next_state, reward, complete, _ = self.game.step(future_action)
                q_learning_list.append((state, future_action, reward, next_state))
                state = next_state

                reward_sum += reward
                if reward >= 1:
                    curr_score += 1
                if complete:
                    break

            # update maximum score and reward
            if curr_score > maximum_score:
                maximum_score = curr_score
            if reward_sum > maximum_reward:
                maximum_reward = reward_sum

            # update Q-values
            for (state, future_action, reward, next_state) in q_learning_list[::-1]:
                self.update_Q(state, future_action, reward, next_state)

            # every 250 iterations, display iteration number
            if i % 250 == 0:
                print("Iter: ", i)

            # every 500 iterations, evaluate the q-learning model
            if (i + 1) % 500 == 0:
                maximum_score = self.evaluate(n_iter=num_iter_eval)
                scores_test.append(maximum_score)

        # save q-learning scores to relevant csv file for future use
        save_score = pd.DataFrame(scores_test, columns=['curr_scores'])
        save_score.to_csv("qlearning.csv")
        # close the game
        self.game.close()

    def evaluate(self, n_iter):
        """evaluate the q-learning agent"""

        self.epsilon = 0
        self.game.seed(0)

        complete = False
        maximum_score = 0
        maximum_reward = 0
        output = defaultdict(int)

        for i in range(n_iter):
            # initialize current score, total reward sum, etcetera.
            curr_score = 0
            reward_sum = 0
            ob = self.game.reset()
            curr_state = self.game.getGameState()

            while True:
                action = self.select_action(curr_state)
                curr_state, reward, complete, _ = self.game.step(action)
                reward_sum += reward
                if reward >= 1:
                    curr_score += 1
                if complete:
                    break

            # update maximum score and reward
            output[curr_score] += 1
            if curr_score > maximum_score:
                maximum_score = curr_score
            if reward_sum > maximum_reward:
                maximum_reward = reward_sum

        # close the game
        self.game.close()
        # display the maximum score
        print("Max score on Evaluation: ", maximum_score)

        return maximum_score

if __name__ == "__main__":
    # initialize and train the q-learning agent
    agent = Q_Agent(actions=[0, 1])
    agent.train(50000, 100)

    # read in q-learning scores from relevant csv file
    q_curr_scores = pd.read_csv("/Users/renee/Downloads/FlappyBird_using_RL-master/curr_scores/qlearning.csv", index_col=0)

    i = list(range(0, 50001, 500))[1:]
    q_curr_scores.index = i[:80]

    # plot the q-learning scores
    plt.plot(q_curr_scores['curr_scores'][:80])
    plt.legend(['Q-Learning'])

    # display the created plot
    plt.show()
