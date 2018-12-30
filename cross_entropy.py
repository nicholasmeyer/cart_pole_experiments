import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *

env = create_env()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def play(agent):
    # load the weights from file
    agent.load_state_dict(torch.load('checkpoint.pth'))
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    while True:
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            action = agent(state).numpy().argmax()
        img.set_data(env.render(mode='rgb_array'))
        plt.axis('off')
        next_state, reward, done, _ = env.step(action)
        state = next_state
        if done:
            break
    env.close()


class Agent(nn.Module):
    def __init__(self, env, h_size=16):
        super(Agent, self).__init__()
        self.env = env
        # state, hidden layer, action sizes
        self.s_size = env.observation_space.shape[0]
        self.h_size = h_size
        self.a_size = 2
        # define layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)

    def set_weights(self, weights):
        s_size = self.s_size
        h_size = self.h_size
        a_size = self.a_size
        # separate the weights for each layer
        fc1_end = (s_size * h_size) + h_size
        fc1_W = torch.from_numpy(weights[:s_size * h_size].reshape(s_size, h_size))
        fc1_b = torch.from_numpy(weights[s_size * h_size:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end + (h_size * a_size)].reshape(h_size, a_size))
        fc2_b = torch.from_numpy(weights[fc1_end + (h_size * a_size):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def get_weights_dim(self):
        return (self.s_size + 1) * self.h_size + (self.h_size + 1) * self.a_size

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x.cpu().data

    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(device)
            action = self.forward(state).numpy().argmax()
            state, reward, done, _ = self.env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return


def cem(n_iterations=500, max_t=1000, gamma=1.0, print_every=10, pop_size=50, elite_frac=0.2, sigma=1e-2):
    """PyTorch implementation of a cross-entropy method.

    Params
    ======
        n_iterations (int): maximum number of training iterations
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        pop_size (int): size of population at each iteration
        elite_frac (float): percentage of top performers to use in update
        sigma (float): standard deviation of additive noise
    """
    agent = Agent(env).to(device)
    n_elite = int(pop_size * elite_frac)
    scores_deque = deque(maxlen=100)
    scores = []
    best_weight = sigma * np.random.randn(agent.get_weights_dim())
    weights_pop = [best_weight + (sigma * np.random.randn(agent.get_weights_dim())) for _ in range(pop_size)]
    best_reward = -np.Inf
    for i_iteration in range(1, n_iterations + 1):
        rewards = np.array([agent.evaluate(weights, gamma, max_t) for weights in weights_pop])
        elite_idxs = rewards.argsort()[-n_elite:]
        elite_weights = [weights_pop[i] for i in elite_idxs]
        current_weight = np.array(elite_weights).mean(axis=0)
        reward = agent.evaluate(current_weight, gamma=1.0)
        scores_deque.append(reward)
        scores.append(reward)

        if reward >= best_reward:  # found better weights
            best_reward = reward
            best_weight = current_weight
            sigma = max(1e-3, sigma / 2)
            # adaptively adjust noise scale
            weights_pop = [best_weight + (sigma * np.random.randn(agent.get_weights_dim())) for _ in range(pop_size)]
        else:  # did not find better weights
            sigma = min(2, sigma * 2)
            weights_pop = [best_weight + (sigma * np.random.randn(agent.get_weights_dim())) for _ in range(pop_size)]

        torch.save(agent.state_dict(), 'checkpoint.pth')

        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_iteration, np.mean(scores_deque)))

        if np.mean(scores_deque) >= 195.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score: {:.2f}'.format(i_iteration,
                                                                                           np.mean(scores_deque)))
            break

    plot(scores, 'cross entropy with adaptive noise scaling', 'cross_entropy')
    play(agent)


if __name__ == "__main__":
    cem()
