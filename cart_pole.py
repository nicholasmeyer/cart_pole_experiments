import gym
from collections import deque
import numpy as np
import math
import sys

import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')
env.seed(0)
np.random.seed(0)

print('observation space:', env.observation_space)
print('action space:', env.action_space)


class Policy():

    def __init__(self, s_size=4, a_size=2):
        # weights for simple linear policy: state_space x action_space
        self.w = 1e-4 * np.random.rand(s_size, a_size)

    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x) / sum(np.exp(x))  # softmax activation

    def act(self, state):
        probs = self.forward(state)
        # action = np.random.choice(2, p=probs) # option 1: stochastic policy
        action = np.argmax(probs)              # option 2: deterministic policy
        return action


def hill_climbing(policy, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2):
    """Implementation of hill climbing with adaptive noise scaling.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        noise_scale (float): standard deviation of additive noise
    """
    scores_deque = deque(maxlen=100)
    scores = []
    best_R = -np.Inf
    best_w = policy.w
    for i_episode in range(1, n_episodes + 1):
        rewards = []
        state = env.reset()
        for t in range(max_t):
            action = policy.act(state)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        if R >= best_R:  # found better weights
            best_R = R
            best_w = policy.w
            noise_scale = max(1e-3, noise_scale / 2)
            policy.w += noise_scale * np.random.rand(*policy.w.shape)
        else:  # did not find better weights
            noise_scale = min(2, noise_scale * 2)
            policy.w = best_w + noise_scale * np.random.rand(*policy.w.shape)

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode - 100, np.mean(scores_deque)))
            policy.w = best_w
            break

    return scores


def steepest_ascent(policy, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2, neighbour_size=10):
    """Implementation of steepest ascent adaptive noise scaling.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        noise_scale (float): standard deviation of additive noise
    """
    scores_deque = deque(maxlen=100)
    scores = []
    best_R = -np.Inf
    best_w = policy.w
    possible_w = [policy.w for _ in range(neighbour_size)]
    for i_episode in range(1, n_episodes + 1):
        rewards = []
        idx = 0
        for p, w in enumerate(possible_w):
            current_rewards = []
            state = env.reset()
            policy.w = w
            for t in range(max_t):
                action = policy.act(state)
                state, reward, done, _ = env.step(action)
                current_rewards.append(reward)
                if done:
                    break
            if sum(current_rewards) > sum(rewards):
                rewards = current_rewards
                idx = p

        policy.w = possible_w[idx]
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        if R >= best_R:  # found better weights
            best_R = R
            best_w = policy.w
            noise_scale = max(1e-3, noise_scale / 2)
            possible_w = [policy.w + noise_scale *
                          np.random.rand(*policy.w.shape) for _ in range(neighbour_size)]
        else:  # did not find better weights
            noise_scale = min(2, noise_scale * 2)
            possible_w = [best_w + noise_scale *
                          np.random.rand(*policy.w.shape) for _ in range(neighbour_size)]

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode - 100, np.mean(scores_deque)))
            policy.w = best_w
            break

    return scores, policy


def display(policy):
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    env.render()
    while True:
        action = policy.act(state)
        img.set_data(env.render(mode='rgb_array'))
        plt.axis('off')
        state, reward, done, _ = env.step(action)
        if done:
            break
    env.close()

if __name__ == "__main__":
    policy = Policy()
    scores = hill_climbing(policy)
    display(policy)
