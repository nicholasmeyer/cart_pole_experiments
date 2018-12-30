from collections import deque

from utils import *


def simulated_annealing(policy, n_episodes=2000, max_t=1000, gamma=1.0, print_every=100, temp=1000, temp_final=1):
    """Implementation of hill climbing with simulated annealing.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        gamma (float): discount rate
        print_every (int): how often to print average score (over last 100 episodes)
        noise_scale (float): standard deviation for gaussian noise
    """
    scores_deque = deque(maxlen=100)
    scores = []
    current_R = -np.Inf
    current_w = policy.w
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

        if R >= current_R:  # found better weights
            current_R = R
            current_w = policy.w
            policy.w += np.random.rand(*policy.w.shape)
        else:  # did not find better weights
            prob = np.exp((R - current_R) / temp)
            explore = np.random.choice([0, 1], p=[1-prob, prob])
            if explore:
                current_R = R
                current_w = policy.w
                policy.w += np.random.rand(*policy.w.shape)
            else:
                policy.w = current_w + np.random.rand(*policy.w.shape)
        # anneal temperature
        temp -= (temp - temp_final) / n_episodes
        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))
            policy.w = current_w
            break
    plot(scores, 'simulated annealing', 'simulated_annealing')
    play(policy)


if __name__ == "__main__":
    policy = Policy()
    simulated_annealing(policy)
