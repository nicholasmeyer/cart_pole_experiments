from collections import deque

from utils import *

env = create_env()


def steepest_ascent(policy, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100, noise_scale=1e-2, neighbour_size=5):
    """Implementation of steepest ascent adaptive noise scaling.

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
    best_R = -np.Inf
    best_w = policy.w
    possible_w = [policy.w]
    for i_episode in range(1, n_episodes + 1):
        rewards = []
        tmp_best_R = -np.Inf
        tmp_best_w = policy.w
        for w in possible_w:
            tmp_rewards = []
            state = env.reset()
            policy.w = w
            for t in range(max_t):
                action = policy.act(state)
                state, reward, done, _ = env.step(action)
                tmp_rewards.append(reward)
                if done:
                    break
            if sum(tmp_rewards) > tmp_best_R:
                tmp_best_R = sum(tmp_rewards)
                tmp_best_w = w
                rewards = tmp_rewards.copy()
        policy.w = tmp_best_w
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        discounts = [gamma**i for i in range(len(rewards) + 1)]
        R = sum([a * b for a, b in zip(discounts, rewards)])

        if R >= best_R:  # found better weights
            best_R = R
            best_w = policy.w
            noise_scale = max(1e-3, noise_scale / 2)
            possible_w = [policy.w + noise_scale *
                          np.random.rand(*policy.w.shape) for _ in range(neighbour_size + 1)]
        else:  # did not find better weights
            noise_scale = min(2, noise_scale * 2)
            possible_w = [best_w + noise_scale *
                          np.random.rand(*policy.w.shape) for _ in range(neighbour_size + 1)]

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))
        if np.mean(scores_deque) >= 195.0:
            print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))
            policy.w = best_w
            break
    plot(scores, 'steepest ascent with adaptive noise scaling', 'steepest_ascent')
    play(policy, env)


if __name__ == "__main__":
    policy = Policy()
    steepest_ascent(policy)
