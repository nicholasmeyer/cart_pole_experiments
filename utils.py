import gym
import numpy as np
import matplotlib.pyplot as plt


def create_env():
    env = gym.make('CartPole-v0')
    env.seed(0)
    np.random.seed(0)

    print('observation space:', env.observation_space)
    print('action space:', env.action_space)
    return env


def plot(scores, method, fig_name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid()
    ax.plot(np.arange(len(scores)), scores)
    ax.set(xlabel="Episode #", ylabel="Score", title=str(method))
    fig.savefig(fig_name + '.pdf')


def play(policy, env):
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


class Policy:
    def __init__(self, s_size=4, a_size=2):
        # weights for simple linear policy: state_space x action_space
        self.w = 1e-4 * np.random.rand(s_size, a_size)

    def forward(self, state):
        x = np.dot(state, self.w)
        return np.exp(x) / sum(np.exp(x))  # softmax activation

    def act(self, state):
        probs = self.forward(state)
        # action = np.random.choice(2, p=probs) # option 1: stochastic policy
        action = np.argmax(probs)               # option 2: deterministic policy
        return action
