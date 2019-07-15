import time

import gym
import envs
import numpy as np

np.set_printoptions(precision=3, suppress=True, threshold=10000, linewidth=250)

""" Load environment """
# env_name = 'MazeSample3x3-v0'
env_name = 'MazeSample5x5-v0'
# env_name = 'MazeSample10x10-v0'
# env_name = 'MazeRandom10x10-v0'
# env_name = 'MazeRandom10x10-plus-v0'
# env_name = 'MazeRandom20x20-v0'
# env_name = 'MazeRandom20x20-plus-v0'
# env_name = 'MyCartPole-v0'
# env_name = 'MyMountainCar-v0'

env = gym.make(env_name)
env.T = env.R = None

"""
env.S: the number of states (integer)
env.A: the number of actions (integer)
env.gamma: discount factor (0 ~ 1)
"""


def epsilon_greedy(Q, s, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(0, env.A)
    else:
        return np.argmax(Q[s, :])


alpha = 0.2

Q = np.zeros((env.S, env.A))
epsilon = 0.5
epsilon_min = 0.1
lamb = 0.9

for episode in range(1000):
    e = np.zeros((env.S, env.A)) #e
    state = env.reset()  #s
    action = epsilon_greedy(Q, state, epsilon) #a
    env.render()

    episode_reward = 0. #e(s,a) ← for all s,a
    for t in range(10000):
        next_state, reward, done, info = env.step(action)
        next_action = epsilon_greedy(Q, next_state, epsilon)

        # Update Q-table
        target_Q = reward if done else reward + env.gamma * np.max(Q[next_state, :])
        delta = target_Q - Q[state, action]
        ###################
        # TODO: 아래 부분을 수정하여 Q(\lambda)를 구현해 봅시다.
        e[state, action] += 1
        Q[state, action] = Q[state, action] + alpha * delta  # 이건 일반 Q-learning

        for s in range(env.S):
            for a in range(env.A):
                Q[state, action] = Q[state, action] + alpha * delta*(e[state, action])
                if action == np.max(Q[next_state, :]):
                    e[state, action]=env.gamma*lamb*(e[state, action])
                else:
                    e = np.zeros((env.S, env.A))

        ####################

        episode_reward += reward
        print("[epi=%4d,t=%4d] state=%4s / action=%d / reward=%7.4f / next_state=%4s / info=%s / Q[s]=%s" % (episode, t, state, action, reward, next_state, info, Q[state, :]))

        env.draw_policy_evaluation(Q)
        env.render()
        time.sleep(0.01)

        if done:
            break
        state = next_state
        action = next_action

    epsilon = np.max([epsilon * 0.9, epsilon_min])
    print('[%4d] Episode reward=%.4f / epsilon=%f' % (episode, episode_reward, epsilon))

    time.sleep(0.1)
time.sleep(10)
