import numpy as np
from numpy.linalg import inv
import gym
from tqdm import trange

env_name = "CartPole-v1"
env = gym.make(env_name)

train_episodes = 1000
test_episodes = 50

observation = env.reset()
observation = np.reshape(observation, (-1, 1))


batch_size = 10
lr = 1e-4
batch_x = np.zeros((len(observation) + 1, batch_size), dtype=np.float32)
batch_y = np.zeros((env.action_space.n, batch_size), dtype=np.float32)
alfa = np.zeros((env.action_space.n, len(observation) + 1), dtype=np.float32)
weights = np.zeros((env.action_space.n, len(observation)), dtype=np.float32)
biases = np.zeros((env.action_space.n, 1), dtype=np.float32)

def train_batch():
    global alfa
    try:
        Z = inv(np.matmul(batch_x, np.transpose(batch_x)))
        alfa *= (1 - lr)
        alfa += lr * np.transpose(np.matmul(np.matmul(Z, batch_x), np.transpose(batch_y)))
        for i in range(env.action_space.n):
            for j in range(len(observation) + 1):
                if j == 0:
                    biases[i][0] = alfa[i][j]
                else:
                    weights[i][j-1] = alfa[i][j]
    except:
        print("Wyznacznik równy 0")

def train():
    progress_bar = trange(train_episodes)
    for _ in progress_bar:
        counter = 0
        score = 0
        observation = env.reset()
        while True:
            observation = np.reshape(observation, (-1, 1))
            batch_x[0][counter % batch_size] = 1
            for j in range(len(observation)):
                batch_x[j+1][counter % batch_size] = observation[j]
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            score += reward

            for j in range(env.action_space.n):
                batch_y[j][counter % batch_size] = 0
                if np.matmul(weights, observation)[j] + biases[j] > 0:
                    batch_y[j][counter % batch_size] = np.matmul(weights, observation)[j] + biases[j]
            batch_y[action][counter % batch_size] = score

            counter += 1

            if counter % batch_size == 0:
                train_batch()

            if done == True:
                train_batch()
                env.reset()
                break


def test():
    for i in range(test_episodes):
        score = 0
        observation = env.reset()
        while True:
            env.render()
            observation = np.reshape(observation, (-1, 1))
            action = np.argmax(np.matmul(weights, observation) + biases)
            observation, reward, done, info = env.step(action)
            score+=reward
            if done == True:
                env.reset()
                break
        print("Test episode ",i,": ",score,sep="")

train()
test()
