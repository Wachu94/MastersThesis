import personalGym, random, time
import numpy as np, numba as nb
import pickle
from tqdm import trange

env_name = "Othello-v0"
BOARD_SIZE = 4
env = personalGym.make(env_name, BOARD_SIZE)

def setupWeights(*hidden_layers):
    input_size = len(np.reshape(env.reset(), (-1, 1)))
    layer_sizes = [input_size, *hidden_layers, BOARD_SIZE**2]
    weights = []
    for i in range(1, len(layer_sizes)):
        weights.append([])
        for _ in range((layer_sizes[i - 1]+1) * layer_sizes[i]):
            weights[i - 1].append((random.random()-0.5)*2)
        weights[i - 1] = np.reshape(weights[i - 1], (layer_sizes[i], layer_sizes[i - 1] + 1))
    return weights

def calculateOutput(observation, weights, max = False):
    output = observation
    for i in range(len(weights)):
        output = np.append(output, 1)
        output = weights[i].dot(output)
        for j in range(len(output)):
            if output[j] < 0:
                output[j] = 0
    if max:
        return np.argmax(output)
    return output

def mutateWeights(weights, probability = 0.1, max_value = 1):
    for i in range(len(weights)):
        for x in range(len(weights[i])):
            for y in range(len(weights[i][x])):
                if random.random() < probability:
                    weights[i][x][y] += (-0.5 + random.random()) * 2 * max_value
    return weights

def combineWeights(weights, prev_weights):
    for i in range(len(weights)):
        min_x = random.randint(0, len(weights[i]))
        min_y = random.randint(0, len(weights[i][0]))
        for x in range(len(weights[i])):
            if x >= min_x:
                for y in range(len(weights[i][x])):
                    if y >= min_y:
                        weights[i][x][y] = prev_weights[i][x][y]
    return weights

def train(episodes, batch_size, load_weights = False):
    best_score = -np.inf
    if load_weights:
        player1 = load()
    else:
        player1 = setupWeights()
    progress_bar = trange(episodes)
    for _ in progress_bar:
        combined_score = 0
        for _ in range(batch_size):
            player2 = setupWeights()
            observation = env.reset()
            while True:
                observation = np.reshape(observation, (-1, 1))
                action = calculateOutput(observation, player1)
                observation, reward, done, info = env.step(action)
                if done:
                    combined_score += reward
                    break
                observation = np.reshape(observation, (-1, 1))
                action = calculateOutput(observation, player2)
                observation, reward, done, info = env.step(action)
                if done:
                    combined_score += reward
                    break
            observation = env.reset()
            while True:
                observation = np.reshape(observation, (-1, 1))
                action = calculateOutput(observation, player2)
                observation, reward, done, info = env.step(action)
                if done:
                    combined_score -= reward
                    break
                observation = np.reshape(observation, (-1, 1))
                action = calculateOutput(observation, player1)
                observation, reward, done, info = env.step(action)
                if done:
                    combined_score -= reward
                    break
        if combined_score > best_score:
            best_score = combined_score
            combined_score /= batch_size
            progress_bar.set_description("Best score: %f" % combined_score)
            save(player1)
        if random.random() < 0.5:
            player1 = combineWeights(load(), setupWeights())
        else:
            player1 = combineWeights(setupWeights(), load())
        player1 = mutateWeights(player1)
    progress_bar.close()
    return player1


def test(episodes, weights = None):
    if weights:
        player1 = weights
    else:
        player1 = load()
    for _ in range(episodes):
        player2 = setupWeights()
        observation = env.reset()
        combined_score = 0
        while True:
            env.render()
            time.sleep(0.3)
            observation = np.reshape(observation, (-1, 1))
            action = calculateOutput(observation, player1)
            observation, reward, done, info = env.step(action)
            if done:
                combined_score += reward
                print(reward)
                break
            env.render()
            time.sleep(0.3)
            observation = np.reshape(observation, (-1, 1))
            action = calculateOutput(observation, player2)
            observation, reward, done, info = env.step(action)
            if done:
                combined_score += reward
                print(reward)
                break
        observation = env.reset()
        while True:
            env.render()
            time.sleep(0.3)
            observation = np.reshape(observation, (-1, 1))
            action = calculateOutput(observation, player2)
            observation, reward, done, info = env.step(action)
            if done:
                combined_score -= reward
                print(-reward)
                break
            env.render()
            time.sleep(0.3)
            observation = np.reshape(observation, (-1, 1))
            action = calculateOutput(observation, player1)
            observation, reward, done, info = env.step(action)
            if done:
                combined_score -= reward
                print(-reward)
                break
        print("Combined score:", combined_score)

def challenge():
    player1 = load()
    observation = env.reset()
    combined_score = 0
    while True:
        env.render()
        observation = np.reshape(observation, (-1, 1))
        action = calculateOutput(observation, player1)
        observation, reward, done, info = env.step(action)
        if done:
            combined_score -= reward
            print(combined_score)
            if combined_score > 0:
                print("You won!")
            elif combined_score == 0:
                print("It's a tie.")
            else:
                print("You lost!")
            break
        env.render()
        observation, reward, done, info = env.manual_step()
        if done:
            combined_score -= reward
            print(combined_score)
            if combined_score > 0:
                print("You won!")
            elif combined_score == 0:
                print("It's a tie.")
            else:
                print("You lost!")
            break
    env.reset()
    combined_score = 0
    while True:
        env.render()
        observation, reward, done, info = env.manual_step()
        if done:
            combined_score += reward
            print(combined_score)
            if combined_score > 0:
                print("You won!")
            elif combined_score == 0:
                print("It's a tie.")
            else:
                print("You lost!")
            break
        env.render()
        observation = np.reshape(observation, (-1, 1))
        action = calculateOutput(observation, player1)
        observation, reward, done, info = env.step(action)
        if done:
            combined_score += reward
            print(combined_score)
            if combined_score > 0:
                print("You won!")
            elif combined_score == 0:
                print("It's a tie.")
            else:
                print("You lost!")
            break


def save(data, file = env_name):
    pickle.dump(data, open(file + "_" + str(BOARD_SIZE) + ".p", "wb"))

def load(file = env_name):
    return pickle.load(open(file + "_" + str(BOARD_SIZE) + ".p","rb"))


train(50, 100)
test(10)
# challenge()