import personalGym, random, time
import numpy as np
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
        if i == len(weights) - 1:
            output = softmax(output)
    if max:
        return np.argmax(output)
    return output

def softmax(output_layer):
    output = output_layer
    sum = 0
    for element in output_layer:
        # sum += np.exp(element)
        sum += element
    for i in range(len(output)):
        # output[i] = np.exp(output[i])/sum
        output[i] = output[i] / sum
    return output

def calculate_neuron_values(observation, weights):
    neuron_values = [observation]
    output = observation
    for i in range(len(weights)):
        output = np.append(output, 1)
        output = np.matmul(weights[i], output)
        for j in range(len(output)):
            if output[j] < 0:
                output[j] /= 1000
        if i == len(weights) - 1:
            output = softmax(output)
        neuron_values.append(output)
        neuron_values[i] = np.append(neuron_values[i], 1)
    return neuron_values

def calculate_errors(neuron_values, weights, y):
    combined_errors = 0
    errors = [[]]*(len(neuron_values)-1)
    errors[len(errors) - 1] = y - neuron_values[len(errors)]
    for i in range(2, len(weights)+1):
        errors[len(errors) - i] = np.matmul(errors[len(errors) - (i-1)], weights[len(weights)-(i-1)])
    return errors

def update_weights(errors, weights, neuron_values, lr=0.01):
    for i in range(len(weights)):
        for j in range(len(weights[i])):
            for k in range(len(weights[i][j])):
                try:
                    correction = errors[i][j] * neuron_values[i][k] * lr
                    weights[i][j][k] += correction
                except:
                    pass
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

def train(episodes, batch_size, load_weights = False, against_myself=False):
    try:
        counter, sum = 0, 0
        if load_weights:
            player1 = load()
        else:
            player1 = setupWeights(64)
        progress_bar = trange(episodes)
        for _ in progress_bar:
            combined_score = 0
            for _ in range(batch_size):
                neuron_values_p2 = []
                chosen_pos_p2 = []
                if against_myself:
                    player2 = load()
                else:
                    player2 = setupWeights(64)
                observation = env.reset()
                while True:
                    observation = np.reshape(observation, (-1, 1))
                    # neuron_values_p1.append(calculate_neuron_values(observation, player1))
                    action = calculateOutput(observation, player1)
                    observation, reward, done, info = env.step(action)
                    # chosen_pos_p1.append(info)
                    if done:
                        if reward < 0:
                            for i in range(len(chosen_pos_p2)):
                                if chosen_pos_p2[i]:
                                    errors = calculate_errors(neuron_values_p2[i], player1, 1)
                                    temp = np.zeros_like(errors[len(errors)-1])
                                    temp[chosen_pos_p2[i][0]*BOARD_SIZE + chosen_pos_p2[i][1]] = np.array(errors[len(errors)-1][chosen_pos_p2[i][0]*BOARD_SIZE + chosen_pos_p2[i][1]])
                                    errors[len(errors)-1] = np.array(temp)
                                    player1 = update_weights(errors, player1, neuron_values_p2[i], lr=0.01)
                        combined_score += reward
                        break
                    observation = np.reshape(observation, (-1, 1))
                    neuron_values_p2.append(calculate_neuron_values(observation, player1))
                    action = calculateOutput(observation, player2)
                    observation, reward, done, info = env.step(action)
                    chosen_pos_p2.append(info)
                    if done:
                        if reward < 0:
                            for i in range(len(chosen_pos_p2)):
                                if chosen_pos_p2[i]:
                                    errors = calculate_errors(neuron_values_p2[i], player2, 1)
                                    temp = np.zeros_like(errors[len(errors) - 1])
                                    temp[chosen_pos_p2[i][0] * BOARD_SIZE + chosen_pos_p2[i][1]] = np.array(
                                        errors[len(errors) - 1][chosen_pos_p2[i][0] * BOARD_SIZE + chosen_pos_p2[i][1]])
                                    errors[len(errors) - 1] = np.array(temp)
                                    player1 = update_weights(errors, player1, neuron_values_p2[i], lr=0.01)
                        combined_score += reward
                        break
                observation = env.reset()
                neuron_values_p2.clear()
                chosen_pos_p2.clear()
                while True:
                    observation = np.reshape(observation, (-1, 1))
                    neuron_values_p2.append(calculate_neuron_values(observation, player2))
                    action = calculateOutput(observation, player2)
                    observation, reward, done, info = env.step(action)
                    chosen_pos_p2.append(info)
                    if done:
                        if reward > 0:
                            for i in range(len(chosen_pos_p2)):
                                if chosen_pos_p2[i]:
                                    errors = calculate_errors(neuron_values_p2[i], player1, 1)
                                    temp = np.zeros_like(errors[len(errors)-1])
                                    temp[chosen_pos_p2[i][0]*BOARD_SIZE + chosen_pos_p2[i][1]] = np.array(errors[len(errors)-1][chosen_pos_p2[i][0]*BOARD_SIZE + chosen_pos_p2[i][1]])
                                    errors[len(errors)-1] = np.array(temp)
                                    player1 = update_weights(errors, player1, neuron_values_p2[i], lr=0.01)
                        combined_score -= reward
                        break
                    observation = np.reshape(observation, (-1, 1))
                    action = calculateOutput(observation, player1)
                    observation, reward, done, info = env.step(action)
                    if done:
                        if reward > 0:
                            for i in range(len(chosen_pos_p2)):
                                if chosen_pos_p2[i]:
                                    errors = calculate_errors(neuron_values_p2[i], player1, 1)
                                    temp = np.zeros_like(errors[len(errors)-1])
                                    temp[chosen_pos_p2[i][0]*BOARD_SIZE + chosen_pos_p2[i][1]] = np.array(errors[len(errors)-1][chosen_pos_p2[i][0]*BOARD_SIZE + chosen_pos_p2[i][1]])
                                    errors[len(errors)-1] = np.array(temp)
                                    player1 = update_weights(errors, player1, neuron_values_p2[i], lr=0.01)
                        combined_score -= reward
                        break
            sum += combined_score
            counter += 1
            avg_score = sum/counter
            progress_bar.set_description("Avg score: %f" % avg_score)
    finally:
        save(player1)
        progress_bar.close()
        return player1


def test(episodes, weights = None):
    if weights:
        player1 = weights
    else:
        player1 = load()
    for _ in range(episodes):
        player2 =  setupWeights()
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

def challenge(episodes=1):
    player1 = load()
    for _ in range(episodes):
        neuron_values_p2 = []
        chosen_pos_p2 = []
        observation = env.reset()
        combined_score = 0
        while True:
            env.render()
            observation = np.reshape(observation, (-1, 1))
            action = calculateOutput(observation, player1)
            print(action)
            observation, reward, done, info = env.step(action)
            if done:
                combined_score -= reward
                print(combined_score)
                if combined_score > 0:
                    for i in range(len(chosen_pos_p2)):
                        if chosen_pos_p2[i]:
                            errors = calculate_errors(neuron_values_p2[i], player1, 1)
                            temp = np.zeros_like(errors[len(errors) - 1])
                            temp[chosen_pos_p2[i][0] * BOARD_SIZE + chosen_pos_p2[i][1]] = np.array(
                                errors[len(errors) - 1][chosen_pos_p2[i][0] * BOARD_SIZE + chosen_pos_p2[i][1]])
                            errors[len(errors) - 1] = np.array(temp)
                            player1 = update_weights(errors, player1, neuron_values_p2[i], lr=1)
                    save(player1)
                    print("You won!")
                elif combined_score == 0:
                    print("It's a tie.")
                else:
                    print("You lost!")
                break
            env.render()
            neuron_values_p2.append(calculate_neuron_values(observation, player1))
            observation, reward, done, info = env.manual_step()
            chosen_pos_p2.append(info)
            if done:
                combined_score -= reward
                print(combined_score)
                if combined_score > 0:
                    for i in range(len(chosen_pos_p2)):
                        if chosen_pos_p2[i]:
                            errors = calculate_errors(neuron_values_p2[i], player1, 1)
                            temp = np.zeros_like(errors[len(errors) - 1])
                            temp[chosen_pos_p2[i][0] * BOARD_SIZE + chosen_pos_p2[i][1]] = np.array(
                                errors[len(errors) - 1][chosen_pos_p2[i][0] * BOARD_SIZE + chosen_pos_p2[i][1]])
                            errors[len(errors) - 1] = np.array(temp)
                            player1 = update_weights(errors, player1, neuron_values_p2[i], lr=1)
                    save(player1)
                    print("You won!")
                elif combined_score == 0:
                    print("It's a tie.")
                else:
                    print("You lost!")
                break
        env.reset()
        neuron_values_p2.clear()
        chosen_pos_p2.clear()
        combined_score = 0
        while True:
            env.render()
            neuron_values_p2.append(calculate_neuron_values(observation, player1))
            observation, reward, done, info = env.manual_step()
            chosen_pos_p2.append(info)
            if done:
                combined_score += reward
                print(combined_score)
                if combined_score > 0:
                    for i in range(len(chosen_pos_p2)):
                        if chosen_pos_p2[i]:
                            errors = calculate_errors(neuron_values_p2[i], player1, 1)
                            temp = np.zeros_like(errors[len(errors) - 1])
                            temp[chosen_pos_p2[i][0] * BOARD_SIZE + chosen_pos_p2[i][1]] = np.array(
                                errors[len(errors) - 1][chosen_pos_p2[i][0] * BOARD_SIZE + chosen_pos_p2[i][1]])
                            errors[len(errors) - 1] = np.array(temp)
                            player1 = update_weights(errors, player1, neuron_values_p2[i], lr=1)
                    save(player1)
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
                    for i in range(len(chosen_pos_p2)):
                        if chosen_pos_p2[i]:
                            errors = calculate_errors(neuron_values_p2[i], player1, 1)
                            temp = np.zeros_like(errors[len(errors) - 1])
                            temp[chosen_pos_p2[i][0] * BOARD_SIZE + chosen_pos_p2[i][1]] = np.array(
                                errors[len(errors) - 1][chosen_pos_p2[i][0] * BOARD_SIZE + chosen_pos_p2[i][1]])
                            errors[len(errors) - 1] = np.array(temp)
                            player1 = update_weights(errors, player1, neuron_values_p2[i], lr=1)
                    save(player1)
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

train(1000, 1)
test(10)
# challenge(10)