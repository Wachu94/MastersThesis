import numpy as np, random, pickle
from tqdm import trange


def save(data, name="save"):
    with open(name + ".p", 'wb') as file:
        pickle.dump(data, file)


def load(name):
    try:
        with open(name + ".p", 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None


def calculate_gradient(X, Y, weights, activation="ReLU", logits=True):
    neuron_values = []
    output = X
    for i in range(len(weights)):
        output = np.c_[np.ones(len(output)), output]
        neuron_values.append(np.array(output))
        output = output.dot(weights[i])
        if i < len(weights) - 1:
            output = activate(output, activation)
    if logits:
        output = activate(output, "Sigmoid")
    errors = [Y - output]
    loss = np.sum(np.sum((errors[len(errors) - 1]) ** 2))
    for i in range(len(weights) - 1):
        errors.append(errors[i].dot(weights[len(weights) - 1 - i].T))
        errors[i + 1] *= neuron_values[len(neuron_values) - 1 - i]
        errors[i + 1] = np.delete(errors[i + 1], 0, 1)
        if activation == "Sigmoid":
            errors[i+1] = errors[i + 1] * (1 - errors[i + 1])
    errors.reverse()

    gradient = []
    for i in range(len(neuron_values)):
        gradient.append(neuron_values[i].T.dot(errors[i]) / neuron_values[i].shape[0])

    return gradient, loss


def setup_weights(env, hidden_layers, minimum=0, maximum=1, v_weights=False):
    if env != None:
        input_size = len(np.reshape(env.reset(), -1))
        if hasattr(env.reset(), "shape"):
            if len(env.reset().shape) == 3:
                input_size = len(np.reshape(preprocess(env.reset()), -1))
        if isinstance(env.action_space.sample(), int):
            output_size = env.action_space.n
        else:
            output_size = len(env.action_space.sample())
        layers = [input_size, *hidden_layers, output_size]
    else:
        layers = hidden_layers
    if v_weights:
        # layers[0] += 1
        layers[-1] = 1
    weights = []
    for i in range(len(layers) - 1):
        # weights.append(np.random.uniform(minimum, maximum, size=(layers[i] + 1, layers[i + 1])))
        weights.append(np.random.randn(layers[i] + 1, layers[i + 1]) / (layers[i] + 1))
    return weights


def activate(X, func_name):
   if func_name == "ReLU":
       X[X<0] = 0
   elif func_name == "Sigmoid":
       for i in range(len(X)):
           X[i] = 1/(1 + np.exp(-X[i]))
   elif func_name == "Softmax":
       sum = 0
       for i in range(len(X)):
           X[i] = np.exp(X[i])
           sum += X[i]
       X /= sum
   return X


def forward_prop(X, weights, activation="ReLU"):
    for i, weight in enumerate(weights):
        if hasattr(X[0], '__iter__'):
            X = np.c_[np.ones((len(X), 1)), X]
        else:
            X = np.r_[1, X]
        X = np.dot(X, weight)
        if i != len(weights) - 1:
            activate(X, activation)
    return X


def backward_prop(errors, neuron_values, weights, activation="ReLU"):
    errors = [errors]
    for i in range(len(weights) - 1):
        errors.append(np.dot(errors[i], weights[len(weights) - 1 - i].T))
        errors[i + 1] = errors[i + 1] * neuron_values[len(neuron_values) - 1 - i]
        if activation == "Sigmoid":
            errors[i + 1] = errors[i + 1] * (1 - errors[i + 1])
        elif activation == "ReLU":
            errors[i + 1][neuron_values[len(neuron_values) - 1 - i] < 0] = 0
        errors[i + 1] = np.delete(errors[i + 1], errors[i + 1].shape[1] - 1, 1)
    errors.reverse()
    gradient = []
    for i in range(len(neuron_values)):
        gradient.append(neuron_values[i].T.dot(errors[i]) / -neuron_values[i].shape[0])
    return gradient


def combine(A, B):
    for i in range(len(A)):
        min_r = random.randint(0,len(A[i]))
        min_c = random.randint(0,len(A[i][0]))
        A[i][min_r:][min_c:] = B[i][min_r:][min_c:]
    return A

def mutate(weights, probability=0.1, magnitude=1):
    for i in range(len(weights)):
        for r in range(len(weights[i])):
            for c in range(len(weights[i][0])):
                if np.random.uniform() < probability:
                    weights[i][r][c] += np.random.uniform(-magnitude,
                                                          magnitude)
    return weights


def measure_min_max(weights):
   minimum = np.inf
   maximum = -np.inf
   for layer in weights:
       if np.max(layer) > maximum:
           maximum = np.max(layer)
       if np.min(layer) < minimum:
           minimum = np.min(layer)
   return minimum, maximum


def preprocess(observation):
    observation = observation[::4, ::4, 0]
    return observation.astype(np.float).ravel()


def pick_action(probabilities):
    random_value = random.random()
    temp = 0
    for i, prob in enumerate(probabilities):
        temp += prob
        if random_value < temp:
            return i
    return 0


def get_v(reward_buffer, horizon, discount):
    V = []
    for episode in range(len(reward_buffer)):
        for i in range(len(reward_buffer[episode])):
            V.append(0)
            for t in range(i, min(i + horizon, len(reward_buffer[episode]))):
                V[len(V) - 1] += reward_buffer[episode][t] * discount**(t-i)
    return np.array(V)


def run_env(env, weights, episodes, activation="ReLU", preprocess_img=False, discrete=True, render=False, show_progress_bar=True, only_score=False, v_weights=None):

    state_buffer, action_buffer, reward_buffer = [], [], [[] for _ in range(episodes)]

    if show_progress_bar:
        progress_bar = trange(episodes)
    else:
        progress_bar = range(episodes)

    score = [0] * episodes
    for i, _ in enumerate(progress_bar):
        observation = env.reset()
        if preprocess_img:
            observation = preprocess(observation)
            current_observation = observation
        while True:
            if render:
                env.render()

            observation = np.reshape(observation, -1)
            logits = forward_prop(observation, weights, activation)
            probs = activate(logits, "Sigmoid")
            probs /= np.sum(probs)

            if discrete:
                action = pick_action(probs)

                decision = -probs
                decision[action] += 1

            else:
                action = probs
                decision = action

            state_buffer.append(observation)
            action_buffer.append(decision)

            if preprocess_img:
                prev_observation = current_observation

            observation, reward, done, info = env.step(action)
            score[i] += reward

            if preprocess_img:
                observation = preprocess(observation)
                current_observation = observation
                observation -= prev_observation

            reward_buffer[i].append(reward)

            if done:
                if show_progress_bar:
                    progress_bar.set_description("Avg score: {:.7f}".format(np.sum(score)/(i+1)))
                break
    progress_bar.close()
    if only_score:
        return score
    return np.array(state_buffer), action_buffer, np.reshape(get_v(reward_buffer, 10000, 0.99), (-1, 1)), score