import random, numpy as np
from tqdm import trange
from common import forward_prop, calculate_gradient, setup_weights


def SGD(X, Y, weights, activation="ReLU", lr=1e-2, mini_batch=500, epochs=10, logits=True, show_progress_bar=True):
    if show_progress_bar:
        progress_bar = trange(int(epochs))
    else:
        progress_bar = range(int(epochs))
    for _ in progress_bar:

        learning_vectors = list(zip(X, Y))
        random.shuffle(learning_vectors)
        X, Y = zip(*learning_vectors)
        batch_X = X[:min(mini_batch, len(X))]
        batch_Y = Y[:min(mini_batch, len(Y))]

        gradient, loss = calculate_gradient(batch_X, batch_Y, weights, activation=activation, logits=logits)
        if show_progress_bar:
            progress_bar.set_description("Current loss: {:.7f}".format(loss))
        for i in range(len(gradient)):
            weights[i] += gradient[i] * lr
    return weights


def Adam(X, Y, weights, activation="ReLU", lr=1e-3, b1=0.9, b2=0.999, epsilon=1e-8, mini_batch=500, epochs=500, logits=True, show_progress_bar=True, descent=True):
    if show_progress_bar:
        progress_bar = trange(int(epochs))
    else:
        progress_bar = range(int(epochs))

    m = [[] for _ in range(len(weights))]
    v = [[] for _ in range(len(weights))]

    for j in progress_bar:

        learning_vectors = list(zip(X, Y))
        random.shuffle(learning_vectors)
        X, Y = zip(*learning_vectors)
        batch_X = X[:min(mini_batch, len(X))]
        batch_Y = Y[:min(mini_batch, len(Y))]

        gradient, loss = calculate_gradient(batch_X, batch_Y, weights, activation=activation, logits=logits)

        if show_progress_bar:
            progress_bar.set_description("Current loss: {:.7f}".format(loss))
        for i, layer in enumerate(gradient):
            if j == 0:
                m[i] = np.zeros_like(layer)
                v[i] = np.zeros_like(layer)
            m[i] = b1 * m[i] + (1 - b1) * layer
            v[i] = b2 * v[i] + (1 - b2) * (layer ** 2)
            M = m[i] / (1 - b1 ** (j + 1))
            V = v[i] / (1 - b2 ** (j + 1))
            if descent:
                weights[i] += ((lr * M) / ((V ** 0.5) + epsilon))
            else:
                weights[i] -= ((lr * M) / ((V ** 0.5) + epsilon))

    return weights


if __name__ == "__main__":
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    Y = [[0], [1], [1], [0]]

    weights = setup_weights(None, [2, 2, 1])
    weights = Adam(X, Y, weights, epochs=3e4, logits=False)
    print(forward_prop(X, weights))