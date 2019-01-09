import random, numpy as np
from tqdm import trange
from common import forward_prop, calculate_gradient, activate


def SGD(X, Y, weights, activation="ReLU", lr=1e-2, mini_batch=500, epochs=10, logits=True, show_progress_bar=False):
    if show_progress_bar:
        progress_bar = trange(epochs)
    else:
        progress_bar = range(epochs)
    for _ in progress_bar:

        learning_vectors = list(zip(X, Y))
        random.shuffle(learning_vectors)
        X, Y = zip(*learning_vectors)
        X = X[:min(mini_batch, len(X))]
        Y = Y[:min(mini_batch, len(Y))]

        gradient, loss = calculate_gradient(X, Y, weights, activation=activation, logits=logits)
        if show_progress_bar:
            progress_bar.set_description("Current loss: {:.7f}".format(loss))
        for i in range(len(gradient)):
            weights[i] += gradient[i] * lr
    return weights


def Adam(X, Y, weights, activation="ReLU", lr=1e-3, b1=0.9, b2=0.999, epsilon=1e-8, mini_batch=500, epochs=500, logits=True, show_progress_bar=False, descent=True):
    if show_progress_bar:
        progress_bar = trange(epochs)
    else:
        progress_bar = range(epochs)
    prev_loss = np.inf
    for j, _ in enumerate(progress_bar):

        learning_vectors = list(zip(X, Y))
        random.shuffle(learning_vectors)
        X, Y = zip(*learning_vectors)
        X = X[:min(mini_batch, len(X))]
        Y = Y[:min(mini_batch, len(Y))]

        gradient, loss = calculate_gradient(X, Y, weights, activation=activation, logits=logits)
        # if loss > prev_loss and lr > 1e-12:
        #     lr /= 10
        # prev_loss = loss

        if show_progress_bar:
            progress_bar.set_description("Current loss: {:.7f}".format(loss))
        m = [[] for _ in range(len(gradient))]
        v = [[] for _ in range(len(gradient))]
        for i, layer in enumerate(gradient):
            if len(m[i]) == 0:
                m[i] = np.zeros_like(layer)
                v[i] = np.zeros_like(layer)
            m[i] = b1 * m[i] + (1 - b1) * layer
            v[i] = b2 * v[i] + (1 - b2) * (layer ** 2)
            m[i] = m[i] / (1 - b1 ** (j + 1))
            v[i] = v[i] / (1 - b2 ** (j + 1))
            if descent:
                weights[i] += ((lr*m[i])/((v[i]**0.5)+epsilon))
            else:
                weights[i] -= ((lr * m[i]) / ((v[i] ** 0.5) + epsilon))

    return weights