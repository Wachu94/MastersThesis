import numpy as np
from tqdm import trange
from matplotlib import pyplot as plt, colors

def activation_function(input, type = "relu"):
    if type == "sigmoid":
        return 1 / (1 + np.exp(-input))
    elif type == "relu":
        for i in range(len(input)):
            for j in range(len(input[0])):
                if input[i][j] < 0:
                    input[i][j] = 0
        return input
    elif type == "unipolar":
        for i in range(len(input)):
            for j in range(len(input[0])):
                if input[i][j] < 0:
                    input[i][j] = 0
                else:
                    input[i][j] = 1
        return input
    elif type == "elu":
        for i in range(len(input)):
            for j in range(len(input[0])):
                if input[i][j] < 0:
                    input[i][j] = np.exp(input[i][j]) - 1
        return input

def calculate_predictions(X, weights, activation="sigmoid"):
    neuron_values = []
    output = X
    for i in range(len(weights)):
        output = np.c_[output, np.ones(output.shape[0])]
        neuron_values.append(np.array(output))
        output = activation_function(output.dot(weights[i]), activation)
    errors = [y - output]
    for i in range(len(weights) - 1):
        errors.append(errors[i].dot(weights[len(weights) - 1 - i].T))
        errors[i + 1] = errors[i + 1] * neuron_values[len(neuron_values) - 1 - i]
        errors[i + 1] = np.delete(errors[i + 1], errors[i + 1].shape[1] - 1, 1)
        if activation == "sigmoid":
            errors[i+1] = errors[i + 1] * (1 - errors[i + 1])
    errors.reverse()
    return neuron_values, errors

def calculate_output(X, weights, activation="sigmoid"):
    output = X
    for i in range(len(weights)):
        output = np.c_[output, np.ones(output.shape[0])]
        output = activation_function(output.dot(weights[i]), activation)
    return output

X = np.array([[0, 0], [0, 1] , [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
episodes = 10000
lr = 1e-0
layer_sizes = (X.shape[1], 2, 1)
activation_func = "relu"
required_loss = 1e-5
pixels_per_side = 50

# X = np.c_[X, np.ones(X.shape[0])]

weights = []

for i in range(len(layer_sizes) - 1):
    weights.append(np.random.uniform(size=(layer_sizes[i] + 1,layer_sizes[i+1])))
    # weights.append(np.ones((layer_sizes[i] + 1,layer_sizes[i+1])))

# weights = np.random.uniform(size=(X.shape[1],))

loss_history = []

progress_bar = trange(episodes)
try:
    for i in progress_bar:
        neuron_values, errors = calculate_predictions(X, weights, activation_func)

        loss = 0
        # loss += np.sum(np.sum(abs(errors[len(errors) - 1])))
        loss += np.sum(np.sum(errors[len(errors) - 1] ** 2))
        loss_history.append(loss)

        gradient = []
        for i in range(len(neuron_values)):
            gradient.append(neuron_values[i].T.dot(errors[i]) / -neuron_values[i].shape[0])
        for i in range(len(weights)):
            weights[i] -=  lr * gradient[i]

        progress_bar.set_description("Current loss: {:.7f}".format(loss))
        if loss < required_loss:
            break
except KeyboardInterrupt:
    pass
finally:
    progress_bar.close()

# print(weights)
# print(calculate_output(X, weights, activation_func))
plt.plot(np.arange(1, (len(loss_history) + 1), 1),loss_history)
# plt.show()

x0, x1 = 0, 1
y0, y1 = 0, 1
x = np.linspace(x0, x1, pixels_per_side)
y = np.linspace(y0, y1, pixels_per_side)
X, Y = np.meshgrid(x, y)
Z = np.zeros((pixels_per_side, pixels_per_side))

for r in range(pixels_per_side):
    for c in range(pixels_per_side):
        Z[r][c] = calculate_output(np.array([[X[r][c], Y[r][c]]]), weights, activation_func)

palette = plt.cm.gray
Zm = np.ma.masked_where(Z > 1, Z)

fig, ax1 = plt.subplots(nrows=1, figsize=(6, 5.4))

im = ax1.imshow(Zm, interpolation='none',
                cmap=palette,
                norm=colors.Normalize(vmin=0, vmax=1.0),
                aspect='auto',
                origin='lower',
                extent=[x0, x1, y0, y1])
cbar = fig.colorbar(im, extend='both', shrink=0.9, ax=ax1)
plt.show()