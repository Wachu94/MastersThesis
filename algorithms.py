import numpy as np, random, gym, personalGym
from common import save, combine, mutate, setup_weights, run_env, forward_prop, forward_prop, activate, pick_action, load, measure_min_max, preprocess
from optimizers import Adam, SGD
from matplotlib import pyplot as plt

def init(env_name, hidden_layers, tweak_param, algorithmID, load_weights, load_v_weights=True):
    if personalGym.environments.__contains__(env_name):
        env = personalGym.make(env_name, tweak_param)
    else:
        env = gym.make(env_name)
    if tweak_param:
        env_name = "".join((env_name, ".{}".format(tweak_param)))

    logits = isinstance(env.action_space.sample(), int)
    preprocess_img = False
    if hasattr(env.reset(), "shape"):
        preprocess_img = len(env.reset().shape) == 3

    policy_weights = None
    if load_weights:
        if isinstance(load_weights, bool):
            policy_weights = load("{}_{}".format(env_name, algorithmID))
        else:
            policy_weights = load(load_weights)
    if not hasattr(policy_weights, '__iter__'):
        policy_weights = setup_weights(env, hidden_layers)

    if load_v_weights:
        v_weights = load("{}_{}".format(env_name, "V"))
        if not hasattr(v_weights, '__iter__'):
            v_weights = setup_weights(env, hidden_layers=[64, 128], v_weights=True)
        return env, policy_weights, v_weights, logits, preprocess_img
    return env, policy_weights, logits, preprocess_img

def genetic_algorithm(env_name, hidden_layers=tuple(), batch_size=10, lr=1e-3, load_weights=None, tweak_param=None, stochastic=True, render=False):

    env, policy_weights, logits, preprocess_img = init(env_name, hidden_layers, tweak_param, "GA", load_weights, False)
    best_score = -np.inf

    score_buffer = []

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for _ in range(5000):
        score = run_env(env, policy_weights, batch_size, render=render, only_score=True, discrete=logits, preprocess_img= preprocess_img)

        score_buffer.append(score)

        if score > 30000:
            # render = True
            batch_size += 1
            if batch_size > 15:
                return
            continue
        if score > best_score:
            # print(score)
            best_score = score
            save(policy_weights, "{}_GA".format(env_name))
        minimum, maximum = measure_min_max(load("{}_GA".format(env_name)))
        # policy_weights = combine(load("{}_GA".format(env_name)), setup_weights(env, hidden_layers, minimum, maximum))
        policy_weights = mutate(policy_weights, probability=0.5, magnitude=(maximum - minimum))

        ax.plot(range(len(score_buffer)), score_buffer, 'r-')
        fig.canvas.draw()
        fig.canvas.flush_events()

def vanilla_policy_gradient(env_name, hidden_layers=tuple(), batch_size=10, mini_batch=500, epochs=5, lr=1e-3, load_weights=True, tweak_param=None, lr_decrease_interval=10, stochastic=True, render=False):

    env, policy_weights, v_weights, logits, preprocess_img = init(env_name, hidden_layers, tweak_param, "VPG", load_weights)
    counter = 0
    score_buffer = []

    plt.ion()
    fig = plt.figure()
    plt.subplot(111)
    plt.title('batch_size = {}; mini_batch = {}; epochs = {}; lr = {}'.format(batch_size, mini_batch, epochs, lr))

    for _ in range(5000):
        counter += 1
        state_buffer, advantage_buffer, reward_buffer, score = run_env(env, policy_weights, episodes=batch_size, preprocess_img=preprocess_img, discrete=logits, render=render, v_weights=v_weights)

        score_buffer = np.append(score_buffer, score)

        v_weights = Adam(state_buffer[:, 1:], reward_buffer, v_weights, lr=lr, mini_batch=mini_batch,
                         epochs=epochs, logits=False, show_progress_bar=True)

        for i, state in enumerate(state_buffer):
            advantage_buffer[i] *= reward_buffer[i] - forward_prop(state[1:], v_weights)      #Getting actual advatnage estimates

        policy_weights = Adam(state_buffer[:, 1:], advantage_buffer, policy_weights, lr=lr, mini_batch=mini_batch, epochs=epochs, logits=logits, show_progress_bar=True)

        # if counter % lr_decrease_interval == 0:
        #     lr *= 0.9
        #     counter = 0

        save(policy_weights, "{}_VPG".format(env_name))
        save(v_weights, "{}_V".format(env_name))

        plt.plot(range(len(score_buffer)), score_buffer, 'b')

        fig.canvas.draw()
        fig.canvas.flush_events()


def tweaked_policy_gradient(env_name, hidden_layers=tuple(), batch_size=10, mini_batch=500, epochs=5, lr=1e-3, load_weights=True, tweak_param=None, lr_decrease_interval=10, stochastic=True, render=False):

    env, policy_weights, v_weights, logits, preprocess_img = init(env_name, hidden_layers, tweak_param, "VPG", load_weights)
    counter = 0
    score_buffer = []
    best_score = -np.inf

    plt.ion()
    fig = plt.figure()
    plt.subplot(111)
    plt.title('batch_size = {}; mini_batch = {}; epochs = {}; lr = {}'.format(batch_size, mini_batch, epochs, lr))

    for _ in range(5000):
        counter += 1
        state_buffer, advantage_buffer, reward_buffer, avg_score = run_env(env, policy_weights, episodes=batch_size, preprocess_img=preprocess_img, discrete=logits, render=render)

        score_buffer.append(avg_score)


        if avg_score > best_score:
            best_score = avg_score
            v_weights = Adam(state_buffer, reward_buffer, v_weights, lr=lr, mini_batch=mini_batch, epochs=epochs, logits=False, show_progress_bar=True)
        # else:
        #     v_weights = Adam(state_buffer, reward_buffer, v_weights, lr=lr/100, mini_batch=mini_batch,
        #                      epochs=epochs, logits=False, show_progress_bar=True)

        for i, state in enumerate(state_buffer):
            advantage_buffer[i] *= (reward_buffer[i] - forward_prop(state, v_weights))      #Getting actual advatnage estimates

        policy_weights = Adam(state_buffer[:, 1:], advantage_buffer, policy_weights, lr=lr, mini_batch=mini_batch,
                              epochs=epochs, logits=logits, show_progress_bar=True)
        #
        # if counter % lr_decrease_interval == 0:
        #     lr *= 0.9
        #     counter = 0

        save(policy_weights, "{}_VPG".format(env_name))
        save(v_weights, "{}_V".format(env_name))

        plt.plot(range(len(score_buffer)), score_buffer, 'b')

        fig.canvas.draw()
        fig.canvas.flush_events()

def proximal_policy_optimization(env, policy_weights, v_weights, batch_size=10, mini_batch=500, epochs=5, lr=1e-3, epsilon=1e-3, save_name="save", save_interval=1, preprocess_img=False, logits=True, render=False):
    try:
        counter = 0
        while True:
            counter += 1
            state_buffer, advantage_buffer, reward_buffer = run_env(env, policy_weights, episodes=batch_size, preprocess_img=preprocess_img, discrete=logits, render=render)

            possible_actions = len(advantage_buffer[0])

            sum = 0
            estimation_buffer = []

            for i, state in enumerate(state_buffer):
                advantage_buffer[i] *= (reward_buffer[i] - forward_prop(state, v_weights))
                estimation = forward_prop(state[1:], policy_weights)
                estimation_buffer.append(estimation)
                for j in range(possible_actions):
                    if advantage_buffer[i][j] >= 0:
                        if advantage_buffer[i][j]/estimation[j] > (1 + epsilon):
                            advantage_buffer[i][j] = (estimation[j] * (1 + epsilon))
                    elif advantage_buffer[i][j]/estimation[j] < (1 - epsilon):
                        advantage_buffer[i][j] = (estimation[j] * (1 - epsilon))
                    sum += np.log10(advantage_buffer[i][j] / estimation[j])
            advantage_buffer = estimation_buffer + (advantage_buffer/(abs(sum)/100))

                # for j in range(possible_actions):
                #     if advantage_buffer[i][j] >= 0:
                #         if advantage_buffer[i][j]/estimation[j] > (1 + epsilon):
                #             advantage_buffer[i][j] = (estimation[j] * (1 + epsilon)) * advantage_buffer[i][j]
                #     elif advantage_buffer[i][j]/estimation[j] < (1 - epsilon):
                #         advantage_buffer[i][j] = (estimation[j] * (1 - epsilon)) * advantage_buffer[i][j]

            policy_weights = Adam(state_buffer[:, 1:], advantage_buffer, policy_weights, lr=lr, mini_batch=mini_batch, epochs=epochs, logits=logits)
            v_weights = SGD(state_buffer, np.reshape(reward_buffer, (-1, 1)), v_weights, lr=lr, mini_batch=mini_batch, epochs=50, logits=False)

            if counter % save_interval == 0:
                counter = 0
                save(policy_weights, "{}_PO".format(save_name))
                save(v_weights, "{}_V".format(save_name))
    except KeyboardInterrupt:
        env.close()

def soft_actor_critic(env, policy_weights, q1_weights, q2_weights, v_weights, v_target_weights, batch_size=10, mini_batch=1000, epochs=5, lr=0.999, save_name="save", save_interval=1, preprocess_img=False, logits=True, render=False):
    try:
        observation = env.reset()
        score = [0 for _ in range(5)]
        counter = 0
        replay_buffer = []

        discount_factor = 0.999
        v_target_decay = 0.9
        while True:
            observation = np.reshape(observation, -1)

            logits = forward_prop(observation, policy_weights)
            probs = activate(logits, "Sigmoid")
            probs /= np.sum(probs)

            action = pick_action(probs)

            # if preprocess_img:
            #     prev_observation = current_observation

            new_observation, reward, done, info = env.step(action)
            score[counter%len(score)] += reward

            if not done:
                replay_buffer.append((observation, action, reward, new_observation, 0))

            #
            # if preprocess_img:
            #     current_observation = observation
            #     observation -= prev_observation

            if done:
                replay_buffer.append((observation, action, reward, new_observation, 1))

                counter += 1
                # if counter < len(score):
                #     print("Avg score: {:.7f}".format(np.sum(score) / counter))
                # else:
                #     print("Avg score: {:.7f}".format(np.sum(score) / len(score)))
                score[counter % len(score)] = 0
                observation = env.reset()

                if counter % batch_size == 0:
                    print("Avg score: {:.7f}".format(np.sum(score) / len(score)))
                    for _ in range(epochs):
                        random.shuffle(replay_buffer)
                        batch = np.array(replay_buffer[:batch_size])
                        y_q = [0 for _ in range(batch_size)]
                        y_v = [0 for _ in range(batch_size)]
                        y_p = [0 for _ in range(batch_size)]
                        for i, transition in enumerate(batch):
                            y_q[i] = [transition[2] + discount_factor * (1 - transition[4]) * forward_prop(transition[3], v_target_weights)[0]]
                            Q1 = forward_prop(transition[0], q1_weights)[transition[1]]
                            Q2 = forward_prop(transition[0], q2_weights)[transition[1]]
                            action_prob = lr * forward_prop(transition[0], policy_weights)[transition[1]]
                            y_v[i] = [min(Q1, Q2) - action_prob]
                            y_p[i] = [Q1 - action_prob]
                        q1_weights = Adam(batch[:, 0], y_q, q1_weights)
                        q2_weights = Adam(batch[:, 0], y_q, q2_weights)
                        v_weights = Adam(batch[:, 0], y_v, v_weights)
                        policy_weights = Adam(batch[:, 0], y_p, policy_weights)

                        for i in range(len(v_target_weights)):
                            v_target_weights[i] = v_target_decay * v_target_weights[i] + (1 - v_target_decay) * v_weights[i]


    except KeyboardInterrupt:
        env.close()

if __name__ == "__main__":

    env_name = "LunarLander-v2"
    # genetic_algorithm(env_name, hidden_layers=[], render=True, batch_size=10, lr=1e-3, stochastic=True, load_weights=True)
    vanilla_policy_gradient(env_name, render=True, batch_size=10, epochs=500, mini_batch=2000, lr=1e-3, stochastic=True, load_weights="LunarLander-v2_VPG_best")
    # tweaked_policy_gradient(env_name, hidden_layers=[32], render=False, batch_size=10, epochs=500, mini_batch=1000, lr=1e-3, stochastic=True, load_weights=False)