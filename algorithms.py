import numpy as np, random, gym, personalGym, time
from common import save, combine, mutate, setup_weights, run_env, forward_prop, activate, pick_action, \
    load, measure_min_max, get_v
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
            v_weights = setup_weights(env, hidden_layers=[128], v_weights=True)
        return env, env_name, policy_weights, v_weights, logits, preprocess_img
    return env, env_name, policy_weights, logits, preprocess_img


def genetic_algorithm(env_name, hidden_layers=tuple(), batch_size=10, load_weights=None, tweak_param=None, deterministic=True, render=False):
    env, save_name, policy_weights, logits, preprocess_img = init(env_name, hidden_layers, tweak_param, "GA", load_weights, load_v_weights=False)
    best_score = -np.inf

    score_buffer = []

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    epsilon = 1

    for i in range(5000):
        score = run_env(env, policy_weights, batch_size, render=render, only_score=True, discrete=logits, deterministic=deterministic,
                        preprocess_img=preprocess_img)

        score = np.sum(score) / batch_size

        score_buffer.append(score)

        if score >= best_score:
            best_score = score
            save(policy_weights, "{}_GA".format(save_name))
        policy_weights = load("{}_GA".format(save_name))
        minimum, maximum = measure_min_max(policy_weights)
        # policy_weights = setup_weights(env, hidden_layers, minimum, maximum)
        # policy_weights = combine(load("{}_GA".format(save_name)), setup_weights(env, hidden_layers, minimum, maximum))
        policy_weights = mutate(policy_weights, probability=0.1, magnitude=epsilon * (maximum - minimum))
        epsilon *= 0.95

        plt.cla()
        ax.plot(range(len(score_buffer)), score_buffer, 'r-')
        ax.set_ylim(-200, 100)
        fig.canvas.draw()
        fig.canvas.flush_events()
        # if time.time() - start > 1:
        if i == 250:
            if deterministic:
                plt.savefig("{}_GA_deterministic.png".format(save_name))
            else:
                plt.savefig("{}_GA.png".format(save_name))
            break

def reinforce(env_name, hidden_layers=tuple(), batch_size=10, mini_batch=500, policy_epochs=50, policy_lr=5e-5,
                            load_weights=True, tweak_param=None, render=False, deterministic=False):
    env, save_name, policy_weights, logits, preprocess_img = init(env_name, hidden_layers, tweak_param, "RNF",
                                                                  load_weights, load_v_weights=False)
    score_buffer = []
    mean_buffer = []

    plt.ion()
    fig = plt.figure()
    plt.subplot(111)
    plt.title('batch_size = {}; mini_batch = {}; epochs = {}; lr = {}'.format(batch_size, mini_batch, policy_epochs,
                                                                              policy_lr))

    for _ in range(5000):
        state_buffer, probs_buffer, actions_buffer, reward_buffer, score = run_env(env, policy_weights, episodes=batch_size,
                                                                       preprocess_img=preprocess_img, discrete=logits, deterministic=deterministic,
                                                                       render=render)

        reward_buffer = np.reshape(get_v(reward_buffer, np.inf, 1), (-1, 1))
        reward_buffer = activate(reward_buffer, "Sigmoid")

        score_buffer = np.append(score_buffer, score)
        mean_buffer = np.append(mean_buffer, [np.mean(score)] * batch_size)

        for i in range(len(probs_buffer)):
            probs_buffer[i] = (np.log(probs_buffer[i]) - np.log(1 - probs_buffer[i])) * reward_buffer[i]

        probs_buffer = activate(probs_buffer, "Sigmoid")

        if logits:
            for i in range(len(probs_buffer)):
                probs_buffer[i] *= -1
                probs_buffer[i][actions_buffer[i]] += +1

        policy_weights = Adam(state_buffer, probs_buffer, policy_weights, lr=policy_lr, mini_batch=mini_batch,
                              epochs=policy_epochs, logits=logits, show_progress_bar=True)


        save(policy_weights, "{}_RNF".format(save_name))

        plt.cla()
        plt.clf()
        plt.plot(np.arange(len(score_buffer)), score_buffer, np.arange(len(mean_buffer)), mean_buffer)


        fig.canvas.draw()
        fig.canvas.flush_events()



def vanilla_policy_gradient(env_name, hidden_layers=tuple(), batch_size=10, mini_batch=2000, policy_epochs=50, v_epochs=80, policy_lr=3e-4, v_lr=1e-3,
                            load_weights=True, tweak_param=None, deterministic=False, render=False):
    env, save_name, policy_weights, v_weights, logits, preprocess_img = init(env_name, hidden_layers, tweak_param, "VPG",
                                                                  load_weights)
    score_buffer = []
    mean_buffer = []

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('batch_size = {}; mini_batch = {}; epochs = {}; lr = {}'.format(batch_size, mini_batch, policy_epochs, policy_lr))

    for j in range(5000):
        state_buffer, probs_buffer, actions_buffer, reward_buffer, score, = run_env(env, policy_weights, episodes=batch_size,
                                                                       preprocess_img=preprocess_img, discrete=logits, deterministic=deterministic,
                                                                       render=render, v_weights=v_weights)

        if logits:
            for i in range(len(probs_buffer)):
                probs_buffer[i] *= -1
                probs_buffer[i][actions_buffer[i]] += 1

        reward_buffer = np.reshape(get_v(reward_buffer, 100, 0.99), (-1, 1))

        score_buffer = np.append(score_buffer, score)
        mean_buffer = np.append(mean_buffer, [np.mean(score)] * batch_size)

        for i, state in enumerate(state_buffer):
            probs_buffer[i] *= reward_buffer[i] - forward_prop(state,
                                                                   v_weights)  # Getting actual advatnage estimates

        policy_weights = Adam(state_buffer, probs_buffer, policy_weights, lr=policy_lr, mini_batch=mini_batch,
                              epochs=policy_epochs, logits=logits, show_progress_bar=True)

        v_weights = Adam(state_buffer, reward_buffer, v_weights, lr=v_lr, mini_batch=mini_batch,
                         epochs=v_epochs, logits=False, show_progress_bar=True)

        save(policy_weights, "{}_VPG".format(save_name))
        save(v_weights, "{}_V".format(save_name))

        # ax.cla()
        plt.cla()
        ax.plot(np.arange(len(score_buffer)), score_buffer)
        ax.set_ylim(-15, 15)
        fig.canvas.draw()
        fig.canvas.flush_events()

        if j == 25:
            if deterministic:
                plt.savefig("{}_VPG_deterministic.png".format(save_name))
            else:
                plt.savefig("{}_VPG.png".format(save_name))
            break


def soft_actor_critic(env, policy_weights, q1_weights, q2_weights, v_weights, v_target_weights, batch_size=10,
                      mini_batch=1000, epochs=5, lr=0.999, save_name="save", save_interval=1, preprocess_img=False,
                      logits=True, render=False):
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
        score[counter % len(score)] += reward

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
                        y_q[i] = [transition[2] + discount_factor * (1 - transition[4]) *
                                  forward_prop(transition[3], v_target_weights)[0]]
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

# def deep_deterministic_policy_gradient(env, policy_weights, q_weights, batch_size=10, mini_batch=1000, epochs=5,
#                                        lr=0.999, save_name="save", save_interval=1,
#                                        preprocess_img=False, logits=True, render=False):
#
#     replay_buffer = []
#     policy_target = policy_weights.copy()
#     q_target = q_weights.copy()
#
#     observation = env.reset()
#     score = [0 for _ in range(5)]
#     counter = 0
#
#
#     discount_factor = 0.999
#     v_target_decay = 0.9
#     while True:
#         observation = np.reshape(observation, -1)
#
#         logits = forward_prop(observation, policy_weights)
#         probs = activate(logits, "Sigmoid")
#         probs /= np.sum(probs)
#
#         action = pick_action(probs)
#
#         # if preprocess_img:
#         #     prev_observation = current_observation
#
#         new_observation, reward, done, info = env.step(action)
#         score[counter % len(score)] += reward
#
#         if not done:
#             replay_buffer.append((observation, action, reward, new_observation, 0))
#
#         #
#         # if preprocess_img:
#         #     current_observation = observation
#         #     observation -= prev_observation
#
#         if done:
#             replay_buffer.append((observation, action, reward, new_observation, 1))
#
#             counter += 1
#             # if counter < len(score):
#             #     print("Avg score: {:.7f}".format(np.sum(score) / counter))
#             # else:
#             #     print("Avg score: {:.7f}".format(np.sum(score) / len(score)))
#             score[counter % len(score)] = 0
#             observation = env.reset()
#
#             if counter % batch_size == 0:
#                 print("Avg score: {:.7f}".format(np.sum(score) / len(score)))
#                 for _ in range(epochs):
#                     random.shuffle(replay_buffer)
#                     batch = np.array(replay_buffer[:batch_size])
#                     y_q = [0 for _ in range(batch_size)]
#                     y_v = [0 for _ in range(batch_size)]
#                     y_p = [0 for _ in range(batch_size)]
#                     for i, transition in enumerate(batch):
#                         y_q[i] = [transition[2] + discount_factor * (1 - transition[4]) *
#                                   forward_prop(transition[3], v_target_weights)[0]]
#                         Q1 = forward_prop(transition[0], q1_weights)[transition[1]]
#                         Q2 = forward_prop(transition[0], q2_weights)[transition[1]]
#                         action_prob = lr * forward_prop(transition[0], policy_weights)[transition[1]]
#                         y_v[i] = [min(Q1, Q2) - action_prob]
#                         y_p[i] = [Q1 - action_prob]
#                     q1_weights = Adam(batch[:, 0], y_q, q1_weights)
#                     q2_weights = Adam(batch[:, 0], y_q, q2_weights)
#                     v_weights = Adam(batch[:, 0], y_v, v_weights)
#                     policy_weights = Adam(batch[:, 0], y_p, policy_weights)
#
#                     for i in range(len(v_target_weights)):
#                         v_target_weights[i] = v_target_decay * v_target_weights[i] + (1 - v_target_decay) * \
#                                               v_weights[i]


if __name__ == "__main__":
    env_name = "BipedalWalker-v2"
    genetic_algorithm(env_name, hidden_layers=[32], render=False, batch_size=1, load_weights=False, deterministic=True)
    # vanilla_policy_gradient(env_name, tweak_param=5, hidden_layers=[128], batch_size=10, render=False, load_weights=False, deterministic=False)
    # reinforce(env_name, hidden_layers=[], render=False, load_weights=False)

    # tweaked_policy_gradient(env_name, hidden_layers=[32], render=False, batch_size=10, epochs=500, mini_batch=1000, lr=1e-3, stochastic=True, load_weights=False)
