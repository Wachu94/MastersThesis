import gym, pygame, time
from gym import spaces
from gym.utils import seeding
from figures import square, circle
import numpy as np

class GridWorldEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, tweak_param=1):
        self.elements = {"player": -1, "empty": 0, "exit": 1, "bacteria": 2, "wall": 3, "movable": 4}
        self.rows = 9
        self.cols = 7

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(self.rows * self.cols)

        self.seed()
        self.screen = None
        self.state = None
        self.player_pos = [0, 0]
        self.step_counter = 0
        self.level = tweak_param

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        self.step_counter += 1
        if self.step_counter == 50:
            return self.state, -1, True, None
        reward = 0
        momentum = 0
        dir = {0: [-1, 0], 1: [0, 1], 2: [1, 0], 3: [0, -1]}
        if self.screen:
            time.sleep(0.2)
        self.state[self.player_pos[0], self.player_pos[1]] = self.elements["empty"]
        self.player_pos = np.add(self.player_pos, dir[action])
        try:
            while self.state[self.player_pos[0], self.player_pos[1]] == 0:
                if self.player_pos[0] < 0 or self.player_pos[1] < 0:
                    return self.state, -1, True, None
                if self.screen:
                    self.state[self.player_pos[0], self.player_pos[1]] = self.elements["player"]
                    self.render()
                    self.state[self.player_pos[0], self.player_pos[1]] = self.elements["empty"]
                    time.sleep(0.2)
                self.player_pos = np.add(self.player_pos, dir[action])
                momentum += 1
        except:
            return self.state, -1, True, None
        object = self.state[self.player_pos[0], self.player_pos[1]]
        if object == self.elements["exit"]:
            return self.state, 10, True, None
        elif object == self.elements["bacteria"]:
            reward = 1
        elif object == self.elements["wall"]:
            self.player_pos = np.subtract(self.player_pos, dir[action])
        elif object == self.elements["movable"]:
            next_tile = np.add(self.player_pos, dir[action])
            try:
                if momentum == 0 or self.state[next_tile[0], next_tile[1]] != self.elements["empty"]:
                    self.player_pos = np.subtract(self.player_pos, dir[action])
                else:
                    self.state[next_tile[0], next_tile[1]] = self.elements["movable"]
            except:
                pass
        self.state[self.player_pos[0], self.player_pos[1]] = self.elements["player"]
        return self.state, reward, False, None


    def reset(self):
        self.setup_map()
        self.step_counter = 0
        self.steps_beyond_done = None
        return np.array(self.state)

    # def render(self, mode='human'):
    #     screen_width = 700
    #     screen_height = 900
    #     if self.viewer is None:
    #         from gym.envs.classic_control import rendering
    #         self.viewer = rendering.Viewer(screen_width, screen_height)
    #         for r in range(self.rows):
    #             for c in range(self.cols):
    #                 tile = rendering.FilledPolygon([(100*c,screen_height - 100*r), (100*(c + 1), screen_height - 100*r), (100*(c + 1), screen_height - 100*(r + 1)), (100*c, screen_height - 100*(r + 1))])
    #                 tile.set_color(1, 1, 1)
    #                 self.viewer.add_geom(tile)
    #                 object = rendering.make_circle(25)
    #                 object.add_attr(rendering.Transform((100 * (c + 0.5), screen_height - 100 * (r + 0.5))))
    #                 if self.state[r, c] == self.elements["player"]:
    #                     object.set_color(0, 0, 1)
    #                     self.viewer.add_geom(object)
    #                 elif self.state[r, c] == self.elements["exit"]:
    #                     tile.set_color(0, 0, 1)
    #                 elif self.state[r, c] == self.elements["bacteria"]:
    #                     object.set_color(.5, 0, .5)
    #                     self.viewer.add_geom(object)
    #                 elif self.state[r, c] == self.elements["wall"]:
    #                     tile.set_color(0, 0, 0)
    #                 elif self.state[r, c] == self.elements["movable"]:
    #                     object.set_color(.9, .9, 0)
    #                     self.viewer.add_geom(object)
    #                 tile = rendering.PolyLine(
    #                     [(100 * c, screen_height - 100 * r), (100 * (c + 1), screen_height - 100 * r),
    #                      (100 * (c + 1), screen_height - 100 * (r + 1)),
    #                      (100 * c, screen_height - 100 * (r + 1))], True)
    #                 tile.set_color(0, 0, 0)
    #                 self.viewer.add_geom(tile)
    #     if self.state is None: return None
    #
    #     return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render(self, mode='human'):
        screen_width = 700
        screen_height = 900
        if not self.screen:
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        self.screen.fill((255, 255, 255))
        for r in range(self.rows):
            for c in range(self.cols):
                tile = square(c, r, 100, offset=2)
                draw_object = False
                object = circle(100*(c+0.5), 100*(r+0.5), 25)
                if self.state[r, c] == self.elements["player"]:
                    object.color = (255, 0, 0)
                    draw_object = True
                elif self.state[r, c] == self.elements["exit"]:
                    object = square(c, r, 100, color=(255, 0, 0), offset=10)
                    draw_object = True
                elif self.state[r, c] == self.elements["bacteria"]:
                    object.color = (155, 0, 155)
                    draw_object = True
                elif self.state[r, c] == self.elements["wall"]:
                    object = square(c, r, 100, color=(0, 0, 0), offset=10)
                    draw_object = True
                elif self.state[r, c] == self.elements["movable"]:
                    object.color = (255, 255, 0)
                    draw_object = True
                tile.draw(self.screen)
                if draw_object:
                    object.draw(self.screen)
        pygame.display.flip()
        for _ in pygame.event.get():
            pass

    def setup_map(self):
        elements = self.elements
        self.state = np.zeros((self.rows, self.cols))
        if self.level == 1:
            self.player_pos = [4, 1]
            self.state[4, 5] = elements["exit"]
            self.state[4, 3] = elements["bacteria"]
        elif self.level == 2:
            self.player_pos = [6, 2]
            self.state[1, 4] = elements["exit"]
            self.state[4, 4] = elements["bacteria"]
            self.state[7, 4] = elements["wall"]
            self.state[3, 2] = elements["wall"]
        elif self.level == 3:
            self.player_pos = [5, 1]
            self.state[7, 2] = elements["exit"]
            self.state[2, 2] = elements["bacteria"]
            self.state[4, 3] = elements["bacteria"]
            self.state[2, 5] = elements["wall"]
            self.state[4, 1] = elements["wall"]
            self.state[5, 4] = elements["wall"]
        elif self.level == 4:
            self.player_pos = [7, 3]
            self.state[1, 3] = elements["exit"]
            self.state[4, 1] = elements["bacteria"]
            self.state[6, 1] = elements["bacteria"]
            self.state[6, 5] = elements["bacteria"]
            self.state[2, 1] = elements["wall"]
            self.state[2, 5] = elements["wall"]
            self.state[4, 3] = elements["movable"]
        elif self.level == 5:
            self.player_pos = [4, 3]
            self.state[6, 5] = elements["exit"]
            self.state[4, 2] = elements["bacteria"]
            self.state[1, 4] = elements["wall"]
            self.state[2, 0] = elements["wall"]
            self.state[2, 5] = elements["movable"]
            self.state[4, 4] = elements["movable"]
            self.state[6, 1] = elements["movable"]
        elif self.level == 6:
            self.player_pos = [4, 3]
        self.state[self.player_pos[0], self.player_pos[1]] = elements["player"]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

if __name__ == "__main__":
    level = 4
    env = GridWorldEnv(level + 1)
    actions = [[1],
               [0, 1],
               [1, 0, 3, 0, 2],
               [0, 3, 2, 1, 0, 3],
               [3, 1, 0, 3, 2, 1]]
    for _ in range(1000):
        observation = env.reset()
        score = 0
        counter = 0
        while True:
            env.render()
            # action = env.action_space.sample()
            action = actions[level][counter % len(actions[level])]
            observation, reward, done, _ = env.step(action)
            score += reward
            counter += 1
            if done:
                break
    env.close()