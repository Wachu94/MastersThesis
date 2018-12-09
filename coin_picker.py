import pygame, sys, random
from figures import circle
from actionSpace import actionSpace
import numpy as np

class Agent(circle):
    SPEED = 0.002

    def __init__(self, x, y, r, color = (0, 0, 0)):
        super().__init__(x, y, r, color)
        self.vel = [0, 0]

    def update(self):
        self.x += self.vel[0]
        self.y += self.vel[1]

class Coin(circle):
    def __init__(self, x, y, r=5):
        super().__init__(x, y, r, (255, 255, 0))
        self.vel = [0, 0.2]

    def update(self):
        self.y += self.vel[1]


class env:
    RESOLUTION = (160,210)
    COIN_DROP_TIME = 40*60
    LIVES = 1

    action_space = actionSpace(2)
    objects = []
    agent = None
    lives = 0
    coin_timer = 0
    score = 0

    def __init__(self, pixel_observation = False):
        self.pixel_observation = pixel_observation
        self.screen = pygame.Surface(self.RESOLUTION)

    def reset(self):
        global agent
        pygame.init()
        self.objects.clear()
        agent = Agent(self.RESOLUTION[0] / 2, self.RESOLUTION[1] - 20, 5)
        self.objects.append(agent)
        self.lives = self.LIVES
        self.score = 0

        self.screen.fill((255, 255, 255))
        for i in range(len(self.objects), 0, -1):
            self.objects[i - 1].draw(self.screen)

        observation = self.step(0)[0]

        return observation
    
    def handleEvents(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_ESCAPE:
                    sys.exit()
    
        if pygame.key.get_pressed()[pygame.K_a]:
            agent.vel[0] -= agent.SPEED
        elif pygame.key.get_pressed()[pygame.K_d]:
            agent.vel[0] += agent.SPEED
    
    def step(self, action):
        global coin_timer
        done = False
        reward = 0
    
        if action == 0:
            agent.vel[0] -= agent.SPEED
        elif action == 1:
            agent.vel[0] += agent.SPEED
    
        self.coin_timer += 1
        if self.coin_timer >= self.COIN_DROP_TIME:
            self.coin_timer = 0
            self.dropCoin()
    
        for object in self.objects:
            object.update()
            if object is not agent:
                if pow(object.x - agent.x,2) + pow(object.y - agent.y,2) <= pow(agent.r,2):
                    self.objects.remove(object)
                    reward = 1
                    self.score += 1
                elif object.y > self.RESOLUTION[1] - 20:
                    self.objects.remove(object)
                    self.lives -= 1
        if agent.x < agent.r:
            agent.x = agent.r
            agent.vel[0] = 0
        elif agent.x > self.RESOLUTION[0] - agent.r:
            agent.x = self.RESOLUTION[0] - agent.r
            agent.vel[0] = 0

        if self.pixel_observation:
            observation = pygame.surfarray.array3d(self.screen)
        else:
            observation = [agent.x/160, agent.vel[0], -99, -99]
            if len(self.objects) > 1:
                observation[2] = self.objects[1].x/160
                observation[3] = self.objects[1].y/210
        if self.lives <= 0 or self.score >= 100:
            done = True
        return observation, reward, done, None
    
    def render(self):
        self.screen = pygame.display.set_mode(self.RESOLUTION)
        self.screen.fill((255, 255, 255))
        for i in range(len(self.objects), 0, -1):
            self.objects[i - 1].draw(self.screen)
        pygame.display.flip()
    
    def dropCoin(self):
        coin = Coin(random.randrange(5, self.RESOLUTION[0] - 5), -5)
        self.objects.append(coin)