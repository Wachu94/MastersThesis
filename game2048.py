import pygame, random, time
from figures import square
import numpy as np
from actionSpace import actionSpace


class tile(square):
    def __init__(self, x, y, value=0, a=200, offset=15):
        self.changed = False
        self.value = value
        self.color = (208, 195, 183)
        self.font = pygame.font.SysFont("AdobeGothicStd-Bold",a)
        self.text_color = (129, 116, 100)
        self.colorCheck()
        super().__init__(x, y, a, self.color, offset)
        self.x = x
        self.y = y
        self.posX = x*a
        self.posY = y*a

    def colorCheck(self):
        if self.value > 4:
            self.text_color = (255, 255, 255)
        else:
            self.text_color = (129, 116, 100)
        if self.value == 0:
            self.color = (208,195,183)
        if self.value == 2:
            self.color = (237,225,217)
        elif self.value == 4:
            self.color = (238,224,205)
        elif self.value == 8:
            self.color = (246, 180, 123)
        elif self.value == 16:
            self.color = (254, 158, 102)
        elif self.value == 32:
            self.color = (255, 138, 98)
        elif self.value == 64:
            self.color = (255, 110, 61)
        elif self.value == 128:
            self.color = (241, 205, 114)
        elif self.value == 256:
            self.color = (239, 203, 94)
        elif self.value == 512:
            self.color = (241, 198, 81)
        elif self.value == 1024:
            self.color = (241, 195, 64)
        elif self.value == 2048:
            self.color = (255, 218, 113)

    def update(self, value):
        self.value = value
        self.colorCheck()

    def draw(self, screen):
        super().draw(screen)
        if self.value > 0:
            font_name = "AdobeGothicStd-Bold"
            if self.value < 10:
                self.font_size = 200
                self.label_pos = ((self.x + 0.31) * self.a, (self.y + 0.18) * self.a)
            elif self.value < 100:
                self.font_size = 160
                self.label_pos = ((self.x + 0.20) * self.a, (self.y + 0.24) * self.a)
            elif self.value < 1000:
                self.font_size = 120
                self.label_pos = ((self.x + 0.14) * self.a, (self.y + 0.3) * self.a)
            else:
                self.font_size = 80
                self.label_pos = ((self.x + 0.18) * self.a, (self.y + 0.38) * self.a)
            self.font = pygame.font.SysFont(font_name, self.font_size)
            label = self.font.render(str(self.value), 1, self.text_color)
            screen.blit(label, self.label_pos)

class grid:
    def __init__(self, sizeX, sizeY):
        self.tiles = []
        for r in range(sizeY):
            for c in range(sizeX):
                self.tiles.append(tile(r, c))
        self.tiles = np.resize(self.tiles, (sizeY, sizeX))

    def draw(self, screen):
        for r in range(len(self.tiles)):
            for c in range(len(self.tiles[0])):
                self.tiles[r][c].draw(screen)

    def changedReset(self):
        for r in range(len(self.tiles)):
            for c in range(len(self.tiles[0])):
                self.tiles[r][c].changed = False


class env:
    RESOLUTION = (800, 800)

    def __init__(self):
        pygame.init()
        self.reward = 0
        self.tiles = 0
        self.done = False
        self.action_space = actionSpace(4)

    def getObservation(self):
        observation = []
        for y in range(4):
            observation.append([])
            for x in range(4):
                value = self.grid.tiles[x][y].value
                if value == 2048:
                    self.done = True
                observation[y].append(value)
        return observation

    def reset(self):
        self.done = False
        self.tiles = 0
        self.grid = grid(4, 4)
        self.addTile()
        return self.getObservation()

    def render(self):
        self.screen = pygame.display.set_mode(self.RESOLUTION)
        self.screen.fill((190, 176, 162))
        self.grid.draw(self.screen)
        pygame.display.flip()
        time.sleep(0.5)

    def addTile(self):
        value = 2
        if random.random() < 0.2:
            value = 4
        while True:
            x = random.randrange(0, 4)
            y = random.randrange(0, 4)
            if self.grid.tiles[x][y].value == 0:
                self.grid.tiles[x][y].update(value)
                self.tiles += 1
                break

    def step(self, dir):
        self.reward = 0
        self.grid.changedReset()
        acceptable_move = False
        if dir == 0:
            for y in range(1,4):
                for x in range(4):
                    if self.grid.tiles[x][y].value != 0:
                        temp_y = y
                        while self.grid.tiles[x][temp_y-1].value == 0 and temp_y > 0:
                            self.grid.tiles[x][temp_y-1].update(self.grid.tiles[x][temp_y].value)
                            self.grid.tiles[x][temp_y].update(0)
                            acceptable_move = True
                            temp_y -= 1
                        if self.grid.tiles[x][temp_y-1].value == self.grid.tiles[x][temp_y].value and not self.grid.tiles[x][temp_y-1].changed:
                            self.grid.tiles[x][temp_y-1].update(self.grid.tiles[x][temp_y-1].value*2)
                            self.grid.tiles[x][temp_y - 1].changed = True
                            self.grid.tiles[x][temp_y].update(0)
                            self.reward += self.grid.tiles[x][temp_y-1].value*2
                            acceptable_move = True
                            self.tiles -= 1
        elif dir == 1:
            for x in range(2,-1,-1):
                for y in range(4):
                    if self.grid.tiles[x][y].value != 0:
                        temp_x = x
                        while self.grid.tiles[temp_x + 1][y].value == 0:
                            self.grid.tiles[temp_x + 1][y].update(self.grid.tiles[temp_x][y].value)
                            self.grid.tiles[temp_x][y].update(0)
                            acceptable_move = True
                            temp_x += 1
                            if temp_x == 3:
                                break
                        if temp_x != 3:
                            if self.grid.tiles[temp_x + 1][y].value == self.grid.tiles[temp_x][y].value and not self.grid.tiles[temp_x + 1][y].changed:
                                self.grid.tiles[temp_x + 1][y].update(self.grid.tiles[temp_x + 1][y].value * 2)
                                self.grid.tiles[temp_x + 1][y].changed = True
                                self.grid.tiles[temp_x][y].update(0)
                                self.reward += self.grid.tiles[temp_x + 1][y].value * 2
                                acceptable_move = True
                                self.tiles -= 1
        elif dir == 2:
            for y in range(2,-1,-1):
                for x in range(4):
                    if self.grid.tiles[x][y].value != 0:
                        temp_y = y
                        while self.grid.tiles[x][temp_y+1].value == 0:
                            self.grid.tiles[x][temp_y+1].update(self.grid.tiles[x][temp_y].value)
                            self.grid.tiles[x][temp_y].update(0)
                            acceptable_move = True
                            temp_y += 1
                            if temp_y == 3:
                                break
                        if temp_y != 3:
                            if self.grid.tiles[x][temp_y+1].value == self.grid.tiles[x][temp_y].value and not self.grid.tiles[x][temp_y+1].changed:
                                self.grid.tiles[x][temp_y+1].update(self.grid.tiles[x][temp_y+1].value*2)
                                self.grid.tiles[x][temp_y + 1].changed = True
                                self.grid.tiles[x][temp_y].update(0)
                                self.reward += self.grid.tiles[x][temp_y+1].value*2
                                acceptable_move = True
                                self.tiles -= 1
        elif dir == 3:
            for x in range(1,4):
                for y in range(4):
                    if self.grid.tiles[x][y].value != 0:
                        temp_x = x
                        while self.grid.tiles[temp_x-1][y].value == 0 and temp_x > 0:
                            self.grid.tiles[temp_x - 1][y].update(self.grid.tiles[temp_x][y].value)
                            self.grid.tiles[temp_x][y].update(0)
                            acceptable_move = True
                            temp_x -= 1
                        if self.grid.tiles[temp_x-1][y].value == self.grid.tiles[temp_x][y].value and not self.grid.tiles[temp_x-1][y].changed:
                            self.grid.tiles[temp_x - 1][y].update(self.grid.tiles[temp_x-1][y].value*2)
                            self.grid.tiles[temp_x - 1][y].changed = True
                            self.grid.tiles[temp_x][y].update(0)
                            self.reward += self.grid.tiles[temp_x-1][y].value*2
                            acceptable_move = True
                            self.tiles -= 1
        if acceptable_move:
            self.addTile()
        else:
            # if self.tiles == 16:
            self.done = True
            self.reward = -10
        self.reward -= self.tiles
        return self.getObservation(), self.reward, self.done, None


# my_env = env()
# while True:
#     my_env.render()