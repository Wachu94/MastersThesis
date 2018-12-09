import pygame, random
import numpy as np
from figures import circle, square
from actionSpace import actionSpace


class grid:
    def __init__(self, sizeX, sizeY):
        self.grid = []
        for r in range(sizeY):
            for c in range(sizeX):
                self.grid.append(empty(r,c))
        self.grid = np.resize(self.grid,(sizeY,sizeX))

    def draw(self, screen):
        for r in range(len(self.grid)):
            for c in range(len(self.grid[0])):
                self.grid[r][c].draw(screen=screen)


class tile(square):
    def __init__(self, x, y, a=100, color=(255, 255, 255), offset=5):
        super().__init__(x, y, a, color, offset)
        self.terminal = False
        self.x = x
        self.y = y
        self.posX = x*a
        self.posY = y*a

    def draw(self, screen):
        super().draw(screen)


class empty(tile):
    def draw(self, screen):
        super().draw(screen)
        # pygame.gfxdraw.line(screen, self.a*self.x + self.offset, self.a*self.y + self.offset, self.a*self.x + self.a - self.offset,
        #                     self.a *self.y + self.a - self.offset, (0, 0, 0))
        # pygame.gfxdraw.line(screen, self.a*self.x + self.offset, self.a*self.y + self.a - self.offset, self.a*self.x + self.a - self.offset,
        #                     self.a *self.y + self.offset, (0, 0, 0))


class exit(tile):
    def __init__(self, x, y, id = 0):
        super().__init__(x, y)
        self.id = id

    def draw(self, screen):
        super().draw(screen)
        inner_square = square(self.x, self.y, self.a, offset=15, color=(255*(1-self.id%2),255*(self.id//3),255*(self.id%2)))
        inner_square.draw(screen)

    def update(self, env, dir):
        if env.agent.x == self.x and env.agent.y == self.y:
            if env.agent.id == self.id:
                env.agent.done = True
            return True



class bacteria(circle):
    def __init__(self,x,y):
        super().__init__(x,y,25,(150,50,255))

    def draw(self, screen):
        pygame.gfxdraw.filled_ellipse(screen, int(100*self.x+50), int(100*self.y+50), self.r, self.r, self.color)
        pygame.gfxdraw.aaellipse(screen, int(100*self.x+50), int(100*self.y+50), self.r, self.r, (0,0,0))

    def update(self, env, dir):
        for obstacle in env.obstacles:
            if obstacle.x == self.x and obstacle.y == self.y and obstacle != self:
                env.agent.reward = 15
                env.obstacles.remove(self)
                return False

        if env.agent.x == self.x and env.agent.y == self.y:
            env.agent.reward = 15
            env.obstacles.remove(self)
            return True

class wall(tile):
    def __init__(self, x, y):
        super().__init__(x, y)

    def draw(self, screen):
        super().draw(screen)
        inner_square = square(self.x, self.y, self.a, offset=15, color=(0,0,0))
        inner_square.draw(screen)

    def update(self, env, dir):
        if env.agent.x == self.x and env.agent.y == self.y:
            if dir == 0:
                env.agent.y += 1
            elif dir == 1:
                env.agent.x -= 1
            elif dir == 2:
                env.agent.y -= 1
            elif dir == 3:
                env.agent.x += 1
            return True


class movable(circle):
    def __init__(self,x,y):
        super().__init__(x,y,25,(255,255,0))

    def draw(self, screen):
        pygame.gfxdraw.filled_ellipse(screen, int(100*self.x+50), int(100*self.y+50), self.r, self.r, self.color)
        pygame.gfxdraw.aaellipse(screen, int(100*self.x+50), int(100*self.y+50), self.r, self.r, (0,0,0))

    def update(self, env, dir):
        for obstacle in env.obstacles:
            if obstacle != self:
                if obstacle.x == self.x and obstacle.y == self.y:
                    if type(obstacle) == type(self) or isinstance(obstacle, wall) or isinstance(obstacle, exit):
                        if dir == 0:
                            self.y += 1
                            env.agent.y += 2
                        elif dir == 1:
                            self.x -= 1
                            env.agent.x -= 2
                        elif dir == 2:
                            self.y -= 1
                            env.agent.y -= 2
                        elif dir == 3:
                            self.x += 1
                            env.agent.x += 2
                    else:
                        obstacle.update(env, dir)
        if env.agent.x == self.x and env.agent.y == self.y:
            if dir == 0:
                if env.agent.startPos[1] != self.y + 1:
                    self.y -= 1
                else:
                    env.agent.y += 1
            elif dir == 1:
                if env.agent.startPos[0] != self.x - 1:
                    self.x += 1
                else:
                    env.agent.x -= 1
            elif dir == 2:
                if env.agent.startPos[1] != self.y - 1:
                    self.y += 1
                else:
                    env.agent.y -= 1
            elif dir == 3:
                if env.agent.startPos[0] != self.x + 1:
                    self.x -= 1
                else:
                    env.agent.x += 1
            return True


class trap(tile):
    def __init__(self, x, y, hole = False):
        super().__init__(x, y, color=(150,150,150))
        self.hole = hole

    def draw(self, screen):
        super().draw(screen)
        # pygame.gfxdraw.line(screen, self.a*self.x + self.offset, self.a*self.y + self.offset, self.a*self.x + self.a - self.offset,
        #                     self.a *self.y + self.a - self.offset, (0, 0, 0))
        # pygame.gfxdraw.line(screen, self.a*self.x + self.offset, self.a*self.y + self.a - self.offset, self.a*self.x + self.a - self.offset,
        #                     self.a *self.y + self.offset, (0, 0, 0))

    def update(self, env, dir):
        if env.agent.x == self.x and env.agent.y == self.y:
            if self.hole:
                env.die()
                return True
            else:
                self.hole = True
                self.color = (0,0,0)
                return False


class slider(tile):
    def __init__(self, x, y, dir):
        super().__init__(x, y, color=(200,240,255))
        self.dir = dir

    def draw(self, screen):
        super().draw(screen)
        if self.dir % 2 == 0:
            pygame.gfxdraw.line(screen, int(self.a * (self.x + 0.5)), int(self.a * (self.y + 0.2)), int(self.a * (self.x + 0.5)), int(self.a * (self.y + 0.8)), (0, 0, 0))
            dir_coeff = 0.2
            if self.dir == 2:
                dir_coeff = 0.8
            pygame.gfxdraw.line(screen, int(self.a * (self.x + 0.2)), int(self.a * (self.y + 0.5)), int(self.a * (self.x + 0.5)), int(self.a * (self.y + dir_coeff)), (0, 0, 0))
            pygame.gfxdraw.line(screen, int(self.a * (self.x + 0.5)), int(self.a * (self.y + dir_coeff)), int(self.a * (self.x + 0.8)), int(self.a * (self.y + 0.5)), (0, 0, 0))
        else:
            pygame.gfxdraw.line(screen, int(self.a * (self.x + 0.2)), int(self.a * (self.y + 0.5)), int(self.a * (self.x + 0.8)), int(self.a * (self.y + 0.5)), (0, 0, 0))
            dir_coeff = 0.2
            if self.dir == 1:
                dir_coeff = 0.8
            pygame.gfxdraw.line(screen, int(self.a * (self.x + 0.5)), int(self.a * (self.y + 0.2)), int(self.a * (self.x + dir_coeff)), int(self.a * (self.y + 0.5)), (0, 0, 0))
            pygame.gfxdraw.line(screen, int(self.a * (self.x + dir_coeff)), int(self.a * (self.y + 0.5)), int(self.a * (self.x + 0.5)), int(self.a * (self.y + 0.8)), (0, 0, 0))

    def update(self, env, dir):
        for agent in env.agents:
            if agent.x == self.x and agent.y == self.y:
                if agent is env.agent:
                    if dir == (self.dir + 2) % 4:
                        if dir == 0:
                            env.agent.y += 1
                        elif dir == 1:
                            env.agent.x -= 1
                        elif dir == 2:
                            env.agent.y -= 1
                        elif dir == 3:
                            env.agent.x += 1
                    else:
                        env.counter += 1
                        if env.counter > 50:
                            env.die()
                            return True
                        env.move(self.dir)
                    return True
                else:
                    if self.dir == 0 and not (env.agent.x == agent.x and env.agent.y == agent.y - 1):
                        agent.y -= 1
                    elif self.dir == 1 and not (env.agent.x == agent.x + 1 and env.agent.y == agent.y):
                        agent.x += 1
                    elif self.dir == 2  and not (env.agent.x == agent.x and env.agent.y == agent.y + 1):
                        agent.y += 1
                    elif self.dir == 3 and not (env.agent.x == agent.x - 1 and env.agent.y == agent.y):
                        agent.x -= 1

class hidden(circle):
    def __init__(self,x,y):
        self.visible = False
        super().__init__(x,y,25,(100,150,255))

    def draw(self, screen):
        if self.visible == True:
            pygame.gfxdraw.filled_ellipse(screen, int(100*self.x+50), int(100*self.y+50), self.r, self.r, self.color)
            pygame.gfxdraw.aaellipse(screen, int(100*self.x+50), int(100*self.y+50), self.r, self.r, (0, 0, 0))
            pygame.gfxdraw.filled_ellipse(screen, int(100 * self.x + 50), int(100 * self.y + 50), 20, 20,(255, 255, 255))
            pygame.gfxdraw.aaellipse(screen, int(100 * self.x + 50), int(100 * self.y + 50), 20, 20, (0, 0, 0))

    def update(self, env):
        if env.agent.x == self.x and env.agent.y == self.y:
            if self.visible == False:
                env.agent.reward += 10
            self.visible = True
            return False


class agent(circle):
    def __init__(self,x,y,id=0):
        self.id = id
        self.done = False
        self.reward = 0
        self.startPos = (x,y)
        super().__init__(x,y,20,(255*(1-id%2),255*(id//3),255*(id%2)))

    def draw(self, screen):
        pygame.gfxdraw.filled_ellipse(screen, int(100*self.x+50), int(100*self.y+50), self.r, self.r, self.color)
        pygame.gfxdraw.aaellipse(screen, int(100*self.x+50), int(100*self.y+50), self.r, self.r, (0,0,0))

    def update(self, env, dir):
        if not self is env.agent:
            # for obstacle in env.obstacles:
            #     if obstacle != self:
            #         if obstacle.x == self.x and obstacle.y == self.y:
            #             if isinstance(obstacle, movable) or isinstance(obstacle, wall) or isinstance(obstacle, exit):
            #                 if dir == 0:
            #                     self.y += 1
            #                     env.agent.y += 2
            #                 elif dir == 1:
            #                     self.x -= 1
            #                     env.agent.x -= 2
            #                 elif dir == 2:
            #                     self.y -= 1
            #                     env.agent.y -= 2
            #                 elif dir == 3:
            #                     self.x += 1
            #                     env.agent.x += 2
            #             else:
            #                 obstacle.update(env, dir)
            if env.agent.x == self.x and env.agent.y == self.y:
                if dir == 0:
                    env.agent.y += 1
                elif dir == 1:
                    env.agent.x -= 1
                elif dir == 2:
                    env.agent.y -= 1
                elif dir == 3:
                    env.agent.x += 1
                return True


class environment:
    RESOLUTION = (700, 900)

    def __init__(self, map, agent_pos):
        self.done = False
        self.action_space = actionSpace(2*len(agent_pos))
        self.agent_pos = agent_pos
        self.map = map
        self.hidden = None
        self.grid = grid(len(map), len(map[0]))
        self.agents = []
        for i in range(len(agent_pos)//2):
            self.agents.append(agent(agent_pos[2*i], agent_pos[2*i+1], i))
        self.agent = self.agents[0]
        self.counter = 0
        self.discount = 0.999

    def setupMap(self, map):
        obstacles = []
        for r in range(len(map)):
            for c in range(len(map[0])):
                if map[r][c] == 'b':
                    obstacles.append(bacteria(c, r))
                elif map[r][c] == 'x' or map[r][c] == 'x0':
                    obstacles.append(exit(c, r))
                elif map[r][c] == 'x1':
                    obstacles.append(exit(c, r, 1))
                elif map[r][c] == 'x2':
                    obstacles.append(exit(c, r, 2))
                elif map[r][c] == 'x3':
                    obstacles.append(exit(c, r, 3))
                elif map[r][c] == 'x4':
                    obstacles.append(exit(c, r, 4))
                elif map[r][c] == 'o':
                    obstacles.append(wall(c, r))
                elif map[r][c] == 'm':
                    obstacles.append(movable(c, r))
                elif map[r][c] == 't':
                    obstacles.append(trap(c, r))
                elif map[r][c] == 'tt':
                    obstacles.append(trap(c, r, True))
                elif map[r][c] == 'h':
                    self.hidden = hidden(c, r)
                elif map[r][c] == 's0':
                    obstacles.append(slider(c, r, 0))
                elif map[r][c] == 's1':
                    obstacles.append(slider(c, r, 1))
                elif map[r][c] == 's2':
                    obstacles.append(slider(c, r, 2))
                elif map[r][c] == 's3':
                    obstacles.append(slider(c, r, 3))
        return obstacles

    def reset(self):
        pygame.init()
        for i in range(len(self.agent_pos)//2):
            self.agents[i].x = self.agent_pos[2*i]
            self.agents[i].y = self.agent_pos[2*i+1]
        observation = []
        for agent in self.agents:
            observation.append(agent.x / len(self.map[0]))
            observation.append(agent.y / len(self.map[1]))
            agent.reward = 0
            agent.done = False
        self.obstacles = self.setupMap(self.map)
        self.done = False
        self.counter = 0
        return observation

    def render(self):
        self.screen = pygame.display.set_mode(self.RESOLUTION)
        self.screen.fill((255, 255, 255))
        self.grid.draw(self.screen)
        if self.hidden:
            self.hidden.draw(self.screen)
        for i in range(len(self.obstacles), 0, -1):
            self.obstacles[i - 1].draw(self.screen)
        for agent in self.agents:
            agent.draw(self.screen)
        pygame.display.flip()

    def step(self, input):
        observation = []
        for agent in self.agents:
            observation.append(agent.x/len(self.map[0]))
            observation.append(agent.y/len(self.map[1]))
        self.counter += 1
        if self.counter >= 50:
            self.die()
        self.agent.reward = 0
        self.agent = self.agents[input//4]
        dir = input
        while dir > 3:
            dir = dir - 4
        self.agent.reward = 0
        self.agent.done = False
        self.agent.startPos = (self.agent.x, self.agent.y)
        self.move(dir)

        if self.agent.reward > 0:
            self.agent.reward = self.agent.reward * (self.discount ** self.counter)
        return observation, self.agent.reward, self.done, None

    def move(self, dir):
        if dir == 0:
            while self.agent.y >= 0:
                if self.agent.y == 0:
                    self.die()
                    return
                self.agent.y -= 1
                if self.update(0) == True:
                    break
        elif dir == 1:
            while self.agent.x <= len(self.map[0]) - 1:
                if self.agent.x == len(self.map[0]) - 1:
                    self.die()
                    return
                self.agent.x += 1
                if self.update(1) == True:
                    break
        elif dir == 2:
            while self.agent.y <= len(self.map) - 1:
                if self.agent.y == len(self.map) - 1:
                    self.die()
                    return
                self.agent.y += 1
                if self.update(2) == True:
                    break
        elif dir == 3:
            while self.agent.x >= 0:
                if self.agent.x == 0:
                    self.die()
                    return
                self.agent.x -= 1
                if self.update(3) == True:
                    break
        self.done = self.doneCheck()

    def update(self, dir):
        if self.hidden:
            self.hidden.update(self)
        for agent in self.agents:
            if not agent is self.agent:
                if agent.update(self, dir) == True:
                    return True
            else:
                if agent.x == -1 or agent.x == 7 or agent.y == -1 or agent.y == 9:
                    return True
        for obstacle in self.obstacles:
            if obstacle.update(self, dir) == True:
                return True

    def doneCheck(self):
        for agent in self.agents:
            if not agent.done:
                return False
        self.agent.reward = 5 * len(self.agents)
        return True

    def die(self):
        self.agent.x = -1
        self.agent.y = -1
        self.agent.reward = -100
        self.done = True


map_1 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'b', 'e', 'x', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_2 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'x', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'o', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'b', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'h', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'o', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_3 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'b', 'e', 'e', 'o', 'e'],
        ['e', 'e', 'e', 'e', 'h', 'e', 'e'],
        ['e', 'o', 'e', 'b', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'o', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'x', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_4 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'x', 'e', 'e', 'e'],
        ['e', 'o', 'e', 'e', 'e', 'o', 'e'],
        ['e', 'h', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'b', 'e', 'm', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'b', 'e', 'e', 'e', 'b', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_5 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'o', 'e', 'e'],
        ['o', 'e', 'e', 'e', 'e', 'm', 'h'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'b', 'e', 'm', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'm', 'e', 'e', 'e', 'x', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_6 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'o', 'e'],
        ['e', 'o', 'e', 'b', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 't', 'e', 'b', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'o', 'e'],
        ['o', 'h', 'e', 'b', 'e', 'x', 'e'],
        ['e', 'e', 'o', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_7 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'o', 't', 'b', 'h', 'm', 'e'],
        ['e', 't', 'e', 'e', 'e', 'b', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'x', 'b', 'e', 't', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'o', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_8 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'm', 'e', 'b', 'e', 'm', 'e'],
        ['e', 'h', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'm', 'e', 'x', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'm', 'e', 'b', 'e', 'm', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_9 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'm', 'e', 'e', 'm'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['m', 'e', 'e', 'e', 'e', 'e', 'b'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'h', 'e', 'e'],
        ['b', 'e', 'e', 'm', 'm', 'e', 'x'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_10 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'b', 't', 'b', 't', 'b', 'e'],
        ['e', 't', 'e', 'e', 'e', 't', 'e'],
        ['e', 'b', 'h', 'b', 'e', 'b', 'e'],
        ['e', 't', 'e', 'e', 'e', 't', 'e'],
        ['e', 'b', 't', 'b', 't', 'x', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_11 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'h', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'm', 'b', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'b', 'm', 'e', 'm', 'b', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'b', 'e', 'm', 'e', 'o', 'e'],
        ['e', 'e', 'e', 'x', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_12 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'm', 'e', 'b', 'h', 'm', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'b', 'e', 'e', 'e', 'b', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'm', 'e', 'b', 'e', 'm', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'm', 'e', 'x', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_13 =[['b', 'b', 'e', 'm', 'e', 'e', 'm'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'm', 'e', 'e', 'e', 'b', 'm'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'h'],
        ['e', 'e', 'e', 'e', 'e', 'm', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'x', 'e', 'e', 'e']]

map_14 =[['e', 'e', 'e', 'e', 'e', 'e', 'b'],
        ['e', 'e', 'b', 'e', 'e', 'e', 'e'],
        ['m', 'e', 'e', 'e', 'x', 'e', 'e'],
        ['e', 'o', 'e', 'e', 'e', 'o', 'e'],
        ['e', 'e', 'm', 'e', 'e', 'm', 'e'],
        ['e', 'e', 'o', 'e', 'e', 'e', 'o'],
        ['e', 'e', 'e', 'e', 'e', 'b', 'e'],
        ['h', 'b', 'e', 'e', 'e', 'm', 'e'],
        ['o', 'e', 'e', 'e', 'e', 'e', 'e']]

map_15 =[['e', 'b', 'e', 'x', 'e', 'm', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'm', 'e', 'm', 'e', 'b', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'b', 'e', 'm', 'e', 'm', 'e'],
        ['e', 'e', 'e', 'h', 'e', 'e', 'e'],
        ['e', 'b', 'e', 'm', 'e', 'm', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_16 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'o', 'e', 'h', 'e'],
        ['e', 'b', 'e', 't', 'e', 'e', 'b'],
        ['e', 'e', 'e', 'e', 'e', 'm', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'm', 'e', 'e', 'e', 'm', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'o', 'e'],
        ['e', 'b', 'e', 'e', 'e', 'e', 'x']]

map_17 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'b', 'o', 'o', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'b', 'e', 'o', 'e'],
        ['e', 'e', 'b', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'o', 'e', 'e', 'e', 'e'],
        ['o', 'e', 'e', 'm', 'e', 'e', 'x'],
        ['e', 'e', 'e', 'e', 'e', 'b', 'e'],
        ['e', 'o', 'o', 'h', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'o', 'e', 'e', 'e']]

map_18 =[['e', 'e', 'e', 'e', 'o', 'e', 'e'],
        ['e', 'o', 'h', 'e', 'e', 'e', 'x'],
        ['e', 'e', 'b', 'e', 'b', 'e', 'o'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['o', 'e', 'm', 'm', 'm', 'm', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'b', 'e', 'b', 'o'],
        ['e', 'o', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'o', 'o', 'e', 'e']]

map_19 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'b', 'e', 'e', 'b', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'x', 'e', 'b', 'e'],
        ['e', 'o', 'e', 'b', 'm', 'e', 'e'],
        ['e', 'e', 'h', 'e', 'e', 'm', 'e'],
        ['e', 'e', 'e', 'b', 'e', 'm', 'o'],
        ['e', 'e', 'e', 'e', 'o', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_20 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'o', 'x', 'e', 'e', 'e'],
        ['e', 'b', 'e', 'e', 'm', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['o', 'e', 'e', 'e', 'e', 'm', 'e'],
        ['e', 'o', 'e', 'h', 'e', 'b', 'o'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_21 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 's1', 'e', 'e', 'b', 'o', 'e'],
        ['e', 'e', 'e', 'x', 's2', 'h', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 's0', 'e', 'b', 'e'],
        ['e', 'b', 'e', 'o', 'o', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_22 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 's2', 'e', 'e', 'e'],
        ['e', 'b', 'e', 'e', 'e', 'b', 'e'],
        ['e', 'e', 'e', 'h', 'e', 'e', 'e'],
        ['e', 's2', 'e', 'e', 'e', 's0', 'e'],
        ['e', 'e', 'x', 'e', 'e', 'e', 'e'],
        ['e', 'b', 'e', 'e', 'e', 'b', 'e'],
        ['e', 'e', 'e', 's0', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_23 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 's1', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'b', 'e', 'x0', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'x1', 'h', 's3', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_24 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'b', 'e', 'e', 'e'],
        ['e', 'b', 'b', 'e', 'e', 'h', 'm'],
        ['s2', 'e', 's3', 'e', 's2', 'e', 's3'],
        ['e', 'e', 'e', 's0', 't', 'e', 'e'],
        ['e', 't', 's0', 'e', 's3', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 's3', 'e'],
        ['o', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'x', 'e', 'e']]

map_25 =[['s1', 's2', 's2', 't', 't', 't', 's3'],
        ['s0', 'h', 'e', 'x', 's3', 's1', 's0'],
        ['s0', 's2', 's1', 's1', 's0', 's0', 's3'],
        ['s0', 's2', 's1', 's1', 's2', 'b', 's0'],
        ['s0', 's1', 'e', 's2', 's1', 'm', 's0'],
        ['s0', 's3', 'e', 'e', 's3', 'e', 'tt'],
        ['s2', 's3', 'e', 'e', 'e', 'e', 's3'],
        ['s1', 'e', 't', 's0', 's2', 'e', 's3'],
        ['b', 'e', 'e', 's3', 's3', 's1', 's0']]

map_26 =[['e', 'e', 'e', 's0', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'o', 'o', 'e', 'o', 'o', 'e'],
        ['x0', 'e', 'e', 'e', 'e', 'e', 'x1'],
        ['e', 'b', 'e', 's0', 'e', 'b', 'e'],
        ['e', 'e', 'b', 'e', 'b', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 's0', 'e', 'e', 'e']]

map_27 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 's2', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'b', 'e', 'x1', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'b', 's1', 'e', 's3', 'b', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'b', 'e', 'x0', 'e'],
        ['e', 'e', 'e', 's0', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_28 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 's2', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'o'],
        ['b', 'e', 'm', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'x1', 'e'],
        ['e', 'e', 'e', 's1', 'e', 'e', 'b'],
        ['e', 'e', 'o', 'e', 'x0', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_29 =[['x0', 'b', 'e', 'b', 'e', 'b', 'x1'],
        ['s1', 'e', 'e', 'm', 'e', 'e', 's3'],
        ['e', 'e', 's1', 'e', 's3', 'e', 'e'],
        ['b', 'e', 'e', 'e', 'e', 'e', 'b'],
        ['e', 'e', 's1', 'e', 's3', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'b', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'o', 'e', 'e', 'e']]

map_30 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 's2', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'o', 'e'],
        ['e', 'e', 'e', 'e', 'x1', 'o', 'e'],
        ['e', 'e', 'tt', 's2', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'x0', 'e', 'b', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_31 =[['e', 'e', 'e', 'o', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 's1', 'e', 'e', 'e', 's2', 'e'],
        ['e', 'e', 'e', 'b', 'e', 'e', 'e'],
        ['o', 'e', 'e', 'e', 't', 'm', 'o'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 's0', 'e', 'e', 'e', 's3', 'e'],
        ['e', 'e', 'e', 'b', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'x1', 'x0', 'e']]

map_32 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'o', 's2', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'b', 'e', 'e', 'x1', 'e'],
        ['e', 'e', 's1', 'e', 's3', 'e', 'e'],
        ['e', 'x0', 'e', 'e', 'b', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'b', 'e', 'e'],
        ['e', 'e', 'e', 's0', 'o', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_33 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'b', 'e', 'e', 'e'],
        ['e', 's1', 'm', 'e', 'e', 'b', 'e'],
        ['e', 'e', 'e', 'm', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'x', 'b', 'm', 'e', 'b', 'e'],
        ['e', 'e', 'e', 's0', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_34 =[['s1', 's1', 's2', 'x', 's3', 's3', '3s'],
        ['e', 'e', 's1', 'e', 's2', 'h', 'e'],
        ['b', 's3', 'e', 's2', 'e', 's0', 'b'],
        ['s2', 'e', 's3', 'e', 's1', 'e', 's0'],
        ['e', 's0', 'e', 'e', 'e', 's2', 'e'],
        ['s1', 'b', 's0', 'e', 's2', 'b', 's3'],
        ['s2', 'e', 'b', 's0', 'b', 'e', 's0'],
        ['e', 's2', 's0', 'e', 's3', 's3', 'e'],
        ['s1', 'e', 's1', 'b', 's1', 'e', 's0']]

map_35 =[['e', 'o', 'e', 'x', 'o', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'o', 'e'],
        ['s1', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['o', 'e', 'e', 'm', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'b'],
        ['e', 'e', 'e', 'e', 's3', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'h', 'e', 'o'],
        ['e', 'm', 'e', 's0', 'e', 'e', 'e']]

map_36 =[['s2', 'e', 'e', 'e', 'x0', 'e', 's3'],
        ['e', 'e', 'e', 'e', 'e', 'e', 's0'],
        ['h', 'o', 'e', 's2', 'e', 'e', 's3'],
        ['m', 'e', 'e', 'e', 'e', 'm', 'e'],
        ['e', 'e', 'e', 'e', 's0', 'e', 'o'],
        ['e', 'e', 'e', 'e', 'b', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'x1', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'o', 'e']]

map_37 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'tt', 'e', 'x', 'e'],
        ['e', 's1', 'e', 'e', 'e', 's3', 'e'],
        ['e', 'e', 'e', 'e', 'e', 's2', 'e'],
        ['e', 's0', 'm', 'e', 'e', 's3', 'e'],
        ['e', 'e', 's0', 'e', 'b', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'b', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]

map_38 =[['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e'],
        ['e', 'e', 'e', 'e', 'e', 'e', 'e']]


def chooseLevel(lvl):
    if lvl == 2:
        return environment(map_2, (2, 6))
    elif lvl == 3:
        return environment(map_3, (1, 5))
    elif lvl == 4:
        return environment(map_4, (3, 7))
    elif lvl == 5:
        return environment(map_5, (3, 4))
    elif lvl == 6:
        return environment(map_6, (1, 4))
    elif lvl == 7:
        return environment(map_7, (1, 6))
    elif lvl == 8:
        return environment(map_8, (1, 4))
    elif lvl == 9:
        return environment(map_9, (0, 1))
    elif lvl == 10:
        return environment(map_10, (3, 1))
    elif lvl == 11:
        return environment(map_11, (3, 5))
    elif lvl == 12:
        return environment(map_12, (1, 7))
    elif lvl == 13:
        return environment(map_13, (1, 6))
    elif lvl == 14:
        return environment(map_14, (3, 4))
    elif lvl == 15:
        return environment(map_15, (3, 8))
    elif lvl == 16:
        return environment(map_16, (3, 6))
    elif lvl == 17:
        return environment(map_17, (2, 5))
    elif lvl == 18:
        return environment(map_18, (1, 4))
    elif lvl == 19:
        return environment(map_19, (1, 6))
    elif lvl == 20:
        return environment(map_20, (2, 6))
    elif lvl == 21:
        return environment(map_21, (1, 4))
    elif lvl == 22:
        return environment(map_22, (3, 4))
    elif lvl == 23:
        return environment(map_23, (1, 6, 5, 2))
    elif lvl == 24:
        return environment(map_24, (6, 7))
    elif lvl == 25:
        return environment(map_25, (3, 6))
    elif lvl == 26:
        return environment(map_26, (3, 6, 3, 2))
    elif lvl == 27:
        return environment(map_27, (1, 2, 1, 6))
    elif lvl == 28:
        return environment(map_28, (6, 2, 0, 2))
    elif lvl == 29:
        return environment(map_29, (4, 6, 2, 6))
    elif lvl == 30:
        return environment(map_30, (1, 5, 3, 2))
    elif lvl == 31:
        return environment(map_31, (6, 6, 0, 2))
    elif lvl == 32:
        return environment(map_32, (4, 1, 2, 7))
    elif lvl == 33:
        return environment(map_33, (3, 4))
    elif lvl == 34:
        return environment(map_34, (3, 4))
    elif lvl == 35:
        return environment(map_35, (2, 4))
    elif lvl == 36:
        return environment(map_36, (0, 6, 1, 6))
    elif lvl == 37:
        return environment(map_37, (1, 2))
    else:
        return environment(map_1, (1, 4))


# my_env = chooseLevel(28)
# my_env.reset()
# my_env.step(6)
# my_env.step(5)
# my_env.step(3)
# my_env.step(6)
# my_env.step(5)
# my_env.step(0)
# my_env.step(1)
# my_env.step(2)
# my_env.step(3)
# my_env.step(2)
# my_env.step(3)
# my_env.step(7)
# my_env.step(2)
# my_env.step(4)
# while True:
#     my_env.render()