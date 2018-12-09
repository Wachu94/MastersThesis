from figures import circle, line
from actionSpace import actionSpace
import pygame, sys, numpy as np
import random, time

class Env:
    screen = None
    first_player_turn = True
    no_moves = False

    def __init__(self, size):
        self.TILE_SIZE = 50
        if not size:
            size = 7
        self.BOARD_SIZE = size
        self.RESOLUTION = (self.BOARD_SIZE * self.TILE_SIZE, 6 * self.TILE_SIZE)
        self.action_space = actionSpace(self.BOARD_SIZE)

    def reset(self):
        self.board = [[0] * 6 for _ in range(self.BOARD_SIZE)]
        self.first_player_turn = True
        return self.board

    def step(self, action):
        done = False
        reward = 0
        for i in range(6):
            if self.board[action][i] == 0:
                self.board[action][i] = 1
                reward, done = self.check_status(action, i)
                break
            if i == 5:
                if self.first_player_turn:
                    return self.board, -1, True, None
                else:
                    return self.board, 1, True, None
        for r in range(self.BOARD_SIZE):
            for c in range(6):
                self.board[r][c] *= -1
        self.first_player_turn = not self.first_player_turn
        if self.first_player_turn:
            reward *= -1
        return self.board, reward, done, None

    def check_status(self, r, c):
        counter = 0
        for i in range(6):
            if self.board[r][i] == 1:
                counter += 1
                if counter == 4:
                    return 1, True
            else:
                counter = 0
        counter = 0
        for i in range(self.BOARD_SIZE):
            if self.board[i][c] == 1:
                counter += 1
                if counter == 4:
                    return 1, True
            else:
                counter = 0
        counter = 0
        for i in range(6):
            if 0 <= r - c + i < self.BOARD_SIZE:
                if self.board[r - c + i][i] == 1:
                    counter += 1
                    if counter == 4:
                        return 1, True
                else:
                    counter = 0
        counter = 0
        for i in range(6):
            if 0 <= r + c - i < self.BOARD_SIZE:
                if self.board[r + c - i][i] == 1:
                    counter += 1
                    if counter == 4:
                        return 1, True
                else:
                    counter = 0
        return 0, False

    def manual_step(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    chosen_col = event.pos[0] // self.TILE_SIZE
                    return self.step(chosen_col)


    def render(self):
        if not self.screen:
            self.screen = pygame.display.set_mode(self.RESOLUTION)
        self.screen.fill((140, 140, 255))
        for r in range(self.BOARD_SIZE):
            for c in range(6):
                if (self.first_player_turn and self.board[r][c] == -1) or (not self.first_player_turn and self.board[r][c] == 1):
                    circle((r+0.5)*self.TILE_SIZE,(5-c+0.5)*self.TILE_SIZE,20,(255,255,0)).draw(self.screen)
                elif (self.first_player_turn and self.board[r][c] == 1) or (not self.first_player_turn and self.board[r][c] == -1):
                    circle((r+0.5)*self.TILE_SIZE,(5-c+0.5)*self.TILE_SIZE,20,(255,0,0)).draw(self.screen)
                else:
                    circle((r+0.5)*self.TILE_SIZE,(5-c+0.5)*self.TILE_SIZE,20,(255,255,255)).draw(self.screen)
        pygame.display.flip()


if __name__ == "__main__":
    env = Env(None)
    env.reset()
    while True:
        env.render()
        time.sleep(0.3)
        obs, reward, done, info = env.step(env.action_space.sample())
        if done:
            print(reward)
            break
