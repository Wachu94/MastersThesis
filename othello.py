from figures import circle, line
import pygame, sys, numpy as np
import random, time

class actionSpace:
    def __init__(self, size):
        self.size = size
    def sample(self):
        return [[random.random() for _ in range(self.size)] for _ in range(self.size)]

class Env:
    screen = None
    available_places = set()
    first_player_turn = True
    no_moves = False

    def __init__(self, size):
        self.TILE_SIZE = 50
        if not size:
            size = 8
        self.BOARD_SIZE = size
        self.RESOLUTION = (self.BOARD_SIZE * self.TILE_SIZE, self.BOARD_SIZE * self.TILE_SIZE)
        self.action_space = actionSpace(self.BOARD_SIZE)

    def reset(self):
        self.first_player_turn = True
        self.board = [[0] * (self.BOARD_SIZE + 1) for _ in range(self.BOARD_SIZE + 1)]
        self.board[self.BOARD_SIZE//2 - 1][self.BOARD_SIZE//2 - 1] = -1
        self.board[self.BOARD_SIZE//2][self.BOARD_SIZE//2] = -1
        self.board[self.BOARD_SIZE//2 - 1][self.BOARD_SIZE//2] = 1
        self.board[self.BOARD_SIZE//2][self.BOARD_SIZE//2 - 1] = 1
        return self.board

    def step(self, action):
        # try:
        #     assert len(action) == 64
        # except:
        #     print("Incorrect action type. It should be an array of shape (8,8)")
        action = np.reshape(action, (self.BOARD_SIZE, self.BOARD_SIZE))
        your_pawns, enemy_pawns = [], []
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r][c] == 1:
                    your_pawns.append((r, c))
        self.first_player_turn = not self.first_player_turn
        self.available_places = self.check_available_places(your_pawns)
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                self.board[r][c] *= -1
        if len(self.available_places) == 0:
            if self.no_moves:
                sum = 0
                for r in range(self.BOARD_SIZE):
                    for c in range(self.BOARD_SIZE):
                        sum += self.board[r][c]
                        if self.board[r][c] == 0:
                            sum += 2
                if sum > 0:
                    sum += self.BOARD_SIZE**2
                elif sum < 0:
                    sum -= self.BOARD_SIZE**2
                if not self.first_player_turn:
                    sum *= -1
                return (self.board, sum, True, None)
            self.no_moves = True
            return (self.board, 0, False, None)
        self.no_moves = False
        chosen_pos = None
        pos_value = -np.inf
        for place in self.available_places:
            if action[place[0]][place[1]] > pos_value or not chosen_pos:
                chosen_pos = [place[0], place[1]]
                pos_value = action[place[0]][place[1]]
        self.board[chosen_pos[0]][chosen_pos[1]] = -1
        self.swap_pawns(chosen_pos)

        your_pawns.clear()
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r][c] == 1:
                    your_pawns.append((r, c))
        self.available_places = self.check_available_places(your_pawns)
        return (self.board, 0, False, chosen_pos)
        # your_pawns.append((chosen_pos[0], chosen_pos[1]))
        # self.available_places = self.check_available_places(your_pawns)

    def manual_step(self):
        your_pawns, enemy_pawns = [], []
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                if self.board[r][c] == 1:
                    your_pawns.append((r, c))
        self.first_player_turn = not self.first_player_turn
        self.available_places = self.check_available_places(your_pawns)
        for r in range(self.BOARD_SIZE):
            for c in range(self.BOARD_SIZE):
                self.board[r][c] *= -1
        self.render()
        if len(self.available_places) == 0:
            if self.no_moves:
                sum = 0
                for r in range(self.BOARD_SIZE):
                    for c in range(self.BOARD_SIZE):
                        sum += self.board[r][c]
                        if self.board[r][c] == 0:
                            sum += 2
                if sum > 0:
                    sum += self.BOARD_SIZE**2
                elif sum < 0:
                    sum -= self.BOARD_SIZE**2
                if not self.first_player_turn:
                    sum *= -1
                return (self.board, sum, True, None)
            self.no_moves = True
            return (self.board, 0, False, None)
        self.no_moves = False
        chosen_pos = self.wait_for_input(self.available_places)
        self.board[chosen_pos[0]][chosen_pos[1]] = -1
        self.swap_pawns(chosen_pos)
        return (self.board, 0, False, chosen_pos)
        # self.board[chosen_pos[0]][chosen_pos[1]] = -1
        # self.swap_pawns(chosen_pos)
        # your_pawns.clear()
        # for r in range(8):
        #     for c in range(8):
        #         if self.board[r][c] == 1:
        #             your_pawns.append((r, c))
        # self.available_places = self.check_available_places(your_pawns)
        # return (self.board, 0, False, None)

    def wait_for_input(self, available_places):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONUP:
                    chosen_pos = (event.pos[0] // self.TILE_SIZE, event.pos[1] // self.TILE_SIZE)
                    if available_places.__contains__(chosen_pos):
                        return chosen_pos
                    else:
                        print("Incorrect move!")

    def check_available_places(self, your_pawns):
        available_places = set()
        for pawn in your_pawns:

            position = [pawn[0], pawn[1]]
            while self.board[position[0] - 1][position[1] - 1] == -1:
                position[0] -= 1
                position[1] -= 1
                if position[0] == 0 or position[1] == 0:
                    break
                if self.board[position[0] - 1][position[1] - 1] == 0:
                    available_places.add((position[0] - 1, position[1] - 1))

            position = [pawn[0], pawn[1]]
            while self.board[position[0]][position[1] - 1] == -1:
                position[1] -= 1
                if position[1] == 0:
                    break
                if self.board[position[0]][position[1] - 1] == 0:
                    available_places.add((position[0], position[1] - 1))

            position = [pawn[0], pawn[1]]
            while self.board[position[0] + 1][position[1] - 1] == -1:
                position[0] += 1
                position[1] -= 1
                if position[0] == self.BOARD_SIZE-1 or position[1] == 0:
                    break
                if self.board[position[0] + 1][position[1] - 1] == 0:
                    available_places.add((position[0] + 1, position[1] - 1))

            position = [pawn[0], pawn[1]]
            while self.board[position[0] - 1][position[1]] == -1:
                position[0] -= 1
                if position[0] == 0:
                    break
                if self.board[position[0] - 1][position[1]] == 0:
                    available_places.add((position[0] - 1,position[1]))

            position = [pawn[0], pawn[1]]
            while self.board[position[0] + 1][position[1]] == -1:
                position[0] += 1
                if position[0] == self.BOARD_SIZE-1:
                    break
                if self.board[position[0] + 1][position[1]] == 0:
                    available_places.add((position[0] + 1,position[1]))

            position = [pawn[0], pawn[1]]
            while self.board[position[0] - 1][position[1] + 1] == -1:
                position[0] -= 1
                position[1] += 1
                if position[0] == 0 or position[1] == self.BOARD_SIZE-1:
                    break
                if self.board[position[0] - 1][position[1] + 1] == 0:
                    available_places.add((position[0] - 1, position[1] + 1))

            position = [pawn[0], pawn[1]]
            while self.board[position[0]][position[1] + 1] == -1:
                position[1] += 1
                if position[1] == self.BOARD_SIZE-1:
                    break
                if self.board[position[0]][position[1] + 1] == 0:
                    available_places.add((position[0], position[1] + 1))

            position = [pawn[0], pawn[1]]
            while self.board[position[0] + 1][position[1] + 1] == -1:
                position[0] += 1
                position[1] += 1
                if position[0] == self.BOARD_SIZE-1 or position[1] == self.BOARD_SIZE-1:
                    break
                if self.board[position[0] + 1][position[1] + 1] == 0:
                    available_places.add((position[0] + 1, position[1] + 1))

        return available_places

    def swap_pawns(self, position):
        temp_pos = [position[0], position[1]]

        while True:
            temp_pos[0] -= 1
            temp_pos[1] -= 1
            if temp_pos[0] < 0 or temp_pos[1] < 0 or self.board[temp_pos[0]][temp_pos[1]] == 0:
                temp_pos = [position[0], position[1]]
                break
            if self.board[temp_pos[0]][temp_pos[1]] == -1:
                while temp_pos[0] != position[0]:
                    temp_pos[0] += 1
                    temp_pos[1] += 1
                    self.board[temp_pos[0]][temp_pos[1]] = -1
                break

        while True:
            temp_pos[1] -= 1
            if temp_pos[1] < 0 or self.board[temp_pos[0]][temp_pos[1]] == 0:
                temp_pos[1] = position[1]
                break
            if self.board[temp_pos[0]][temp_pos[1]] == -1:
                while temp_pos[1] != position[1]:
                    temp_pos[1] += 1
                    self.board[temp_pos[0]][temp_pos[1]] = -1
                break

        while True:
            temp_pos[0] += 1
            temp_pos[1] -= 1
            if temp_pos[0] >= self.BOARD_SIZE or temp_pos[1] < 0 or self.board[temp_pos[0]][temp_pos[1]] == 0:
                temp_pos = [position[0], position[1]]
                break
            if self.board[temp_pos[0]][temp_pos[1]] == -1:
                while temp_pos[0] != position[0]:
                    temp_pos[0] -= 1
                    temp_pos[1] += 1
                    self.board[temp_pos[0]][temp_pos[1]] = -1
                break

        while True:
            temp_pos[0] -= 1
            if temp_pos[0] < 0 or self.board[temp_pos[0]][temp_pos[1]] == 0:
                temp_pos[0] = position[0]
                break
            if self.board[temp_pos[0]][temp_pos[1]] == -1:
                while temp_pos[0] != position[0]:
                    temp_pos[0] += 1
                    self.board[temp_pos[0]][temp_pos[1]] = -1
                break

        while True:
            temp_pos[0] += 1
            if temp_pos[0] >= self.BOARD_SIZE or self.board[temp_pos[0]][temp_pos[1]] == 0:
                temp_pos[0] = position[0]
                break
            if self.board[temp_pos[0]][temp_pos[1]] == -1:
                while temp_pos[0] != position[0]:
                    temp_pos[0] -= 1
                    self.board[temp_pos[0]][temp_pos[1]] = -1
                break

        while True:
            temp_pos[0] -= 1
            temp_pos[1] += 1
            if temp_pos[0] < 0 or temp_pos[1] >= self.BOARD_SIZE or self.board[temp_pos[0]][temp_pos[1]] == 0:
                temp_pos = [position[0], position[1]]
                break
            if self.board[temp_pos[0]][temp_pos[1]] == -1:
                while temp_pos[0] != position[0]:
                    temp_pos[0] += 1
                    temp_pos[1] -= 1
                    self.board[temp_pos[0]][temp_pos[1]] = -1
                break

        while True:
            temp_pos[1] += 1
            if temp_pos[1] >= self.BOARD_SIZE or self.board[temp_pos[0]][temp_pos[1]] == 0:
                temp_pos[1] = position[1]
                break
            if self.board[temp_pos[0]][temp_pos[1]] == -1:
                while temp_pos[1] != position[1]:
                    temp_pos[1] -= 1
                    self.board[temp_pos[0]][temp_pos[1]] = -1
                break

        while True:
            temp_pos[0] += 1
            temp_pos[1] += 1
            if temp_pos[0] >= self.BOARD_SIZE or temp_pos[1] >= self.BOARD_SIZE or self.board[temp_pos[0]][temp_pos[1]] == 0:
                temp_pos = [position[0], position[1]]
                break
            if self.board[temp_pos[0]][temp_pos[1]] == -1:
                while temp_pos[0] != position[0]:
                    temp_pos[0] -= 1
                    temp_pos[1] -= 1
                    self.board[temp_pos[0]][temp_pos[1]] = -1
                break

    def render(self):
        if not self.screen:
            self.screen = pygame.display.set_mode(self.RESOLUTION)
        self.screen.fill((40, 120, 40))
        for r in range(self.BOARD_SIZE):
            line(r * self.TILE_SIZE, 0, r * self.TILE_SIZE, self.RESOLUTION[0]).draw(self.screen)
            line(0, r * self.TILE_SIZE, self.RESOLUTION[0], r * self.TILE_SIZE).draw(self.screen)
            for c in range(self.BOARD_SIZE):
                if (self.first_player_turn and self.board[r][c] == -1) or (not self.first_player_turn and self.board[r][c] == 1):
                    circle((r+0.5)*self.TILE_SIZE,(c+0.5)*self.TILE_SIZE,20,(255,255,255)).draw(self.screen)
                elif (self.first_player_turn and self.board[r][c] == 1) or (not self.first_player_turn and self.board[r][c] == -1):
                    circle((r+0.5)*self.TILE_SIZE,(c+0.5)*self.TILE_SIZE,20).draw(self.screen)
        for place in self.available_places:
            circle((place[0] + 0.5) * self.TILE_SIZE, (place[1] + 0.5) * self.TILE_SIZE, 10, (255, 255, 255, 100)).draw(self.screen)
        pygame.display.flip()


if __name__ == "__main__":
    env = Env()
    env.reset()
    while True:
        env.render()
        time.sleep(0.3)
        env.step(env.action_space.sample())
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         sys.exit()