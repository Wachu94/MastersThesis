import random

class actionSpace:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randrange(0, self.n)