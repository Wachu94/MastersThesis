from pygame import gfxdraw

class circle:
    def __init__(self, x, y, r, color = (0, 0, 0)):
        self.x = x
        self.y = y
        self.r = r
        self.color = color

    def draw(self, screen):
        gfxdraw.filled_ellipse(screen, int(self.x), int(self.y), self.r, self.r, self.color)
        gfxdraw.aaellipse(screen, int(self.x), int(self.y), self.r, self.r, (0,0,0))


class square:
    def __init__(self, x, y, a, color = (255, 255, 255), offset=5):
        self.x = x
        self.y = y
        self.a = a
        self.color = color
        self.offset = offset

    def draw(self, screen):
        gfxdraw.filled_polygon(screen,((self.a*self.x+self.offset,self.a*self.y+self.offset),
                                       (self.a*self.x+self.a-self.offset,self.a*self.y+self.offset),
                                       (self.a*self.x+self.a-self.offset,self.a*self.y+self.a-self.offset),
                                       (self.a*self.x+self.offset,self.a*self.y+self.a-self.offset)),
                                        self.color)
        gfxdraw.aapolygon(screen,((self.a*self.x+self.offset,self.a*self.y+self.offset),
                                  (self.a*self.x+self.a-self.offset,self.a*self.y+self.offset),
                                  (self.a*self.x+self.a-self.offset,self.a*self.y+self.a-self.offset),
                                  (self.a*self.x+self.offset,self.a*self.y+self.a-self.offset)),(0,0,0))
        gfxdraw.line(screen, self.a*self.x+self.offset, self.a*self.y+self.a-self.offset, self.a*self.x+self.a-self.offset, self.a*self.y+self.a-self.offset, (0,0,0))

class line:
    def __init__(self, x1, y1, x2, y2, color = (0,0,0)):
        self.positions = (x1, y1, x2, y2)
        self.color = color

    def draw(self, screen):
        gfxdraw.line(screen, self.positions[0], self.positions[1], self.positions[2], self.positions[3], self.color)