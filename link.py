class Link2D:
    def __init__(self, length, rot, loc):
        self.length = length
        self.rot = 0
        if rot != None:
            self.rot = rot

        self.loc = [0,0]
        if loc != None:
            self.loc = loc