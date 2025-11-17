class Stone:
    def __init__(self, name = "Stone", effect = None, n = 0, s = 0, e = 0, w = 0):
        self.name = name
        self.effect = effect
        self.n = n
        self.s = s
        self.e = e
        self.w = w

    def set_Owner(self, player):
        # Keep backward compatibility: set both `owner` and `player` attributes
        self.owner = player
        self.player = player

    def get_Owner(self):
        return self.owner
    
    def get_Attributes(self):
        return (self.n, self.s, self.e, self.w)
