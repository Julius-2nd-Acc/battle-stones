class Stone:
    def __init__(self, name = "Stone", effect = None, n = 0, s = 0, e = 0, w = 0, owner = None):
        self.name = name
        self.effect = effect
        self.n = n
        self.s = s
        self.e = e
        self.w = w
        self.owner = owner

    def set_Owner(self, player):
        # Keep backward compatibility: set both `owner` and `player` attributes
        self.owner = player
        self.player = player

    def get_Owner(self):
        return self.owner
    
    def get_representation(self):
        return f"{self.name}({self.n},{self.s},{self.e},{self.w})"
    
    def get_Attributes(self):
        return (self.n, self.s, self.e, self.w)
