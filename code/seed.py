import random, math

class Seed:
    def __init__(self, init, dim=(0,0)):
        self.init = init
        self.dim = dim
        self.bias = random.uniform(-init, init)
        
        #TODO: CasCorr = [{}, ...] #one dict (&bias) for each sigmoid!
        self.con = {} #{(dx,dy,dz): weight}
        for i in range(-dim[0], dim[0]+1):
            for j in range(-dim[1], dim[1]+1):
                self.con[(i, j, -1)] = random.uniform(-init, init)
    
    #TODO: More complex grow functions
    def grow(self):
        return Seed(self.init, (self.dim[0]+1, self.dim[1]+1)) 

class Node:
    def __init__(self, loc):
        self.loc = loc
        self.output = 0.0

class Sigmoid(Node):
    def __init__(self, loc):
        Node.__init__(self, loc)
        self.con = {} #{(dx,dy,dz): Connection}  #incoming cons
        self.delta = 0.0

    def activate(self):
        inp = 0.0
        for con in self.con.values():
            inp += con.weight * con.node.output
        self.output = 1.0 / (1 + math.exp(-inp))

    def updateDw(self, target):
        delta = (target - self.output) * (self.output*(1-self.output))
        for con in self.con.values():
            con.dw += con.node.output*delta
    
    def computeError(self, target):
        return ((target - self.output)**2) / 2
    
    def updateWeights(self, learn):
        for con in self.con.values():
            con.weight += learn*con.dw
            con.dw = 0.0

class Input(Node):
    def __init__(self, loc):
        Node.__init__(self, loc)

    def setValue(self, value):
        self.output = value

class Bias(Node):
    def __init__(self, loc):
        Node.__init__(self, loc)
        self.output = 1.0

class Connection:
    def __init__(self, weight, node):
        self.weight = weight
        self.dw = 0.0
        self.node = node

