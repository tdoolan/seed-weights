from seed import *

class Substrate:
    
    #TODO: Check default values
    def __init__(self, dim, seed, train=[], learn=0.9, thres=0.1):
        self.create(dim, seed)
        if train != []:
            print self.converge(train, learn, thres)

    def create(self, dim, seed):
        self.dim = dim
        self.seed = seed
        self.nodes = {}
        
        bias_loc = (float('inf'), float('inf') ,0)
        bias_node = Bias(bias_loc)
        self.nodes[bias_loc] = bias_node

        for i in range(dim[0]):
            for j in range(dim[1]):
                l = (i,j,0) #Layer 0
                self.nodes[l] = Input(l)
        
        for i in range(dim[0]):
            for j in range(dim[1]):
                l = (i,j,1) #Layer 1
                node = Sigmoid(l)
                self.nodes[l] = node
                for k,w in seed.con.items():
                    prev_node = self.nodes.get((i+k[0], j+k[1], 1+k[2]), None)
                    if prev_node != None:
                        node.con[k] = Connection(w, prev_node)
    
    def process(self, data):
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.nodes[(i, j, 0)].setValue(data[i][j])
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                self.nodes[(i, j, 1)].activate()

    def getOutput(self):
        out = [[0.0]*self.dim[1] for q in range(self.dim[0])]
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                out[i][j] = self.nodes[(i, j, 1)].output
        return out

    def converge(self, train, learn, thres):
        tot = 10**6 #FIX this!
        tot_old = 10**7
        while (tot_old - tot) > thres:
            tot_old = tot
            tot = 0.0
            for pair in train:
                self.process(pair[0])
                for i in range(self.dim[0]):
                    for j in range(self.dim[1]):
                        self.nodes[(i, j, 1)].updateDw(pair[1][i][j])
                        tot += self.nodes[(i, j, 1)].computeError(pair[1][i][j])
            
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    self.nodes[(i, j, 1)].updateWeights(learn)
            tot = (tot / len(train))
            #print tot
        return tot

    def validate(self, val):
        t = 0.0
        for pair in val:
            self.process(pair[0])
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    t += self.nodes[(i, j, 1)].computeError(pair[1][i][j])
        return (t / len(val))

    def modelWeights(self):
        weights = self.seed.con.keys()
        weight_store = dict(zip(weights, [[] for i in range(len(weights))]))
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                node = self.nodes[(i, j, 1)]
                for loc_diff in weight_store.keys():
                    if loc_diff in node.con:
                        weight_store[loc_diff].append(((i,j,1),
                            node.con[loc_diff].weight))

        for loc_diff,store in weight_store.items():
            b = best([elem[1] for elem in store])
            for elem in store:
                self.nodes[elem[0]].con[loc_diff].weight = b
        #self.printWeights()
    
    def printWeights(self):
        sd0 = self.seed.dim[0]
        sd1 = self.seed.dim[1]
        node = self.nodes[(self.dim[0]/2, self.dim[1]/2, 1)]
        print "\n%+.2f" % node.con[(float('inf'), float('inf'), -1)].weight
        for i in range(-sd0, sd0 +1):
            for j in range(-sd1, sd1 +1):
                print "%+.2f " % node.con[(i, j, -1)].weight,
            print
    
    #TODO: Deep_copy substrate (until Connection) for cascorr and rbf!

def best(l):
    return max(l, key=math.fabs)
    
