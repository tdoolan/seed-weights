from seed import *

class Substrate:
    
    #TODO: Check default values
    def __init__(self, dim, seed, train=[], learn=0.9, thres=0.1):
        self.create(dim, seed)
        if train != []:
            train = self.questionmark(train)
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

    def questionmark(self, data):
        marked = []
        allvalues = {}
        for pair in data:
            #Determine the nodes where input & ouput = 0
            zerovalues = {}
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    if (pair[0][i][j] == 0):#and (pair[1][i][j] == 0):
                        zerovalues[(i, j, 1)] = None
                    allvalues[(i, j, 1)] = None
            #Remove the radius of the seed network, thus all nodes that are not
            #surrounded by zerovalues.
            unsure = {}
            for elem in zerovalues.keys():
                reject = False
                for i in range(-self.seed.dim[0], self.seed.dim[0]+1):
                    for j in range(-self.seed.dim[1], self.seed.dim[1]+1):
                        neigh = (elem[0]+i, elem[1]+j, elem[2])
                        #Check if neighbour is not zero and is inside substrate
                        if neigh not in zerovalues and neigh in allvalues:
                            reject = True
                if not reject:
                    unsure[elem] = None
            #Construct the new training tuple, ignore updateDw for unsure nodes
            marked.append((pair[0], pair[1], unsure))
        return marked

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
                        loc = (i, j, 1)
                        #if loc not in pair[2]: #Non questionmarked
                        self.nodes[loc].updateDw(pair[1][i][j], pair[2])
                        tot += self.nodes[loc].computeError(pair[1][i][j])
            
            for i in range(self.dim[0]):
                for j in range(self.dim[1]):
                    self.nodes[(i, j, 1)].updateWeights(learn)
            tot = (tot / len(train))
            print tot
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
                loc = (i, j, 1)
                node = self.nodes[loc]
                for loc_diff in weight_store.keys():
                    if loc_diff in node.con:
                        weight_store[loc_diff].append((loc,
                            node.con[loc_diff].weight, self.counts[loc]))
        #Store: Location, Weight, Update count
        #Recompute all weights according to update ratios
        for loc_diff,store in weight_store.items(): 
            total = sum((elem[2] for elem in store))
            weight = sum((elem[1]*(float(elem[2])/total) for elem in store))
            for elem in store:
                self.nodes[elem[0]].con[loc_diff].weight = weight
        self.printWeights()
    
    """def modelWeights(self): #Max
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
        self.printWeights()"""
    
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
    
