#Also imports seed via substrate
from substrate import *

#Seed-network based weight-sharing
def seed_weights(init, dim, train, learn, thres, val, margin, test):
    seed1 = Seed(init, (1,1))
    sub1 = Substrate(dim, seed1, train, learn, thres)
    print "Reg Val", sub1.validate(val)
    sub1.modelWeights()
    val1 = sub1.validate(val)
    print "Model Val", val1
    
    while True:
        seed2 = seed1.grow()
        sub2 = Substrate(dim, seed2, train, learn, thres)
        print "Reg Val", sub2.validate(val)
        sub2.modelWeights()
        val2 = sub2.validate(val)
        print "Model Val", val2

        if (val1 - val2) < margin:
            break
        seed1 = seed2
        sub1 = sub2
        val1 = val2

    print "Test", sub1.validate(test)
    sub1.printWeights()

def orig_boxes(dim):
    s = Seed(0)
    for i in range(-1, 2):
        for j in range(-1, 2):
            s.con[(i, j, -1)] = 1.0
    s.con[(float('inf'), float('inf'), -1)] = -7.0
    sub = Substrate(dim, s)

    data = []
    for i in range(dim[0] - 2):
        for j in range(dim[1] - 2):
            for x in range(dim[0]):
                for y in range(dim[1]):
                    if (x < i or x > i+2) or (y < j or y > j+2):
                        inp = [[0]*dim[1] for q in range(dim[0])]
                        for di in range(3):
                            for dj in range(3):
                                inp[i+di][j+dj] = 1
                        inp[x][y] = 1
                        sub.process(inp)
                        data.append((inp, sub.getOutput()))
                        """out = [[0]*dim[1] for q in range(dim[0])]
                        out[i+1][j+1] = 1
                        data.append((inp, out))"""
    random.shuffle(data)
    print "Data generated", len(data)

    l = len(data) / 125
    seed_weights(0.3, dim, data[:l], 0.9, 0.0001, data[l:l*2], 0.0, data[l*2:])

if __name__ == "__main__":
    orig_boxes((11,11))

