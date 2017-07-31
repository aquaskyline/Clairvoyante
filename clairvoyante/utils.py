import sys
sys.path.append('/home-4/rluo5@jhu.edu/miniconda2/lib/python2.7/site-packages/')
import intervaltree
import numpy as np
import random
import param

base2num = dict(zip("ACGT",(0, 1, 2, 3)))

def GetBatch(X, Y, size):
    s = random.randint(0,len(X)-size)
    return X[s:s+size], Y[s:s+size]

def GetAlnArray( tensor_fn ):

    X = {}

    with open( tensor_fn ) as f:
        for row in f: # A variant per row
            row = row.strip().split()
            ctgName = row[0]  # Column 1: sequence name
            pos = int(row[1]) # Column 2: position
            key = ctgName + ":" + str(pos)
            refSeq = row[2]  # Column 3: reference seqeunces

            if refSeq[param.flankingBaseNum] not in ["A","C","G","T"]: # Skip non-ACGT reference bases
                continue

            x = np.reshape(np.array([float(x) for x in row[3:]]), (2*param.flankingBaseNum+1,4,param.matrixNum))

            for i in range(1, param.matrixNum):
                x[:,:,i] -= x[:,:,0]

            X[key] = x

    allPos = sorted(X.keys())

    XArray = []
    posArray = []
    for pos in allPos:
        XArray.append(X[pos])
        posArray.append(pos)
    XArray = np.array(XArray)

    return XArray, posArray

def GetTrainingArray( tensor_fn, var_fn, bed_fn ):
    tree = {}
    with open(bed_fn) as f:
        for row in f:
            row = row.strip().split()
            name = row[0]
            if name not in tree:
                tree[name] = intervaltree.IntervalTree()
            begin = int(row[1])
            end = int(row[2])
            tree[name].addi(begin, end)

    Y = {}
    with open( var_fn ) as f:
        for row in f:
            row = row.strip().split()
            ctgName = row[0]
            pos = int(row[1])
            if len(tree[ctgName].search(pos)) == 0:
                continue
            key = ctgName + ":" + str(pos)

            baseVec = [0., 0., 0., 0., 0., 0., 0., 0., 0.]  # A, C, G, T, het, hom, insertion, deletion, ref

            if row[4] == "0" and row[5] == "1":
                baseVec[base2num[row[2][0]]] = 0.5
                baseVec[base2num[row[3][0]]] = 0.5
                baseVec[4] = 1.

            elif row[4] == "1" and row[5] == "1":
                baseVec[base2num[row[3][0]]] = 1
                baseVec[5] = 1.

            if len(row[2]) > 1:  # deletion
                baseVec[4] = 0.
                baseVec[5] = 0.
                baseVec[6] = 0.
                baseVec[7] = 1.
                baseVec[8] = 0.

            if len(row[3]) > 1:  # insertion
                baseVec[4] = 0.
                baseVec[5] = 0.
                baseVec[6] = 1.
                baseVec[7] = 0.
                baseVec[8] = 0.

            Y[key] = baseVec

    X = {}
    with open( tensor_fn ) as f:
        for row in f:
            row = row.strip().split()
            ctgName = row[0]
            pos = int(row[1])
            if ctgName not in tree:
                continue
            if len(tree[ctgName].search(pos)) == 0:
                continue
            key = ctgName + ":" + str(pos)
            refSeq = row[2]
            if refSeq[param.flankingBaseNum] not in ["A","C","G","T"]:
                continue

            x = np.reshape(np.array([float(x) for x in row[3:]]), (2*param.flankingBaseNum+1,4,param.matrixNum))

            for i in range(1, param.matrixNum):
                x[:,:,i] -= x[:,:,0]

            X[key] = x

            if key not in Y:
                baseVec = [0., 0., 0., 0., 0., 0., 0., 0., 1.]
                baseVec[base2num[refSeq[param.flankingBaseNum]]] = 1.
                Y[key] = baseVec

    allPos = sorted(X.keys())
    random.shuffle(allPos)

    XArray = []
    YArray = []
    posArray = []
    for key in allPos:
        XArray.append(X[key])
        YArray.append(Y[key])
        posArray.append(key)
    XArray = np.array(XArray)
    YArray = np.array(YArray)

    return XArray, YArray, posArray

