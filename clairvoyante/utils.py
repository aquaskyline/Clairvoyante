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
            refSeq = row[2]  # Column 3: reference seqeunces

            if refSeq[param.flankingBaseNum] not in ["A","C","G","T"]: # Skip non-ACGT bases
                continue

            x = np.reshape(np.array([float(x) for x in row[3:]]), (2*param.flankingBaseNum+1,4,param.matrixNum))

            #for i in range(1, param.matrixNum):
            #    x[:,:,i] -= x[:,:,0]

            X[pos] = x

    allPos = sorted(X.keys())

    XArray = []
    posArray = []
    for pos in allPos:
        XArray.append(X[pos])
        posArray.append(pos)
    XArray = np.array(XArray)

    return XArray, posArray

def GetTrainingArray( tensor_fn, var_fn, bed_fn, ctgName ):
    tree = intervaltree.IntervalTree()
    with open(bed_fn) as f:
        for row in f:
            row = row.strip().split()
            if row[0] != ctgName:
                continue
            begin = int(row[1])
            end = int(row[2])
            tree.addi(begin, end)

    Y = {}
    with open( var_fn ) as f:
        for row in f:

            row = row.strip().split()
            ctgName = row[0]

            pos = int(row[1])
            if len(tree.search(pos)) == 0:
                continue

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

            Y[pos] = baseVec

    X = {}

    with open( tensor_fn ) as f:
        for row in f:

            row = row.strip().split()
            ctgName = row[0]

            pos = int(row[1])
            if len(tree.search(pos)) == 0:
                continue

            refSeq = row[2]
            if refSeq[param.flankingBaseNum] not in ["A","C","G","T"]:
                continue

            x = np.reshape(np.array([float(x) for x in row[3:]]), (2*param.flankingBaseNum+1,4,param.matrixNum))

            #for i in range(1, param.matrixNum):
            #    x[:,:,i] -= x[:,:,0]

            X[pos] = x

            if pos not in Y:
                baseVec = [0., 0., 0., 0., 0., 0., 0., 0., 1.]
                baseVec[base2num[refSeq[param.flankingBaseNum]]] = 1.
                Y[pos] = baseVec

    allPos = sorted(X.keys())
    random.shuffle(allPos)

    XArray = []
    YArray = []
    posArray = []
    for pos in allPos:
        XArray.append(X[pos])
        YArray.append(Y[pos])
        posArray.append(pos)
    XArray = np.array(XArray)
    YArray = np.array(YArray)

    return XArray, YArray, posArray

