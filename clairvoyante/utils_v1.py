import os
home_dir = os.path.expanduser('~')
import sys
sys.path.append(home_dir+'/miniconda2/lib/python2.7/site-packages')
import intervaltree
import numpy as np
import random
import param
import blosc
import gc

base2num = dict(zip("ACGT",(0, 1, 2, 3)))

def SetupEnv():

    blosc.set_nthreads(4)
    gc.enable()


def GetTensor( tensor_fn, num ):
        f = open( tensor_fn, "r" )
        c = 0
        XArray = []
        posArray = []
        for row in f: # A variant per row
            row = row.strip().split()
            ctgName = row[0]  # Column 1: sequence name
            pos = int(row[1]) # Column 2: position
            key = ctgName + ":" + str(pos)
            refSeq = row[2]  # Column 3: reference seqeunces

            if refSeq[param.flankingBaseNum] not in ["A","C","G","T"]:
                continue

            x = np.reshape(np.array([float(x) for x in row[3:]]), (2*param.flankingBaseNum+1,4,param.matrixNum))

            for i in range(1, param.matrixNum):
                x[:,:,i] -= x[:,:,0]

            XArray.append(x)
            posArray.append(key)
            c += 1

            if c == num:
                yield 0, c, np.array(XArray), np.array(posArray)
                c = 0
                XArray = []
                posArray = []

        yield 1, c, np.array(XArray), np.array(posArray)


def GetTrainingArray( tensor_fn, var_fn, bed_fn, shuffle = True ):
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
    if shuffle == True:
        random.shuffle(allPos)

    XArrayCompressed = []
    YArrayCompressed = []
    posArrayCompressed = []
    XArray = []
    YArray = []
    posArray = []
    count = 0
    total = 0
    for key in allPos:
        total += 1
        XArray.append(X[key])
        YArray.append(Y[key])
        posArray.append(key)
        count += 1
        if count == param.bloscBlockSize:
            XArrayCompressed.append(blosc.pack_array(np.array(XArray), cname='lz4hc'))
            YArrayCompressed.append(blosc.pack_array(np.array(YArray), cname='lz4hc'))
            posArrayCompressed.append(blosc.pack_array(np.array(posArray), cname='lz4hc'))
            XArray = []
            YArray = []
            posArray = []
            count = 0
    if count >= 0:
        XArrayCompressed.append(blosc.pack_array(np.array(XArray), cname='lz4hc'))
        YArrayCompressed.append(blosc.pack_array(np.array(YArray), cname='lz4hc'))
        posArrayCompressed.append(blosc.pack_array(np.array(posArray), cname='lz4hc'))

    return total, XArrayCompressed, YArrayCompressed, posArrayCompressed


def DecompressArray( array, start, num, maximum ):
    endFlag = 0
    if start + num >= maximum:
        num = maximum - start
        endFlag = 1
    leftEnd = start % param.bloscBlockSize
    startingBlock = int(start / param.bloscBlockSize)
    maximumBlock = int((start+num-1) / param.bloscBlockSize)
    rt = []
    rt.append(blosc.unpack_array(array[startingBlock]))
    startingBlock += 1
    if startingBlock <= maximumBlock:
        for i in range(startingBlock, (maximumBlock+1)):
            rt.append(blosc.unpack_array(array[i]))
    nprt = np.concatenate(rt[:])
    if leftEnd != 0 or num % param.bloscBlockSize != 0:
        nprt = nprt[leftEnd:(leftEnd+num)]

    return nprt, num, endFlag

