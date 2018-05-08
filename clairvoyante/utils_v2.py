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
import shlex
import subprocess

base2num = dict(zip("ACGT",(0, 1, 2, 3)))

def SetupEnv():
    os.environ["CXX"] = "g++"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    blosc.set_nthreads(4)
    gc.enable()

def UnpackATensorRecord(a, b, c, *d):
    return a, b, c, np.array(d, dtype=np.float32)

def GetTensor( tensor_fn, num ):
    if tensor_fn != "PIPE":
        f = subprocess.Popen(shlex.split("gzip -fdc %s" % (tensor_fn) ), stdout=subprocess.PIPE, bufsize=8388608)
        fo = f.stdout
    else:
        fo = sys.stdin
    total = 0
    c = 0
    rows = np.empty((num, ((2*param.flankingBaseNum+1)*4*param.matrixNum)), dtype=np.float32)
    pos = []
    for row in fo: # A variant per row
        try:
            chrom, coord, seq, rows[c] = UnpackATensorRecord(*(row.split()))
        except ValueError:
            print >> sys.stderr, "UnpackATensorRecord Failure", row
        seq = seq.upper()
        if seq[param.flankingBaseNum] not in ["A","C","G","T"]: # TODO: Support IUPAC in the future
            continue
        pos.append(chrom + ":" + coord + ":" + seq)
        c += 1

        if c == num:
            x = np.reshape(rows, (num,2*param.flankingBaseNum+1,4,param.matrixNum))
            for i in range(1, param.matrixNum): x[:,:,:,i] -= x[:,:,:,0]
            total += c; print >> sys.stderr, "Processed %d tensors" % total
            yield 0, c, x, pos
            c = 0
            rows = np.empty((num, ((2*param.flankingBaseNum+1)*4*param.matrixNum)), dtype=np.float32)
            pos = []

    if tensor_fn != "PIPE":
        fo.close()
        f.wait()
    x = np.reshape(rows[:c], (c,2*param.flankingBaseNum+1,4,param.matrixNum))
    for i in range(1, param.matrixNum): x[:,:,:,i] -= x[:,:,:,0]
    total += c; print >> sys.stderr, "Processed %d tensors" % total
    yield 1, c, x, pos


def GetTrainingArray( tensor_fn, var_fn, bed_fn, shuffle = True ):
    tree = {}
    if bed_fn != None:
        f = subprocess.Popen(shlex.split("gzip -fdc %s" % (bed_fn) ), stdout=subprocess.PIPE, bufsize=8388608)
        for row in f.stdout:
            row = row.split()
            name = row[0]
            if name not in tree:
                tree[name] = intervaltree.IntervalTree()
            begin = int(row[1])
            end = int(row[2])-1
            if end == begin: end += 1
            tree[name].addi(begin, end)
        f.stdout.close()
        f.wait()

    Y = {}
    if var_fn != None:
        f = subprocess.Popen(shlex.split("gzip -fdc %s" % (var_fn) ), stdout=subprocess.PIPE, bufsize=8388608)
        for row in f.stdout:
            row = row.split()
            ctgName = row[0]
            pos = int(row[1])
            if bed_fn != None:
                if len(tree[ctgName].search(pos)) == 0:
                    continue
            key = ctgName + ":" + str(pos)

            baseVec = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
            #          --------------  ------  ------------    ------------------
            #          Base chng       Zygo.   Var type        Var length
            #          A   C   G   T   HET HOM REF SNP INS DEL 0   1   2   3   4   >=4
            #          0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15

            if row[4] == "0" and row[5] == "1":
                if len(row[2]) == 1 and len(row[3]) == 1:
                    baseVec[base2num[row[2][0]]] = 0.5
                    baseVec[base2num[row[3][0]]] = 0.5
                elif len(row[2]) > 1 or len(row[3]) > 1:
                    baseVec[base2num[row[2][0]]] = 0.5
                baseVec[4] = 1.

            elif row[4] == "1" and row[5] == "1":
                if len(row[2]) == 1 and len(row[3]) == 1:
                    baseVec[base2num[row[3][0]]] = 1
                elif len(row[2]) > 1 or len(row[3]) > 1:
                    pass
                baseVec[5] = 1.

            if len(row[2]) > 1 and len(row[3]) == 1: baseVec[9] = 1. # deletion
            elif len(row[3]) > 1 and len(row[2]) == 1: baseVec[8] = 1.  # insertion
            else: baseVec[7] = 1.  # SNP

            varLen = abs(len(row[2])-len(row[3]))
            if varLen > 4: baseVec[15] = 1.
            else: baseVec[10+varLen] = 1.

            Y[key] = baseVec
        f.stdout.close()
        f.wait()

    X = {}
    f = subprocess.Popen(shlex.split("gzip -fdc %s" % (tensor_fn) ), stdout=subprocess.PIPE, bufsize=8388608)
    total = 0
    mat = np.empty(((2*param.flankingBaseNum+1)*4*param.matrixNum), dtype=np.float32)
    for row in f.stdout:
        chrom, coord, seq, mat = UnpackATensorRecord(*(row.split()))
        if bed_fn != None:
            if chrom not in tree: continue
            if len(tree[chrom].search(int(coord))) == 0: continue
        seq = seq.upper()
        if seq[param.flankingBaseNum] not in ["A","C","G","T"]: continue
        key = chrom + ":" + coord

        x = np.reshape(mat, (2*param.flankingBaseNum+1,4,param.matrixNum))
        for i in range(1, param.matrixNum): x[:,:,i] -= x[:,:,0]

        X[key] = np.copy(x)

        if key not in Y:
            baseVec = [0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.]
            #          --------------  ------  ------------    ------------------
            #          Base chng       Zygo.   Var type        Var length
            #          A   C   G   T   HET HOM REF SNP INS DEL 0   1   2   3   4   >=4
            #          0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
            baseVec[base2num[seq[param.flankingBaseNum]]] = 1.
            Y[key] = baseVec

        total += 1
        if total % 100000 == 0: print >> sys.stderr, "Processed %d tensors" % total
    f.stdout.close()
    f.wait()

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

