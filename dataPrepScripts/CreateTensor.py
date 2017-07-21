from readfq import readfq
import argparse
import os
import re
import shlex
import subprocess
import sys
import numpy as np
from sklearn import preprocessing
import param

cigarRe = r"(\d+)([MIDNSHP=X])"
base2num = dict(zip("ACGT", (0,1,2,3)))

def GenerateTensor(ctgName, alns, center, refSeq):
    aln_code = np.zeros( (2*param.flankingBaseNum+1, 4, param.matrixNum) )
    for aln in alns:
        for refPos, queryPos, refBase, queryBase in aln:
            if refBase not in ("A", "C", "G", "T"):
                continue
            if refPos - center >= -(param.flankingBaseNum+1) and refPos - center < param.flankingBaseNum:
                offset = refPos - center + (param.flankingBaseNum+1)
                if queryBase != "-":
                    if refBase != "-":
                        aln_code[offset][ base2num[refBase] ][0] += 1.0
                        aln_code[offset][ base2num[queryBase] ][1] += 1.0
                        aln_code[offset][ base2num[refBase] ][2] += 1.0
                        aln_code[offset][ base2num[queryBase] ][3] += 1.0
                        for i in [i for i in range(param.matrixNum) if i != base2num[refBase]]:
                            aln_code[offset][i][0] -= 0.333333
                            aln_code[offset][i][2] -= 0.333333
                        for i in [i for i in range(param.matrixNum) if i != base2num[queryBase]]:
                            aln_code[offset][i][1] -= 0.333333
                            aln_code[offset][i][3] -= 0.333333
                    else:
                        aln_code[offset][ base2num[queryBase] ][1] += 1.0
                        for i in [i for i in range(param.matrixNum) if i != base2num[queryBase]]:
                            aln_code[offset][i][1] -= 0.333333
                elif queryBase == "-":
                    if refBase != "-":
                        aln_code[offset][ base2num[refBase] ][2] += 1.0
                        for i in [i for i in range(param.matrixNum) if i != base2num[refBase]]:
                            aln_code[offset][i][2] -= 0.333333
                    else:
                        print >> sys.stderr, "Should not reach here: %s, %s" % (refBase, queryBase)
                else:
                    print >> sys.stderr, "Should not reach here: %s, %s" % (refBase, queryBase)

    for i in range(param.matrixNum):
        aln_code[:,:,i] = preprocessing.normalize(aln_code[:,:,i])

    outputLine = []
    outputLine.append( "%s %d %s" %  (ctgName, center, refSeq[center-(param.flankingBaseNum+1):center+param.flankingBaseNum]) )
    for x in np.reshape(aln_code, (2*param.flankingBaseNum+1)*4*param.matrixNum):
        outputLine.append("%0.3f" % x)
    return " ".join(outputLine)

def output_aln_tensor(args):

    bam_fn = args.bam_fn
    pi_fn = args.pi_fn
    ctgName = args.ctgName
    ctgStart = args.ctgStart
    ctgEnd = args.ctgEnd
    samtools = args.samtools
    ref_fn = args.ref_fn
    tensor_fn = args.tensor_fn

    refSeq = None
    ref_fp = open(ref_fn, 'r')
    for name, seq, qual in readfq(ref_fp):
        if name != ctgName:
            continue
        refSeq = seq
        break

    if refSeq == None:
        print >> sys.stderr, "Cannot find reference sequence %s" % (ctgName)
        sys.exit(1)


    beginToEnd = {}
    with open(pi_fn) as f:
        for row in f.readlines():
            row = row.strip().split()
            pos = int(row[1])
            beginToEnd[ pos-(param.flankingBaseNum+1) ] = (pos + (param.flankingBaseNum+1), pos)

    p = subprocess.Popen(shlex.split("%s view %s %s:%d-%d" % (samtools, bam_fn, ctgName, ctgStart, ctgEnd) ), stdout=subprocess.PIPE, bufsize=8388608)\
        if ctgStart and ctgEnd\
        else subprocess.Popen(shlex.split("%s view %s %s" % (samtools, bam_fn, ctgName) ), stdout=subprocess.PIPE, bufsize=8388608)

    centerToAln = {}

    tensor_fp = open(tensor_fn, "w")

    for l in p.stdout:
        l = l.strip().split()
        if l[0][0] == "@":
            continue

        QNAME = l[0]
        FLAG = int(l[1])
        RNAME = l[2]
        POS = int(l[3]) - 1 # switch from 1-base to 0-base to match sequence index
        CIGAR = l[5]
        SEQ = l[9]
        refPos = POS
        queryPos = 0

        endToCenter = {}
        activeSet = set()

        for m in re.finditer(cigarRe, CIGAR):
            advance = int(m.group(1))
            if m.group(2) == "S":
                queryPos += advance
            if m.group(2) in ("M", "=", "X"):
                matches = []
                for i in xrange(advance):
                    matches.append( (refPos, SEQ[queryPos]) )
                    if refPos in beginToEnd:
                        r_end, r_center = beginToEnd[refPos]
                        endToCenter[r_end] = r_center
                        activeSet.add(r_center)
                        centerToAln.setdefault(r_center, [])
                        centerToAln[r_center].append([])
                    for center in list(activeSet):
                        centerToAln[center][-1].append( (refPos, queryPos, refSeq[refPos], SEQ[queryPos] ) )
                    if refPos in endToCenter:
                        center = endToCenter[refPos]
                        activeSet.remove(center)
                    refPos += 1
                    queryPos += 1

            elif m.group(2) == "I":
                for i in range(advance):
                    for center in list(activeSet):
                        centerToAln[center][-1].append( (refPos, queryPos, "-", SEQ[queryPos] ))
                    queryPos += 1

            elif m.group(2) == "D":
                for i in xrange(advance):
                    for center in list(activeSet):
                        centerToAln[center][-1].append( (refPos, queryPos, refSeq[refPos], "-" ))
                    if refPos in beginToEnd:
                        r_end, r_center = beginToEnd[refPos]
                        endToCenter[r_end] = r_center
                        activeSet.add(r_center)
                        centerToAln.setdefault(r_center, [])
                        centerToAln[r_center].append([])
                    if refPos in endToCenter:
                        center = endToCenter[refPos]
                        activeSet.remove(center)
                    refPos += 1


        for center in centerToAln.keys():
            if center + (param.flankingBaseNum+1) < POS:
                l =  GenerateTensor(ctgName, centerToAln[center], center, refSeq)
                print >> tensor_fp, l
                del centerToAln[center]

    for center in centerToAln.keys():
        if center + (param.flankingBaseNum+1) < POS:
            l =  GenerateTensor(ctgName, centerToAln[center], center, refSeq)
            print >> tensor_fp, l


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='Generate tensors summarizing local alignments from a BAM file and a list of candidate locations' )

    parser.add_argument('--bam_fn', type=str, default="input.bam", 
            help="Sorted bam file input, default: input.bam")

    parser.add_argument('--ref_fn', type=str, default="ref.fa", 
            help="Reference fasta file input, default: ref.fa")
    
    parser.add_argument('--pi_fn', type=str, default="pileup.out", 
            help="Pile-up count input, default: pileup.out")
    
    parser.add_argument('--tensor_fn', type=str, default="tensor.out", 
            help="Tensor output, default: tensor.out")

    parser.add_argument('--ctgName', type=str, default="chr17", 
            help="The name of sequence to be processed, defaults: chr17")

    parser.add_argument('--ctgStart', type=int, default=None,
            help="The 1-bsae starting position of the sequence to be processed, defaults: None")

    parser.add_argument('--ctgEnd', type=int, default=None,
            help="The inclusive ending position of the sequence to be processed, defaults: None")

    parser.add_argument('--samtools', type=str, default="samtools", 
            help="Path to the 'samtools', default: samtools")

    args = parser.parse_args()

    output_aln_tensor(args)

