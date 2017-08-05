import os
homeDir = os.path.expanduser('~')
import sys
sys.path.append(homeDir+'/miniconda2/lib/python2.7/site-packages')
from readfq import readfq
import argparse
import os
import re
import shlex
import subprocess
from math import log

cigarRe = r"(\d+)([MIDNSHP=X])"

def OutputCandidate(ctgName, pos, baseCount, refBase, minCoverage, threshold):

    totalCount = 0
    totalCount += sum(x[1] for x in baseCount) 
    if totalCount < minCoverage:
        return None

    baseCount.sort(key = lambda x:-x[1]) # sort baseCount descendingly
    p0 = float(baseCount[0][1]) / totalCount
    p1 = float(baseCount[1][1]) / totalCount
    output = []
    if (p0 <= 1.0 - threshold and p1 >= threshold) or baseCount[0][0] != refBase:
        output = [ctgName, pos+1, refBase, totalCount]
        output.extend( ["%s %d" % x for x in baseCount] )
        output = " ".join([str(x) for x in output])
        return totalCount, output
    else:
        return None

def MakeCandidates( args ):

    bam_fn = args.bam_fn
    can_fn = args.can_fn
    threshold = args.threshold
    minCoverage = args.minCoverage
    ctgName = args.ctgName
    ctgStart = args.ctgStart
    ctgEnd = args.ctgEnd
    samtools = args.samtools
    ref_fn = args.ref_fn

    ref_fp = open(ref_fn, 'r')
    refSeq = None
    for name, seq, qual in readfq(ref_fp):
        if name != ctgName:
            continue
        refSeq = seq 
        break

    if refSeq == None:
        print >> sys.stderr, "Cannot find reference sequence %s" % (ctgName)
        sys.exit(1)

    p = subprocess.Popen(shlex.split("%s view %s %s:%d-%d" % (samtools, bam_fn, ctgName, ctgStart, ctgEnd) ), stdout=subprocess.PIPE, bufsize=8388608)\
        if ctgStart and ctgEnd\
        else subprocess.Popen(shlex.split("%s view %s %s" % (samtools, bam_fn, ctgName) ), stdout=subprocess.PIPE, bufsize=8388608)

    pileup = {}
    sweep = 0

    can_fp = open(can_fn, "w")

    for l in p.stdout:
        l = l.strip().split()
        if l[0][0] == "@":
            continue

        QNAME = l[0]
        RNAME = l[2]
        if RNAME != ctgName:
            continue

        FLAG = int(l[1])
        POS = int(l[3]) - 1 # switch from 1-base to 0-base to match sequence index 
        CIGAR = l[5]
        SEQ = l[9]
        refPos = POS
        queryPos = 0

        skipBase = 0
        totalAlnPos = 0
        for m in re.finditer(cigarRe, CIGAR):
            advance = int(m.group(1))
            totalAlnPos += advance
            if m.group(2)  == "S":
                skipBase += advance

        if 1.0 - float(skipBase) / (totalAlnPos + 1) < 0.55: # skip a read less than 55% aligned
            continue

        for m in re.finditer(cigarRe, CIGAR):
            advance = int(m.group(1))
            if m.group(2) == "S":
                queryPos += advance
                continue
            if m.group(2) in ("M", "=", "X"):
                matches = []
                for i in range(advance):
                    matches.append( (refPos, SEQ[queryPos]) )
                    refPos += 1
                    queryPos += 1
                for pos, base in matches:
                    pileup.setdefault(pos, {"A":0, "C":0, "G":0, "T":0, "N": 0})
                    pileup[pos][base] += 1
            elif m.group(2) == "I":
                for i in range(advance):
                    queryPos += 1
            elif m.group(2) == "D":
                for i in range(advance):
                    refPos += 1

        while sweep < POS:
            flag = pileup.get(sweep)
            if flag is None:
                sweep += 1
                continue
            baseCount = pileup[sweep].items()
            refBase = refSeq[sweep]
            out = OutputCandidate(ctgName, sweep, baseCount, refBase, minCoverage, threshold)
            if out != None:
                totalCount, outline = out
                print >> can_fp, outline
            del pileup[sweep]
            sweep += 1;

    # check remaining bases
    remainder = pileup.keys()
    remainder.sort()
    for pos in remainder:
        baseCount = pileup[pos].items()
        refBase = refSeq[pos]
        out = OutputCandidate(ctgName, pos, baseCount, refBase, minCoverage, threshold)
        if out != None:
            totalCount, outline = out
            print >> can_fp, outline


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate variant candidates using alignments")

    parser.add_argument('--bam_fn', type=str, default="input.bam",
            help="Sorted bam file input, default: %(default)s")

    parser.add_argument('--ref_fn', type=str, default="ref.fa",
            help="Reference fasta file input, default: %(default)s")

    parser.add_argument('--can_fn', type=str, default="pileup.out",
            help="Pile-up count output, default: %(default)s")

    parser.add_argument('--threshold', type=float, default=0.125,
            help="Minimum allele frequence of the 1st non-reference allele for a site to be considered as a condidate site, default: %(default)f")

    parser.add_argument('--minCoverage', type=float, default=4,
            help="Minimum coverage required to call a variant, default: %(default)d")

    parser.add_argument('--ctgName', type=str, default="chr17",
            help="The name of sequence to be processed, default: %(default)s")

    parser.add_argument('--ctgStart', type=int, default=None,
            help="The 1-bsae starting position of the sequence to be processed")

    parser.add_argument('--ctgEnd', type=int, default=None,
            help="The inclusive ending position of the sequence to be processed")

    parser.add_argument('--samtools', type=str, default="samtools",
            help="Path to the 'samtools', default: %(default)s")

    args = parser.parse_args()

    MakeCandidates(args)

