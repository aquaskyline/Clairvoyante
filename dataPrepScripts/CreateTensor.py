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
import param

cigarRe = r"(\d+)([MIDNSHP=X])"
base2num = dict(zip("ACGT", (0,1,2,3)))
stripe2 = 4 * param.matrixNum
stripe1 = param.matrixNum

def GenerateTensor(ctgName, alns, center, refSeq):
    alnCode = [0] * ( (2*param.flankingBaseNum+1) * 4 * param.matrixNum )
    for aln in alns:
        for refPos, queryAdv, refBase, queryBase in aln:
            if str(refBase) not in "ACGT-":
                continue
            if str(queryBase) not in "ACGT-":
                continue
            if refPos - center >= -(param.flankingBaseNum+1) and refPos - center < param.flankingBaseNum:
                offset = refPos - center + (param.flankingBaseNum+1)
                if queryBase != "-":
                    if refBase != "-":
                        alnCode[stripe2*offset + stripe1*base2num[refBase] + 0] += 1.0
                        alnCode[stripe2*offset + stripe1*base2num[queryBase] + 1] += 1.0
                        alnCode[stripe2*offset + stripe1*base2num[refBase] + 2] += 1.0
                        alnCode[stripe2*offset + stripe1*base2num[queryBase] + 3] += 1.0
                    elif refBase == "-":
                        idx = min(offset+queryAdv, 2*param.flankingBaseNum+1-1)
                        alnCode[stripe2*idx + stripe1*base2num[queryBase] + 1] += 1.0
                    else:
                      print >> sys.stderr, "Should not reach here: %s, %s" % (refBase, queryBase)
                elif queryBase == "-":
                    if refBase != "-":
                        alnCode[stripe2*offset + stripe1*base2num[refBase] + 2] += 1.0
                    else:
                        print >> sys.stderr, "Should not reach here: %s, %s" % (refBase, queryBase)
                else:
                    print >> sys.stderr, "Should not reach here: %s, %s" % (refBase, queryBase)

    outputLine = "%s %d %s" % (ctgName, center, refSeq[center-(param.flankingBaseNum+1):center+param.flankingBaseNum])
    for x in alnCode:
        outputLine += " %0.1f" % x
    return outputLine

def GetCandidate(args, beginToEnd):
    if args.can_fn != "PIPE":
        f = subprocess.Popen(shlex.split("gzip -fdc %s" % (args.can_fn) ), stdout=subprocess.PIPE, bufsize=8388608)
        fo = f.stdout
    else:
        fo = sys.stdin
    for row in fo:
        row = row.split()
        pos = int(row[1])
        if args.ctgStart != None and pos < args.ctgStart: continue
        if args.ctgEnd != None and pos > args.ctgEnd: continue
        if args.considerleftedge == False:
            beginToEnd[ pos - (param.flankingBaseNum+1) ] = (pos + (param.flankingBaseNum+1), pos)
        elif args.considerleftedge == True:
            for i in range(pos - (param.flankingBaseNum+1), pos + (param.flankingBaseNum+1)):
                beginToEnd[ i ] = (pos + (param.flankingBaseNum+1), pos)
        yield pos

    if args.can_fn != "PIPE":
        fo.close()
        f.wait()
    yield -1


class TensorStdout(object):
    def __init__(self, handle):
        self.stdin = handle

    def __del__(self):
        self.stdin.close()


def OutputAlnTensor(args):
    refSeq = None
    ref_fp = open(args.ref_fn, 'r')
    for name, refSeq, _ in readfq(ref_fp):
        if name != args.ctgName:
            continue
        break

    if refSeq == None:
        print >> sys.stderr, "Cannot find reference sequence %s" % (args.ctgName)
        sys.exit(1)

    beginToEnd = {}
    canPos = 0
    canGen = GetCandidate(args, beginToEnd)

    p = subprocess.Popen(shlex.split("%s view %s %s:%d-%d" % (args.samtools, args.bam_fn, args.ctgName, args.ctgStart, args.ctgEnd) ), stdout=subprocess.PIPE, bufsize=8388608)\
        if args.ctgStart and args.ctgEnd\
        else subprocess.Popen(shlex.split("%s view %s %s" % (args.samtools, args.bam_fn, args.ctgName) ), stdout=subprocess.PIPE, bufsize=8388608)

    centerToAln = {}

    if args.tensor_fn != "PIPE":
        tensor_fpo = open(args.tensor_fn, "wb")
        tensor_fp = subprocess.Popen(shlex.split("gzip -c"), stdin=subprocess.PIPE, stdout=tensor_fpo, stderr=sys.stderr, bufsize=8388608)
    else:
        tensor_fp = TensorStdout(sys.stdout)

    for l in p.stdout:
        l = l.split()
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

        while canPos != -1 and canPos < (POS + len(SEQ) + 100000):
            canPos = next(canGen)

        for m in re.finditer(cigarRe, CIGAR):
            advance = int(m.group(1))
            if m.group(2) == "S":
                queryPos += advance
            if m.group(2) in ("M", "=", "X"):
                matches = []
                for i in xrange(advance):
                    matches.append( (refPos, SEQ[queryPos]) )
                    if refPos in beginToEnd:
                        rEnd, rCenter = beginToEnd[refPos]
                        if rCenter not in activeSet:
                            endToCenter[rEnd] = rCenter
                            activeSet.add(rCenter)
                            centerToAln.setdefault(rCenter, [])
                            centerToAln[rCenter].append([])
                    for center in list(activeSet):
                        centerToAln[center][-1].append( (refPos, 0, refSeq[refPos], SEQ[queryPos] ) )
                    if refPos in endToCenter:
                        center = endToCenter[refPos]
                        activeSet.remove(center)
                    refPos += 1
                    queryPos += 1

            elif m.group(2) == "I":
                queryAdv = 0
                for i in range(advance):
                    for center in list(activeSet):
                        centerToAln[center][-1].append( (refPos, queryAdv, "-", SEQ[queryPos] ))
                    queryPos += 1
                    queryAdv += 1

            elif m.group(2) == "D":
                for i in xrange(advance):
                    for center in list(activeSet):
                        centerToAln[center][-1].append( (refPos, 0, refSeq[refPos], "-" ))
                    if refPos in beginToEnd:
                        rEnd, rCenter = beginToEnd[refPos]
                        if rCenter not in activeSet:
                            endToCenter[rEnd] = rCenter
                            activeSet.add(rCenter)
                            centerToAln.setdefault(rCenter, [])
                            centerToAln[rCenter].append([])
                    if refPos in endToCenter:
                        center = endToCenter[refPos]
                        activeSet.remove(center)
                    refPos += 1


        for center in centerToAln.keys():
            if center + (param.flankingBaseNum+1) < POS:
                l =  GenerateTensor(args.ctgName, centerToAln[center], center, refSeq)
                tensor_fp.stdin.write(l)
                tensor_fp.stdin.write("\n")
                del centerToAln[center]

    for center in centerToAln.keys():
        l =  GenerateTensor(args.ctgName, centerToAln[center], center, refSeq)
        tensor_fp.stdin.write(l)
        tensor_fp.stdin.write("\n")

    p.stdout.close()
    p.wait()
    if args.tensor_fn != "PIPE":
        tensor_fp.stdin.close()
        tensor_fp.wait()
        tensor_fpo.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Generate tensors summarizing local alignments from a BAM file and a list of candidate locations" )

    parser.add_argument('--bam_fn', type=str, default="input.bam",
            help="Sorted bam file input, default: %(default)s")

    parser.add_argument('--ref_fn', type=str, default="ref.fa",
            help="Reference fasta file input, default: %(default)s")

    parser.add_argument('--can_fn', type=str, default="PIPE",
            help="Variant candidate list generated by ExtractVariantCandidates.py or true variant list generated by GetTruth.py, use PIPE for standard input, default: %(default)s")

    parser.add_argument('--tensor_fn', type=str, default="PIPE",
            help="Tensor output, use PIPE for standard output, default: %(default)s")

    parser.add_argument('--ctgName', type=str, default="chr17",
            help="The name of sequence to be processed, default: %(default)s")

    parser.add_argument('--ctgStart', type=int, default=None,
            help="The 1-bsae starting position of the sequence to be processed")

    parser.add_argument('--ctgEnd', type=int, default=None,
            help="The inclusive ending position of the sequence to be processed")

    parser.add_argument('--samtools', type=str, default="samtools",
            help="Path to the 'samtools', default: %(default)s")

    parser.add_argument('--considerleftedge', type=param.str2bool, nargs='?', const=False, default=False,
            help="Count the left-most base-pairs of a read for coverage even if the starting position of a read is after the starting position of a tensor, default: %(default)s")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    OutputAlnTensor(args)

