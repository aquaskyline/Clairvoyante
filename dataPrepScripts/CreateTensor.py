import os
import sys
import argparse
import os
import re
import shlex
import subprocess
import signal
import gc
import param

is_pypy = '__pypy__' in sys.builtin_module_names

def PypyGCCollect(signum, frame):
    gc.collect()
    signal.alarm(60)

cigarRe = r"(\d+)([MIDNSHP=X])"
base2num = dict(zip("ACGT", (0,1,2,3)))
stripe2 = 4 * param.matrixNum
stripe1 = param.matrixNum

def GenerateTensor(args, ctgName, alns, center, refSeq):
    alnCode = [0] * ( (2*param.flankingBaseNum+1) * 4 * param.matrixNum )
    depth = [0] * ((2 * param.flankingBaseNum + 1))
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
                        depth[offset] = depth[offset] + 1
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

    newRefPos = center - (0 if args.refStart == None else (args.refStart - 1))
    if (newRefPos - (param.flankingBaseNum+1) >= 0) and depth[param.flankingBaseNum] >= args.minCoverage:
        outputLine = "%s %d %s %s" % (ctgName, center, refSeq[newRefPos-(param.flankingBaseNum+1):newRefPos+param.flankingBaseNum], " ".join("%0.1f" % x for x in alnCode))
        return outputLine
    else:
        return None

def GetCandidate(args, beginToEnd):
    if args.can_fn != "PIPE":
        f = subprocess.Popen(shlex.split("gzip -fdc %s" % (args.can_fn) ), stdout=subprocess.PIPE, bufsize=8388608)
        fo = f.stdout
    else:
        fo = sys.stdin
    for row in fo:
        row = row.split()
        if args.ctgName != row[0]: continue
        pos = int(row[1])
        if args.ctgStart != None and pos < args.ctgStart: continue
        if args.ctgEnd != None and pos > args.ctgEnd: continue
        if args.considerleftedge == False:
            beginToEnd[ pos - (param.flankingBaseNum+1) ] = [(pos + (param.flankingBaseNum+1), pos)]
        elif args.considerleftedge == True:
            for i in range(pos - (param.flankingBaseNum+1), pos + (param.flankingBaseNum+1)):
                if i not in beginToEnd:
                    beginToEnd[ i ] = [(pos + (param.flankingBaseNum+1), pos)]
                else:
                    beginToEnd[ i ].append((pos + (param.flankingBaseNum+1), pos))
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

    availableSlots = 10000000
    dcov = args.dcov
    args.refStart = None; args.refEnd = None; refSeq = []; refName = None; rowCount = 0
    if args.ctgStart != None and args.ctgEnd != None:
        args.ctgStart += 1 # Change 0-based (BED) to 1-based (VCF and samtools faidx)
        args.refStart = args.ctgStart; args.refEnd = args.ctgEnd
        args.refStart -= param.expandReferenceRegion
        args.refStart = 1 if args.refStart < 1 else args.refStart
        args.refEnd += param.expandReferenceRegion
        p1 = subprocess.Popen(shlex.split("%s faidx %s %s:%d-%d" % (args.samtools, args.ref_fn, args.ctgName, args.refStart, args.refEnd) ), stdout=subprocess.PIPE, bufsize=8388608)
    else:
        args.ctgStart = args.ctgEnd = None
        p1 = subprocess.Popen(shlex.split("%s faidx %s %s" % (args.samtools, args.ref_fn, args.ctgName) ), stdout=subprocess.PIPE, bufsize=8388608)

    for row in p1.stdout:
        if rowCount == 0:
            refName = row.rstrip().lstrip(">")
        else:
            refSeq.append(row.rstrip())
        rowCount += 1
    refSeq = "".join(refSeq)

    p1.stdout.close()
    p1.wait()

    if p1.returncode != 0 or len(refSeq) == 0:
        print >> sys.stderr, "Failed to load reference seqeunce. Please check if the provided reference fasta %s and the ctgName %s are correct." % (args.ref_fn, args.ctgName)
        sys.exit(1)

    beginToEnd = {}
    canPos = 0
    canGen = GetCandidate(args, beginToEnd)

    p2 = subprocess.Popen(shlex.split("%s view -F 2308 %s %s:%d-%d" % (args.samtools, args.bam_fn, args.ctgName, args.ctgStart, args.ctgEnd) ), stdout=subprocess.PIPE, bufsize=8388608)\
        if args.ctgStart != None and args.ctgEnd != None\
        else subprocess.Popen(shlex.split("%s view -F 2308 %s %s" % (args.samtools, args.bam_fn, args.ctgName) ), stdout=subprocess.PIPE, bufsize=8388608)

    centerToAln = {}

    if args.tensor_fn != "PIPE":
        tensor_fpo = open(args.tensor_fn, "wb")
        tensor_fp = subprocess.Popen(shlex.split("gzip -c"), stdin=subprocess.PIPE, stdout=tensor_fpo, stderr=sys.stderr, bufsize=8388608)
    else:
        tensor_fp = TensorStdout(sys.stdout)

    #if is_pypy:
    #    signal.signal(signal.SIGALRM, PypyGCCollect)
    #    signal.alarm(60)

    previousPos = 0; depthCap = 0
    for l in p2.stdout:
        l = l.split()
        if l[0][0] == "@":
            continue

        QNAME = l[0]
        FLAG = int(l[1])
        RNAME = l[2]
        POS = int(l[3]) - 1 # switch from 1-base to 0-base to match sequence index
        MQ = int(l[4])
        CIGAR = l[5]
        SEQ = l[9]
        refPos = POS
        queryPos = 0

        if MQ < args.minMQ:
            continue

        endToCenter = {}
        activeSet = set()

        while canPos != -1 and canPos < (POS + len(SEQ) + 100000):
            canPos = next(canGen)

        if previousPos != POS:
            previousPos = POS
            depthCap = 0
        else:
            depthCap += 1
            if depthCap >= dcov:
                #print >> sys.stderr, "Bypassing POS %d at depth %d\n" % (POS, depthCap)
                continue

        for m in re.finditer(cigarRe, CIGAR):
            if availableSlots == 0:
                break
            advance = int(m.group(1))
            if m.group(2) == "S":
                queryPos += advance
            if m.group(2) in ("M", "=", "X"):
                for i in xrange(advance):
                    if refPos in beginToEnd:
                        for rEnd, rCenter in beginToEnd[refPos]:
                            if rCenter in activeSet:
                                continue
                            endToCenter[rEnd] = rCenter
                            activeSet.add(rCenter)
                            centerToAln.setdefault(rCenter, [])
                            centerToAln[rCenter].append([])
                    for center in list(activeSet):
                        if availableSlots != 0:
                            availableSlots -= 1
                            centerToAln[center][-1].append( (refPos, 0, refSeq[refPos - (0 if args.refStart == None else (args.refStart - 1))], SEQ[queryPos] ) )
                    if refPos in endToCenter:
                        center = endToCenter[refPos]
                        activeSet.remove(center)
                    refPos += 1
                    queryPos += 1

            elif m.group(2) == "I":
                queryAdv = 0
                for i in range(advance):
                    for center in list(activeSet):
                        if availableSlots != 0:
                            availableSlots -= 1
                            centerToAln[center][-1].append( (refPos, queryAdv, "-", SEQ[queryPos] ))
                    queryPos += 1
                    queryAdv += 1

            elif m.group(2) == "D":
                for i in xrange(advance):
                    for center in list(activeSet):
                        if availableSlots != 0:
                            availableSlots -= 1
                            centerToAln[center][-1].append( (refPos, 0, refSeq[refPos - (0 if args.refStart == None else (args.refStart - 1))], "-" ))
                    if refPos in beginToEnd:
                        for rEnd, rCenter in beginToEnd[refPos]:
                            if rCenter in activeSet:
                                continue
                            endToCenter[rEnd] = rCenter
                            activeSet.add(rCenter)
                            centerToAln.setdefault(rCenter, [])
                            centerToAln[rCenter].append([])
                    if refPos in endToCenter:
                        center = endToCenter[refPos]
                        activeSet.remove(center)
                    refPos += 1

        if depthCap == 0:
            for center in centerToAln.keys():
                if center + (param.flankingBaseNum+1) < POS:
                    l =  GenerateTensor(args, args.ctgName, centerToAln[center], center, refSeq)
                    if l != None:
                        tensor_fp.stdin.write(l)
                        tensor_fp.stdin.write("\n")
                    availableSlots += sum(len(i) for i in centerToAln[center])
                    #print >> sys.stderr, "POS %d: remaining slots %d" % (center, availableSlots)
                    del centerToAln[center]

    for center in centerToAln.keys():
        l =  GenerateTensor(args, args.ctgName, centerToAln[center], center, refSeq)
        if l != None:
            tensor_fp.stdin.write(l)
            tensor_fp.stdin.write("\n")

    p2.stdout.close()
    p2.wait()
    if args.tensor_fn != "PIPE":
        tensor_fp.stdin.close()
        tensor_fp.wait()
        tensor_fpo.close()


def main():
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

    parser.add_argument('--minMQ', type=int, default=0,
            help="Minimum Mapping Quality. Mapping quality lower than the setting will be filtered, default: %(default)d")

    parser.add_argument('--ctgName', type=str, default="chr17",
            help="The name of sequence to be processed, default: %(default)s")

    parser.add_argument('--ctgStart', type=int, default=None,
            help="The 1-bsae starting position of the sequence to be processed")

    parser.add_argument('--ctgEnd', type=int, default=None,
            help="The inclusive ending position of the sequence to be processed")

    parser.add_argument('--samtools', type=str, default="samtools",
            help="Path to the 'samtools', default: %(default)s")

    parser.add_argument('--considerleftedge', type=param.str2bool, nargs='?', const=True, default=True,
            help="Count the left-most base-pairs of a read for coverage even if the starting position of a read is after the starting position of a tensor, default: %(default)s")

    parser.add_argument('--dcov', type=int, default=250,
            help="Cap depth per position at %(default)d")

    parser.add_argument('--minCoverage', type=int, default=0,
                        help="Minimum coverage required to generate a tensor, default: %(default)d")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    OutputAlnTensor(args)


if __name__ == "__main__":
    main()
