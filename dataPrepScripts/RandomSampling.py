import os
import sys
import argparse
import param
import intervaltree
import random

class CandidateStdout(object):
    def __init__(self, handle):
        self.stdin = handle

    def __del__(self):
        self.stdin.close()


def MakeCandidates( args ):
    fai_fn = "%s.fai" % (args.ref_fn)
    if os.path.isfile(fai_fn) == False:
        print >> sys.stderr, "Fasta index %s.fai doesn't exist." % (args.ref_fn)
        sys.exit(1)

    if args.ctgName == None:
        print >> sys.stderr, "Please define --ctgName. Exiting ..."
        sys.exit(1)

    start = 1; end = -1
    with open(fai_fn, "r") as f:
        for l in f:
            s = l.strip().split()
            if s[0] == args.ctgName:
                end = int(s[1])

    if end == -1:
       print >> sys.stderr, "Chromosome %s not found in %s" % (args.ctgName, fai_fn)

    if args.ctgEnd != None and args.ctgEnd < end:
       end = args.ctgEnd
    if args.ctgStart != None and args.ctgStart > start:
       start = args.ctgStart

    tree = {}
    if args.bed_fn != None:
        import subprocess
        import shlex
        f = subprocess.Popen(shlex.split("gzip -fdc %s" % (args.bed_fn) ), stdout=subprocess.PIPE, bufsize=8388608)
        for row in f.stdout:
            row = row.strip().split()
            name = row[0]
            if name != args.ctgName:
                continue
            if name not in tree:
                tree[name] = intervaltree.IntervalTree()
            begin = int(row[1])
            last = int(row[2])-1
            if last == begin: last += 1
            tree[name].addi(begin, last)
        f.stdout.close()
        f.wait()
        if args.ctgName not in tree:
            print >> sys.stderr, "ctgName is not in the bed file, are you using the correct bed file (%s)?" % (args.bed_fn)
            sys.exit(1)

    args.outputProb = (args.candidates * 2.) / (args.genomeSize)
    for i in xrange(start, end, 1):
        if args.bed_fn != None and len(tree[args.ctgName].search(i)) == 0:
            continue
        if random.uniform(0, 1) <= args.outputProb:
            print >> sys.stdout, "%s\t%d" % (args.ctgName, i)

    if args.can_fn != "PIPE":
        can_fp.stdin.close()
        can_fp.wait()
        can_fpo.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate variant candidates using alignments")


    parser.add_argument('--ref_fn', type=str, default="ref.fa",
            help="Reference fasta file input, mandatory, default: %(default)s")

    parser.add_argument('--ctgName', type=str, default=None,
            help="The name of the sequence to be processed, mandatory, default: %(default)s")

    parser.add_argument('--can_fn', type=str, default="PIPE",
            help="Randomly sampled genome position output, use PIPE for standard output, optional, default: %(default)s")

    parser.add_argument('--candidates', type=int, default=7000000,
            help="For the whole genome, the number of variant candidates to be generated, optional, default: %(default)s")

    parser.add_argument('--genomeSize', type=int, default=3000000000,
            help="The size of the genome, optional, default: %(default)s")

    parser.add_argument('--bed_fn', type=str, default=None,
            help="Generate positions only in these regions, works in intersection with ctgName, ctgStart and ctgEnd, optional, default: as defined by ctgName, ctgStart and ctgEnd")

    parser.add_argument('--ctgStart', type=int, default=None,
            help="The 1-bsae starting position of the sequence to be processed, optional")

    parser.add_argument('--ctgEnd', type=int, default=None,
            help="The inclusive ending position of the sequence to be processed, optional")


    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    MakeCandidates(args)

