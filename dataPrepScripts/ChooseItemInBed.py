import os
import sys
import intervaltree
import argparse
import subprocess

def Run(args):
    Calc(args)


def Calc(args):

    tree = {}
    f = subprocess.Popen(shlex.split("gzip -fdc %s" % (args.bed_fn) ), stdout=subprocess.PIPE, bufsize=8388608)
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


    f = subprocess.Popen(shlex.split("gzip -fdc %s" % (args.input_fn) ), stdout=subprocess.PIPE, bufsize=8388608)
    for row in f.stdout:
        ctgName, pos = [(row.split()[i]) for i in [0,1]]
        pos = int(pos)
        if ctgName not in tree:
            continue
        if len(tree[ctgName].search(pos)) == 0:
            continue
        sys.stdout.write(row)
    sys.stdout.flush()
    f.stdout.close()
    f.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Print the variants in the bed regions" )

    parser.add_argument('--input_fn', type=str, default = None,
            help="Input with 1st column as contig name and 2nd column as position")

    parser.add_argument('--bed_fn', type=str, default = None,
            help="High confident genome regions input in the BED format")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)

