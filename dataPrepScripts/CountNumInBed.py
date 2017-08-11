import os
home_dir = os.path.expanduser('~')
import sys
sys.path.append(home_dir+'/miniconda2/lib/python2.7/site-packages')
import intervaltree
import argparse
import logging

logging.basicConfig(format='%(message)s', level=logging.INFO)

def Run(args):
    Calc(args)


def Calc(args):

    logging.info("Loading BED file ...")
    tree = {}
    with open(args.bed_fn) as f:
        for row in f:
            row = row.strip().split()
            name = row[0]
            if name not in tree:
                tree[name] = intervaltree.IntervalTree()
            begin = int(row[1])
            end = int(row[2])
            tree[name].addi(begin, end)

    logging.info("Counting number of records in bed regions ...")

    a = 0
    o = 0
    with open(args.input_fn) as f:
        for row in f:
            a += 1
            row = row.strip().split()
            ctgName = row[0]
            pos = int(row[1])
            if ctgName not in tree:
                continue
            if len(tree[ctgName].search(pos)) == 0:
                continue
            o += 1

    logging.info("Total: %d, Overlapped: %d, Percentage: %.3f" % (a, o, float(o)/a*100) )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Calculate the number of variants in the bed regions" )

    parser.add_argument('--input_fn', type=str, default = None,
            help="Input with 1st column as contig name and 2nd column as position")

    parser.add_argument('--bed_fn', type=str, default = None,
            help="High confident genome regions input in the BED format")

    args = parser.parse_args()

    Run(args)

