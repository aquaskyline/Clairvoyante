import os
home_dir = os.path.expanduser('~')
import sys
sys.path.append(home_dir+'/miniconda2/lib/python2.7/site-packages')
import intervaltree
import argparse

def Run(args):
    Calc(args)


def Calc(args):

    tree = {}
    with open(args.bed_fn) as f:
        for row in f:
            row = row.split()
            name = row[0]
            if name not in tree:
                tree[name] = intervaltree.IntervalTree()
            begin = int(row[1])
            end = int(row[2])
            tree[name].addi(begin, end)


    with open(args.input_fn) as f:
        for row in f:
            ctgName, pos = [(row.split()[i]) for i in [0,1]]
            pos = int(pos)
            if ctgName not in tree:
                continue
            if len(tree[ctgName].search(pos)) == 0:
                continue
            sys.stdout.write(row)
    sys.stdout.flush()


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

