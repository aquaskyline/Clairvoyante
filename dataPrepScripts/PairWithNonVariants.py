import os
home_dir = os.path.expanduser('~')
import sys
sys.path.append(home_dir+'/miniconda2/lib/python2.7/site-packages')
import intervaltree
import argparse
import logging
import random

logging.basicConfig(format='%(message)s', level=logging.INFO)


def Run(args):
    Pair(args)


def bufcount(filename):
    f = open(filename)
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count('\n')
        buf = read_f(buf_size)

    return lines


def Pair(args):
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

    logging.info("Counting the number of Truth Variants in %s ..." % args.tensor_var_fn)
    v = 0
    d = {}
    with open(args.tensor_var_fn) as f:
        for row in f:
            row = row.strip().split()
            ctgName = row[0]
            pos = int(row[1])
            key = "-".join([ctgName, str(pos)])
            v += 1
            d[key] = 1
    logging.info("%d Truth Variants" % v)
    t = v * args.amp
    logging.info("%d non-variants to be picked" % t)

    logging.info("Counting the number of usable non-variants in %s ..." % args.tensor_can_fn)
    c = 0
    with open(args.tensor_can_fn) as f:
        for row in f:
            row = row.strip().split()
            ctgName = row[0]
            pos = int(row[1])
            if ctgName not in tree:
                continue
            if len(tree[ctgName].search(pos)) == 0:
                continue
            key = "-".join([ctgName, str(pos)])
            if key in d:
                continue
            c += 1
    logging.info("%d usable non-variant" % c)

    r = float(t) / c
    r = r if r <= 1 else 1
    logging.info("%.2f of all non-variants are selected" % r)


    o1 = 0
    o2 = 0
    output_fh = open(args.output_fn, "w")
    with open(args.tensor_var_fn) as f:
        for row in f:
            row = row.strip()
            print >> output_fh, row
            o1 += 1
    with open(args.tensor_can_fn) as f:
        for row in f:
            rawRow = row.strip()
            row = rawRow.split()
            ctgName = row[0]
            pos = int(row[1])
            if ctgName not in tree:
                continue
            if len(tree[ctgName].search(pos)) == 0:
                continue
            key = "-".join([ctgName, str(pos)])
            if key in d:
                continue
            if random.random() < r:
                print >> output_fh, rawRow
            o2 += 1
    logging.info("%.2f/%.2f Truth Variants/Non-variants outputed" % (o1, o2))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Predict and compare using Clairvoyante" )

    parser.add_argument('--tensor_can_fn', type=str, default = None,
            help="Candidiate variant tensors input")

    parser.add_argument('--tensor_var_fn', type=str, default = None,
            help="Candidiate variant tensors input")

    parser.add_argument('--bed_fn', type=str, default = None,
            help="High confident genome regions input in the BED format")

    parser.add_argument('--output_fn', type=str, default = None,
            help="Tensors output file name")

    parser.add_argument('--amp', type=int, default = 2,
        help="Pick ((# of the Truth Variants)*amp) non-variants to pair with the Truth Variants, default: 2")

    args = parser.parse_args()

    Run(args)

