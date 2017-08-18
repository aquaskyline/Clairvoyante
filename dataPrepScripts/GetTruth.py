import sys
import argparse

def OutputVariant( args ):
    vcf_fn = args.vcf_fn
    var_fn = args.var_fn
    ctgName = args.ctgName

    var_fp = open(var_fn, "w")

    with open(vcf_fn, "r") as vcf_fp:
        for row in vcf_fp.readlines():
            row = row.strip().split()
            if row[0][0] == "#":
                continue
            if row[0] != ctgName:
                continue
            last = row[-1]
            varType = last.split(":")[0].replace("/","|").replace(".","0").split("|")
            p1, p2 = varType
            p1 = int(p1)
            p2 = int(p2)
            p1, p2 = (p1, p2) if p1 < p2 else (p2, p1)
            print >>  var_fp, row[0], row[1], row[3], row[4], p1, p2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Extract variant type and allele from a Truth dataset" )

    parser.add_argument('--vcf_fn', type=str, default="input.vcf",
            help="Truth vcf file input, default: %(default)s")

    parser.add_argument('--var_fn', type=str, default="var.out",
            help="Truth variants list output, default: %(default)s")

    parser.add_argument('--ctgName', type=str, default="chr17",
            help="The name of sequence to be processed, default: %(default)s")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    OutputVariant( args )

