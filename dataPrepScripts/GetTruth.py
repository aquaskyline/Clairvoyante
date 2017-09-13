import sys
import argparse
import subprocess
import shlex

def OutputVariant( args ):
    var_fn = args.var_fn
    vcf_fn = args.vcf_fn
    ctgName = args.ctgName

    var_fpo = open(var_fn, "wb")
    var_fp = subprocess.Popen(shlex.split("gzip -c" ), stdin=subprocess.PIPE, stdout=var_fpo, stderr=sys.stderr, bufsize=8388608)
    vcf_fp = subprocess.Popen(shlex.split("gzip -fdc %s" % (vcf_fn) ), stdout=subprocess.PIPE, bufsize=8388608)
    for row in vcf_fp.stdout:
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
        var_fp.stdin.write(" ".join([row[0], row[1], row[3], row[4], str(p1), str(p2), "\n"]))
    var_fp.stdin.close()
    var_fp.wait()
    vcf_fp.stdout.close()
    vcf_fp.wait()
    var_fpo.close()

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

