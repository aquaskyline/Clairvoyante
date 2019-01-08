import os
import sys
import subprocess
import intervaltree
import shlex
import argparse
import param

majorContigs = {"chr"+str(a) for a in range(0,23)+["X", "Y"]}.union({str(a) for a in range(0,23)+["X", "Y"]})

def CheckFileExist(fn, sfx=""):
    if not os.path.isfile(fn+sfx):
        sys.exit("Error: %s not found" % (fn+sfx))
    return os.path.abspath(fn)

def CheckCmdExist(cmd):
    try:
        subprocess.check_output("which %s" % (cmd), shell=True)
    except:
        sys.exit("Error: %s executable not found" % (cmd))
    return cmd

def Run(args):
    basedir = os.path.dirname(__file__)
    callVarBamBin = CheckFileExist(basedir + "/callVarBam.py")
    pypyBin = CheckCmdExist(args.pypy)
    samtoolsBin = CheckCmdExist(args.samtools)
    chkpnt_fn = CheckFileExist(args.chkpnt_fn, sfx=".meta")
    bam_fn = CheckFileExist(args.bam_fn)
    ref_fn = CheckFileExist(args.ref_fn)
    fai_fn = CheckFileExist(args.ref_fn + ".fai")
    bed_fn = CheckFileExist(args.bed_fn) if args.bed_fn != None else None
    vcf_fn = "--vcf_fn %s" % (CheckFileExist(args.vcf_fn)) if args.vcf_fn != None else ""
    output_prefix = args.output_prefix
    threshold = args.threshold
    minCoverage = args.minCoverage
    sampleName = args.sampleName
    delay = args.delay
    threads = args.tensorflowThreads
    if args.considerleftedge:
        considerleftedge = "--considerleftedge"
    else:
        considerleftedge = ""
    if args.qual:
        qual = "--qual %d" % (args.qual)
    else:
        qual = ""
    includingAllContigs = args.includingAllContigs
    refChunkSize = args.refChunkSize

    tree = {}
    if bed_fn != None:
        bed_fp = subprocess.Popen(shlex.split("gzip -fdc %s" % (bed_fn) ), stdout=subprocess.PIPE, bufsize=8388608)
        for row in bed_fp.stdout:
            row = row.strip().split()
            name = row[0]
            if name not in tree:
                tree[name] = intervaltree.IntervalTree()
            begin = int(row[1])
            end = int(row[2])-1
            if end == begin: end += 1
            tree[name].addi(begin, end)
        bed_fp.stdout.close()
        bed_fp.wait()

    fai_fp = open(fai_fn)
    for line in fai_fp:

        fields = line.strip().split("\t")

        chromName = fields[0]
        if includingAllContigs == False and str(chromName) not in majorContigs:
            continue
        regionStart = 0
        chromLength = int(fields[1])

        while regionStart < chromLength:
            start = regionStart
            end = regionStart + refChunkSize
            if end > chromLength:
                end = chromLength
            output_fn = "%s.%s_%d_%d.vcf" % (output_prefix, chromName, regionStart, end)
            if bed_fn != None:
                if chromName in tree:
                    if len(tree[chromName].search(start, end)) != 0:
                        print("python %s --chkpnt_fn %s --ref_fn %s --bam_fn %s --bed_fn %s --ctgName %s --ctgStart %d --ctgEnd %d --call_fn %s --threshold %f --minCoverage %f --pypy %s --samtools %s --delay %d --threads %d --sampleName %s %s %s %s" % (callVarBamBin, chkpnt_fn, ref_fn, bam_fn, bed_fn, chromName, regionStart, end, output_fn, threshold, minCoverage, pypyBin, samtoolsBin, delay, threads, sampleName, vcf_fn, considerleftedge, qual) )
            else:
                print("python %s --chkpnt_fn %s --ref_fn %s --bam_fn %s --ctgName %s --ctgStart %d --ctgEnd %d --call_fn %s --threshold %f --minCoverage %f --pypy %s --samtools %s --delay %d --threads %d --sampleName %s %s %s %s" % (callVarBamBin, chkpnt_fn, ref_fn, bam_fn, chromName, regionStart, end, output_fn, threshold, minCoverage, pypyBin, samtoolsBin, delay, threads, sampleName, vcf_fn, considerleftedge, qual) )
            regionStart = end


def main():
    parser = argparse.ArgumentParser(
            description="Create commands for calling variants in parallel using a trained Clairvoyante model and a BAM file" )

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a Clairvoyante model")

    parser.add_argument('--ref_fn', type=str, default="ref.fa",
            help="Reference fasta file input, default: %(default)s")

    parser.add_argument('--bed_fn', type=str, default=None,
            help="Call variant only in these regions, optional, default: whole genome")

    parser.add_argument('--refChunkSize', type=int, default=10000000,
            help="Divide job with smaller genome chunk size for parallelism, default: %(default)s")

    parser.add_argument('--bam_fn', type=str, default="bam.bam",
            help="BAM file input, default: %(default)s")

    parser.add_argument('--vcf_fn', type=str, default=None,
            help="Candidate sites VCF file input, if provided, variants will only be called at the sites in the VCF file,  default: %(default)s")

    parser.add_argument('--output_prefix', type=str, default = None,
            help="Output prefix")

    parser.add_argument('--includingAllContigs', type=param.str2bool, nargs='?', const=True, default=False,
            help="Call variants on all contigs, default: chr{1..22,X,Y,M,MT} and {1..22,X,Y,MT}")

    parser.add_argument('--tensorflowThreads', type=int, default = 4,
            help="Number of threads per tensorflow job, default: %(default)s")

    parser.add_argument('--threshold', type=float, default=0.2,
            help="Minimum allele frequence of the 1st non-reference allele for a site to be considered as a condidate site, default: %(default)f")

    parser.add_argument('--minCoverage', type=float, default=4,
            help="Minimum coverage required to call a variant, default: %(default)d")

    parser.add_argument('--qual', type=int, default = None,
            help="If set, variant with equal or higher quality will be marked PASS, or LowQual otherwise, optional")

    parser.add_argument('--sampleName', type=str, default = "SAMPLE",
            help="Define the sample name to be shown in the VCF file")

    parser.add_argument('--considerleftedge', type=param.str2bool, nargs='?', const=True, default=True,
            help="Count the left-most base-pairs of a read for coverage even if the starting position of a read is after the starting position of a tensor, default: %(default)s")

    parser.add_argument('--samtools', type=str, default="samtools",
            help="Path to the 'samtools', default: %(default)s")

    parser.add_argument('--pypy', type=str, default="pypy",
            help="Path to the 'pypy', default: %(default)s")

    parser.add_argument('--delay', type=int, default = 10,
            help="Wait a short while for no more than %(default)s to start the job. This is to avoid starting multiple jobs simultaneously that might use up the maximum number of threads allowed, because Tensorflow will create more threads than needed at the beginning of running the program.")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)


if __name__ == "__main__":
    main()
