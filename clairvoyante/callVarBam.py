import os
import sys
import argparse
import param
import shlex
import subprocess
import multiprocessing
import signal
import random
import time

class InstancesClass(object):
    def __init__(self):
        self.EVCInstance = None
        self.CTInstance = None
        self.CVInstance = None

    def poll(self):
        self.EVCInstance.poll()
        self.CTInstance.poll()
        self.CVInstance.poll()

c = InstancesClass();


def CheckRtCode(signum, frame):
    c.poll()
    #print >> sys.stderr, c.EVCInstance.returncode, c.CTInstance.returncode, c.CVInstance.returncode
    if c.EVCInstance.returncode != None and c.EVCInstance.returncode != 0:
        c.CTInstance.kill(); c.CVInstance.kill()
        sys.exit("ExtractVariantCandidates.py or GetTruth.py exited with exceptions. Exiting...");

    if c.CTInstance.returncode != None and c.CTInstance.returncode != 0:
        c.EVCInstance.kill(); c.CVInstance.kill()
        sys.exit("CreateTensors.py exited with exceptions. Exiting...");

    if c.CVInstance.returncode != None and c.CVInstance.returncode != 0:
        c.EVCInstance.kill(); c.CTInstance.kill()
        sys.exit("callVar.py exited with exceptions. Exiting...");

    if c.EVCInstance.returncode == None or c.CTInstance.returncode == None or c.CVInstance.returncode == None:
        signal.alarm(5)


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
    EVCBin = CheckFileExist(basedir + "/../dataPrepScripts/ExtractVariantCandidates.py")
    GTBin = CheckFileExist(basedir + "/../dataPrepScripts/GetTruth.py")
    CTBin = CheckFileExist(basedir + "/../dataPrepScripts/CreateTensor.py")
    CVBin = CheckFileExist(basedir + "/callVar.py")
    pypyBin = CheckCmdExist(args.pypy)
    samtoolsBin = CheckCmdExist(args.samtools)
    chkpnt_fn = CheckFileExist(args.chkpnt_fn, sfx=".meta")
    bam_fn = CheckFileExist(args.bam_fn)
    ref_fn = CheckFileExist(args.ref_fn)
    if args.bed_fn == None:
        bed_fn = ""
    else:
        bed_fn = CheckFileExist(args.bed_fn)
        bed_fn = "--bed_fn %s" % (bed_fn)
    vcf_fn = None
    if args.vcf_fn != None:
        vcf_fn = CheckFileExist(args.vcf_fn)
    call_fn = args.call_fn
    threshold = args.threshold
    minCoverage = args.minCoverage
    sampleName = args.sampleName
    ctgName = args.ctgName
    if ctgName == None:
        sys.exit("--ctgName must be specified. You can call variants on multiple chromosomes simultaneously.")
    if args.considerleftedge:
        considerleftedge = "--considerleftedge"
    else:
        considerleftedge = ""
    if args.qual:
        qual = "--qual %d" % (args.qual)
    else:
        qual = ""
    if args.ctgStart != None and args.ctgEnd != None and int(args.ctgStart) <= int(args.ctgEnd):
        ctgRange = "--ctgStart %s --ctgEnd %s" % (args.ctgStart, args.ctgEnd)
    else:
        ctgRange = ""
    dcov = args.dcov

    maxCpus = multiprocessing.cpu_count()
    if args.threads == None: numCpus = multiprocessing.cpu_count()
    else: numCpus = args.threads if args.threads < multiprocessing.cpu_count() else multiprocessing.cpu_count()
    cpuSet = ",".join(str(x) for x in random.sample(xrange(0, maxCpus), numCpus))
    taskSet = "taskset -c %s" % cpuSet
    try:
        subprocess.check_output("which %s" % ("taskset"), shell=True)
    except:
        taskSet = ""

    if args.delay > 0:
        delay = random.randrange(0, args.delay)
        print >> sys.stderr, "Delay %d seconds before starting variant calling ..." % (delay)
        time.sleep(delay)

    try:
        if vcf_fn == None:
            c.EVCInstance = subprocess.Popen(\
                shlex.split("%s %s --bam_fn %s --ref_fn %s %s --ctgName %s %s --threshold %s --minCoverage %s --samtools %s" %\
                            (pypyBin, EVCBin, bam_fn, ref_fn, bed_fn, ctgName, ctgRange, threshold, minCoverage, samtoolsBin) ),\
                            stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=8388608)
        else:
            c.EVCInstance = subprocess.Popen(\
                shlex.split("%s %s --vcf_fn %s --ctgName %s %s" %\
                            (pypyBin, GTBin, vcf_fn, ctgName, ctgRange) ),\
                            stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=8388608)
        c.CTInstance = subprocess.Popen(\
            shlex.split("%s %s --bam_fn %s --ref_fn %s --ctgName %s %s %s --samtools %s --dcov %d" %\
                        (pypyBin, CTBin, bam_fn, ref_fn, ctgName, ctgRange, considerleftedge, samtoolsBin, dcov) ),\
                        stdin=c.EVCInstance.stdout, stdout=subprocess.PIPE, stderr=sys.stderr, bufsize=8388608)
        c.CVInstance = subprocess.Popen(\
            shlex.split("%s python %s --chkpnt_fn %s --call_fn %s --sampleName %s --threads %d --ref_fn %s %s" %\
                        (taskSet, CVBin, chkpnt_fn, call_fn, sampleName, numCpus, ref_fn, qual) ),\
                        stdin=c.CTInstance.stdout, stdout=sys.stderr, stderr=sys.stderr, bufsize=8388608)
    except Exception as e:
        print >> sys.stderr, e
        sys.exit("Failed to start required processes. Exiting...")

    signal.signal(signal.SIGALRM, CheckRtCode)
    signal.alarm(2)

    c.CVInstance.wait()
    c.CTInstance.stdout.close()
    c.CTInstance.wait()
    c.EVCInstance.stdout.close()
    c.EVCInstance.wait()


def main():
    parser = argparse.ArgumentParser(
            description="Call variants using a trained Clairvoyante model and a BAM file" )

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a Clairvoyante model")

    parser.add_argument('--ref_fn', type=str, default="ref.fa",
            help="Reference fasta file input, default: %(default)s")

    parser.add_argument('--bed_fn', type=str, default=None,
            help="Call variant only in these regions, works in intersection with ctgName, ctgStart and ctgEnd, optional, default: as defined by ctgName, ctgStart and ctgEnd")

    parser.add_argument('--bam_fn', type=str, default="bam.bam",
            help="BAM file input, default: %(default)s")

    parser.add_argument('--call_fn', type=str, default = None,
            help="Output variant predictions")

    parser.add_argument('--vcf_fn', type=str, default=None,
            help="Candidate sites VCF file input, if provided, variants will only be called at the sites in the VCF file,  default: %(default)s")

    parser.add_argument('--threshold', type=float, default=0.125,
            help="Minimum allele frequence of the 1st non-reference allele for a site to be considered as a condidate site, default: %(default)f")

    parser.add_argument('--minCoverage', type=float, default=4,
            help="Minimum coverage required to call a variant, default: %(default)d")

    parser.add_argument('--qual', type=int, default = None,
            help="If set, variant with equal or higher quality will be marked PASS, or LowQual otherwise, optional")

    parser.add_argument('--sampleName', type=str, default = "SAMPLE",
            help="Define the sample name to be shown in the VCF file")

    parser.add_argument('--ctgName', type=str, default=None,
            help="The name of sequence to be processed, default: %(default)s")

    parser.add_argument('--ctgStart', type=int, default=None,
            help="The 1-bsae starting position of the sequence to be processed")

    parser.add_argument('--ctgEnd', type=int, default=None,
            help="The inclusive ending position of the sequence to be processed")

    parser.add_argument('--considerleftedge', type=param.str2bool, nargs='?', const=True, default=True,
            help="Count the left-most base-pairs of a read for coverage even if the starting position of a read is after the starting position of a tensor, default: %(default)s")

    parser.add_argument('--dcov', type=int, default=250,
            help="Cap depth per position at %(default)s")

    parser.add_argument('--samtools', type=str, default="samtools",
            help="Path to the 'samtools', default: %(default)s")

    parser.add_argument('--pypy', type=str, default="pypy",
            help="Path to the 'pypy', default: %(default)s")

    parser.add_argument('--v3', type=param.str2bool, nargs='?', const=True, default = True,
            help="Use Clairvoyante version 3")

    parser.add_argument('--v2', type=param.str2bool, nargs='?', const=True, default = False,
            help="Use Clairvoyante version 2")

    parser.add_argument('--slim', type=param.str2bool, nargs='?', const=True, default = False,
            help="Train using the slim version of Clairvoyante, optional")

    parser.add_argument('--threads', type=int, default = None,
            help="Number of threads, optional")

    parser.add_argument('--delay', type=int, default = 10,
            help="Wait a short while for no more than %(default)s to start the job. This is to avoid starting multiple jobs simultaneously that might use up the maximum number of threads allowed, because Tensorflow will create more threads than needed at the beginning of running the program.")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)


if __name__ == "__main__":
    main()
