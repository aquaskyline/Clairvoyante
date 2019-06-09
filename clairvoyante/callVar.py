import sys
import os
import time
import argparse
import param
import logging
import numpy as np
from threading import Thread
from math import log

logging.basicConfig(format='%(message)s', level=logging.INFO)
num2base = dict(zip((0, 1, 2, 3), "ACGT"))
base2num = dict(zip("ACGT", (0, 1, 2, 3)))
v1Type2Name = dict(zip((0, 1, 2, 3, 4), ('HET', 'HOM', 'INS', 'DEL', 'REF')))
v2Zygosity2Name = dict(zip((0, 1), ('HET', 'HOM')))
v2Type2Name = dict(zip((0, 1, 2, 3), ('REF', 'SNP', 'INS', 'DEL')))
v2Length2Name = dict(zip((0, 1, 2, 3, 4, 5), ('0', '1', '2', '3', '4', '4+')))
maxVarLength = 5
inferIndelLengthMinimumAF = 0.125

def Run(args):
    # create a Clairvoyante
    logging.info("Loading model ...")
    if args.v2 == True:
        import utils_v2 as utils
        utils.SetupEnv()
        if args.slim == True:
            import clairvoyante_v2_slim as cv
        else:
            import clairvoyante_v2 as cv
    elif args.v3 == True:
        import utils_v2 as utils # v3 network is using v2 utils
        utils.SetupEnv()
        if args.slim == True:
            import clairvoyante_v3_slim as cv
        else:
            import clairvoyante_v3 as cv
    if args.threads == None:
        if args.tensor_fn == "PIPE":
            param.NUM_THREADS = 4
    else:
        param.NUM_THREADS = args.threads
    m = cv.Clairvoyante()
    m.init()

    m.restoreParameters(os.path.abspath(args.chkpnt_fn))
    Test(args, m, utils)


def Output(args, call_fh, num, XBatch, posBatch, base, z, t, l):
    if args.v2 == True or args.v3 == True:
        if num != len(base):
          sys.exit("Inconsistent shape between input tensor and output predictions %d/%d" % (num, len(base)))
        #          --------------  ------  ------------    ------------------
        #          Base chng       Zygo.   Var type        Var length
        #          A   C   G   T   HET HOM REF SNP INS DEL 0   1   2   3   4   >=4
        #          0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
        for j in range(len(base)):
            if args.showRef == False and np.argmax(t[j]) == 0: continue
            # Get variant type, 0:REF, 1:SNP, 2:INS, 3:DEL
            varType = np.argmax(t[j])
            # Get zygosity, 0:HET, 1:HOM
            varZygosity = np.argmax(z[j])
            # Get Indel Length, 0:0, 1:1, 2:2, 3:3, 4:4, 5:>4
            varLength = np.argmax(l[j])
            # Get chromosome, coordination and reference bases with flanking param.flankingBaseNum flanking bases at coordination
            chromosome, coordination, refSeq = posBatch[j].split(":")
            # Get genotype quality
            sortVarType = np.sort(t[j])[::-1]
            sortZygosity = np.sort(z[j])[::-1]
            sortLength = np.sort(l[j])[::-1]
            qual = int(-4.343 * log((sortVarType[1]*sortZygosity[1]*sortLength[1]  + 1e-300) / (sortVarType[0]*sortZygosity[0]*sortLength[0]  + 1e-300)))
            #if qual > 999: qual = 999
            filt = "."
            if args.qual != None:
                if qual >= args.qual:
                    filt = "PASS"
                else:
                    filt = "LowQual"
            # Get possible alternative bases
            sortBase = base[j].argsort()[::-1]
            base1 = num2base[sortBase[0]]
            base2 = num2base[sortBase[1]]
            # Initialize other variables
            refBase = ""; altBase = ""; inferredIndelLength = 0; dp = 0; af = 0.; info = [];
            dp = sum(XBatch[j,param.flankingBaseNum,:,0]) + sum(XBatch[j,param.flankingBaseNum+1,:,1]) + \
                 sum(XBatch[j,param.flankingBaseNum+1,:,2]) + sum(XBatch[j,param.flankingBaseNum,:,3])
            if dp != 0:
                # For SNP
                if varType == 1 or varType == 0: # SNP or REF
                    coordination = int(coordination)
                    refBase = refSeq[param.flankingBaseNum]
                    if varType == 1: # SNP
                        altBase = base1 if base1 != refBase else base2
                        #altBase = "%s,%s" % (base1, base2)
                    elif varType == 0: # REF
                        altBase = refBase
                    af = XBatch[j,param.flankingBaseNum,base2num[altBase],3] / dp
                elif varType == 2: # INS
                    # infer the insertion length
                    if varLength == 0: varLength = 1
                    af = sum(XBatch[j,param.flankingBaseNum+1,:,1]) / dp
                    if varLength != maxVarLength:
                        for k in range(param.flankingBaseNum+1, param.flankingBaseNum+varLength+1):
                            altBase += num2base[np.argmax(XBatch[j,k,:,1])]
                    else:
                        for k in range(param.flankingBaseNum+1, 2*param.flankingBaseNum+1):
                            referenceTensor = XBatch[j,k,:,0]; insertionTensor = XBatch[j,k,:,1]
                            if k < (param.flankingBaseNum + maxVarLength) or sum(insertionTensor) >= (inferIndelLengthMinimumAF * sum(referenceTensor)):
                                inferredIndelLength += 1
                                altBase += num2base[np.argmax(insertionTensor)]
                            else:
                                break
                    coordination = int(coordination)
                    refBase = refSeq[param.flankingBaseNum]
                    # insertions longer than (param.flankingBaseNum-1) are marked SV
                    if inferredIndelLength >= param.flankingBaseNum:
                        altBase = "<INS>"
                        info.append("SVTYPE=INS")
                    else:
                        altBase = refBase + altBase
                elif varType == 3: # DEL
                    if varLength == 0: varLength = 1
                    af = sum(XBatch[j,param.flankingBaseNum+1,:,2]) / dp
                    # infer the deletion length
                    if varLength == maxVarLength:
                        for k in range(param.flankingBaseNum+1, 2*param.flankingBaseNum+1):
                            if k < (param.flankingBaseNum + maxVarLength) or sum(XBatch[j,k,:,2]) >= (inferIndelLengthMinimumAF * sum(XBatch[j,k,:,0])):
                                inferredIndelLength += 1
                            else:
                               break
                    # deletions longer than (param.flankingBaseNum-1) are marked SV
                    coordination = int(coordination)
                    if inferredIndelLength >= param.flankingBaseNum:
                        refBase = refSeq[param.flankingBaseNum]
                        altBase = "<DEL>"
                        info.append("SVTYPE=DEL")
                    elif varLength != maxVarLength:
                        refBase = refSeq[param.flankingBaseNum:param.flankingBaseNum+varLength+1]
                        altBase = refSeq[param.flankingBaseNum]
                    else:
                        refBase = refSeq[param.flankingBaseNum:param.flankingBaseNum+inferredIndelLength+1]
                        altBase = refSeq[param.flankingBaseNum]
                if inferredIndelLength > 0 and inferredIndelLength < param.flankingBaseNum: info.append("LENGUESS=%d" % inferredIndelLength)
                infoStr = ""
                if len(info) == 0: infoStr = "."
                else: infoStr = ";".join(info)
                gtStr = ""
                if varType == 0: gtStr = "0/0"
                elif varZygosity == 0: gtStr = "0/1"
                elif varZygosity == 1: gtStr = "1/1"

                print >> call_fh, "%s\t%d\t.\t%s\t%s\t%d\t%s\t%s\tGT:GQ:DP:AF\t%s:%d:%d:%.4f" % (chromosome, coordination, refBase, altBase, qual, filt, infoStr, gtStr, qual, dp, af)


def PrintVCFHeader(args, call_fh):
    print >> call_fh, '##fileformat=VCFv4.1'
    print >> call_fh, '##FILTER=<ID=PASS,Description="All filters passed">'
    print >> call_fh, '##FILTER=<ID=LowQual,Description="Confidence in this variant being real is below calling threshold.">'
    print >> call_fh, '##ALT=<ID=DEL,Description="Deletion">'
    print >> call_fh, '##ALT=<ID=INS,Description="Insertion of novel sequence">'
    print >> call_fh, '##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">'
    print >> call_fh, '##INFO=<ID=LENGUESS,Number=.,Type=Integer,Description="Best guess of the indel length">'
    print >> call_fh, '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">'
    print >> call_fh, '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">'
    print >> call_fh, '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">'
    print >> call_fh, '##FORMAT=<ID=AF,Number=1,Type=Float,Description="Estimated allele frequency in the range (0,1)">'

    if args.ref_fn != None:
      fai_fn = args.ref_fn + ".fai"
      fai_fp = open(fai_fn)
      for line in fai_fp:
          fields = line.strip().split("\t")
          chromName = fields[0]
          chromLength = int(fields[1])
          print >> call_fh, "##contig=<ID=%s,length=%d>" % (chromName, chromLength)

    print >> call_fh, '#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t%s' % (args.sampleName)

def Test(args, m, utils):
    call_fh = open(args.call_fn, "w")
    if args.v2 == True or args.v3 == True:
        PrintVCFHeader(args, call_fh)
    tensorGenerator = utils.GetTensor( args.tensor_fn, param.predictBatchSize )
    logging.info("Calling variants ...")
    predictStart = time.time()
    end = 0; end2 = 0; terminate = 0
    end2, num2, XBatch2, posBatch2 = next(tensorGenerator)
    m.predictNoRT(XBatch2)
    base = m.predictBaseRTVal; z = m.predictZygosityRTVal; t = m.predictVarTypeRTVal; l = m.predictIndelLengthRTVal
    if end2 == 0:
        end = end2; num = num2; XBatch = XBatch2; posBatch = posBatch2
        end2, num2, XBatch2, posBatch2 = next(tensorGenerator)
        while True:
            if end == 1:
                terminate = 1
            threadPool = []
            if end == 0:
                threadPool.append(Thread(target=m.predictNoRT, args=(XBatch2, )))
            threadPool.append(Thread(target=Output, args=(args, call_fh, num, XBatch, posBatch, base, z, t, l, )))
            for t in threadPool: t.start()
            if end2 == 0:
                end3, num3, XBatch3, posBatch3 = next(tensorGenerator)
            for t in threadPool: t.join()
            base = m.predictBaseRTVal; z = m.predictZygosityRTVal; t = m.predictVarTypeRTVal; l = m.predictIndelLengthRTVal
            if end == 0:
                end = end2; num = num2; XBatch = XBatch2; posBatch = posBatch2
            if end2 == 0:
                end2 = end3; num2 = num3; XBatch2 = XBatch3; posBatch2 = posBatch3
            #print >> sys.stderr, end, end2, end3, terminate
            if terminate == 1:
                break
    elif end2 == 1:
        Output(args, call_fh, num2, XBatch2, posBatch2, base, z, t, l)

    logging.info("Total time elapsed: %.2f s" % (time.time() - predictStart))


def main():
    parser = argparse.ArgumentParser(
            description="Call variants using a trained Clairvoyante model and tensors of candididate variants" )

    parser.add_argument('--tensor_fn', type=str, default = "PIPE",
            help="Tensor input, use PIPE for standard input")

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a checkpoint for testing or continue training")

    parser.add_argument('--call_fn', type=str, default = None,
            help="Output variant predictions")

    parser.add_argument('--qual', type=int, default = None,
            help="If set, variant with equal or higher quality will be marked PASS, or LowQual otherwise, optional")

    parser.add_argument('--sampleName', type=str, default = "SAMPLE",
            help="Define the sample name to be shown in the VCF file")

    parser.add_argument('--showRef', type=param.str2bool, nargs='?', const=True, default = False,
            help="Show reference calls, optional")

    parser.add_argument('--ref_fn', type=str, default=None,
                    help="Reference fasta file input, optional, print contig tags in the VCF header if set")

    parser.add_argument('--threads', type=int, default = None,
            help="Number of threads, optional")

    parser.add_argument('--v3', type=param.str2bool, nargs='?', const=True, default = True,
            help="Use Clairvoyante version 3")

    parser.add_argument('--v2', type=param.str2bool, nargs='?', const=True, default = False,
            help="Use Clairvoyante version 2")

    parser.add_argument('--slim', type=param.str2bool, nargs='?', const=True, default = False,
            help="Train using the slim version of Clairvoyante, optional")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)


if __name__ == "__main__":
    main()
