import sys
import os
import time
import argparse
import param
import logging
import numpy as np
from threading import Thread
from copy import copy

logging.basicConfig(format='%(message)s', level=logging.INFO)
num2base = dict(zip((0, 1, 2, 3), "ACGT"))
v1Type2Name = dict(zip((0, 1, 2, 3, 4), ('HET', 'HOM', 'INS', 'DEL', 'REF')))
v2Zygosity2Name = dict(zip((0, 1), ('HET', 'HOM')))
v2Type2Name = dict(zip((0, 1, 2, 3), ('REF', 'SNP', 'INS', 'DEL')))
v2Length2Name = dict(zip((0, 1, 2, 3, 4, 5), ('0', '1', '2', '3', '4', '>4')))

def Run(args):
    # create a Clairvoyante
    logging.info("Loading model ...")
    if args.v1 == True:
        import utils_v1 as utils
        if args.slim == True:
            import clairvoyante_v1_slim as cv
        else:
            import clairvoyante_v1 as cv
    elif args.v2 == True:
        import utils_v2 as utils
        if args.slim == True:
            import clairvoyante_v2_slim as cv
        else:
            import clairvoyante_v2 as cv
    elif args.v3 == True:
        import utils_v2 as utils # v3 network is using v2 utils
        if args.slim == True:
            import clairvoyante_v3_slim as cv
        else:
            import clairvoyante_v3 as cv
    utils.SetupEnv()
    m = cv.Clairvoyante()
    m.init()

    m.restoreParameters(os.path.abspath(args.chkpnt_fn))
    Test(args, m, utils)


def Output(args, call_fh, num, posBatch, base, z, t, l):
    if args.v1 == True:
        if num != len(base):
          sys.exit("Inconsistent shape between input tensor and output predictions %d/%d" % (num, len(base)))
        #          --------------  -------------------
        #          Base chng       Var type
        #          A   C   G   T   HET HOM INS DEL REF
        #          0   1   2   3   4   5   6   7   8
        for j in range(len(base)):
            if args.show_ref == False and np.argmax(t[j]) == 4: continue
            sortBase = base[j].argsort()[::-1]
            base1 = num2base[sortBase[0]]
            base2 = num2base[sortBase[1]]
            if(base1 > base2): base1, base2 = base2, base1
            varTypeName = v1Type2Name[np.argmax(t[j])]
            if np.argmax(t[j]) == 0: outBase = "%s%s" % (base1, base2)
            else: outBase = "%s%s" % (base1, base1)
            print >> call_fh, " ".join(posBatch[j].split("-")), outBase, varTypeName

    elif args.v2 == True or args.v3 == True:
        if num != len(base):
          sys.exit("Inconsistent shape between input tensor and output predictions %d/%d" % (num, len(base)))
        #          --------------  ------  ------------    ------------------
        #          Base chng       Zygo.   Var type        Var length
        #          A   C   G   T   HET HOM REF SNP INS DEL 0   1   2   3   4   >=4
        #          0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
        for j in range(len(base)):
            if args.show_ref == False and np.argmax(t[j]) == 0: continue
            sortBase = base[j].argsort()[::-1]
            base1 = num2base[sortBase[0]]
            base2 = num2base[sortBase[1]]
            if(base1 > base2): base1, base2 = base2, base1
            if np.argmax(z[j]) == 0: outBase = "%s%s" % (base1, base2)
            else: outBase = "%s%s" % (base1, base1)
            varZygosityName = v2Zygosity2Name[np.argmax(z[j])]
            varTypeName = v2Type2Name[np.argmax(t[j])]
            varLength = v2Length2Name[np.argmax(l[j])]
            print >> call_fh, " ".join(posBatch[j].split("-")), outBase, varZygosityName, varTypeName, varLength


def Test(args, m, utils):
    call_fh = open(args.call_fn, "w")
    tensorGenerator = utils.GetTensor( args.tensor_fn, param.predictBatchSize )
    logging.info("Calling variants ...")
    predictStart = time.time()
    end = 0; end2 = 0; terminate = 0
    end2, num2, XBatch2, posBatch2 = next(tensorGenerator)
    m.predictNoRT(XBatch2)
    base = copy(m.predictBaseRTVal); z = copy(m.predictZygosityRTVal); t = copy(m.predictVarTypeRTVal); l = copy(m.predictIndelLengthRTVal)
    if end2 == 0:
        end = end2; num = num2; posBatch = posBatch2
        end2, num2, XBatch2, posBatch2 = next(tensorGenerator)
        while True:
            if end == 1:
                terminate = 1
            threadPool = []
            if end == 0:
                threadPool.append(Thread(target=m.predictNoRT, args=(XBatch2, )))
            threadPool.append(Thread(target=Output, args=(args, call_fh, num, posBatch, base, z, t, l, )))
            for t in threadPool: t.start()
            if end2 == 0:
                end3, num3, XBatch3, posBatch3 = next(tensorGenerator)
            for t in threadPool: t.join()
            base = copy(m.predictBaseRTVal); z = copy(m.predictZygosityRTVal); t = copy(m.predictVarTypeRTVal); l = copy(m.predictIndelLengthRTVal)
            if end == 0:
                end = end2; num = num2; posBatch = posBatch2
            if end2 == 0:
                end2 = end3; num2 = num3; XBatch2 = XBatch3; posBatch2 = posBatch3
            #print >> sys.stderr, end, end2, end3, terminate
            if terminate == 1:
                break
    elif end2 == 1:
        Output(args, call_fh, num2, posBatch2, base, z, t, l)



    logging.info("Total time elapsed: %.2f s" % (time.time() - predictStart))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Call variants using a trained Clairvoyante model and tensors of candididate variants" )

    parser.add_argument('--tensor_fn', type=str, default = None,
            help="Tensor input")

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a checkpoint for testing or continue training")

    parser.add_argument('--call_fn', type=str, default = None,
            help="Output variant predictions")

    parser.add_argument('--v3', type=bool, default = True,
            help="Use Clairvoyante version 3")

    parser.add_argument('--v2', type=bool, default = False,
            help="Use Clairvoyante version 2")

    parser.add_argument('--v1', type=bool, default = False,
            help="Use Clairvoyante version 1")

    parser.add_argument('--slim', type=bool, default = False,
            help="Train using the slim version of Clairvoyante, optional")

    parser.add_argument('--show_ref', type=bool, default = False,
            help="Show reference calls, optional")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)

