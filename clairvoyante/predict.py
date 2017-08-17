import sys
import time
import argparse
import param
import logging
import numpy as np

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
    else:
      import utils_v2 as utils
      if args.slim == True:
          import clairvoyante_v2_slim as cv
      else:
          import clairvoyante_v2 as cv
    utils.SetupEnv()
    m = cv.Clairvoyante()
    m.init()

    m.restoreParameters(args.chkpnt_fn)
    Test(args, m, utils)


def Test(args, m, utils):
    call_fh = open(args.call_fn, "w")
    logging.info("Calling variants ...")
    predictBatchSize = param.predictBatchSize
    predictStart = time.time()
    for num, XBatch, posBatch in utils.GetTensor( args.tensor_fn, param.predictBatchSize ):
        if args.v1 == True:
            base, t = m.predict(XBatch)
            if num != len(base):
              sys.exit("Inconsistent shape between input tensor and output predictions %d/%d" % (num, len(base)))
            #          --------------  -------------------
            #          Base chng       Var type
            #          A   C   G   T   HET HOM INS DEL REF
            #          0   1   2   3   4   5   6   7   8
            for j in range(len(base)):
                if args.show_ref == False and np.argmax(t[j]) == 4: continue
                sortBase = base[j].argsort()[::-1]
                base1 = num2base[sortBase[0]];
                base2 = num2base[sortBase[1]];
                if(base1 > base2): base1, base2 = base2, base1
                varTypeName = v1Type2Name[np.argmax(t[j])];
                if np.argmax(t[j]) == 0: outBase = "%s%s" % (base1, base2)
                else: outBase = "%s%s" % (base1, base1)
                print >> call_fh, " ".join(posBatch[j].split("-")), outBase, varTypeName

        else:
            base, z, t, l = m.predict(XBatch)
            if num != len(base):
              sys.exit("Inconsistent shape between input tensor and output predictions %d/%d" % (num, len(base)))
            #          --------------  ------  ------------    ------------------
            #          Base chng       Zygo.   Var type        Var length
            #          A   C   G   T   HET HOM REF SNP INS DEL 0   1   2   3   4   >=4
            #          0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15
            for j in range(len(base)):
                if args.show_ref == False and np.argmax(t[j]) == 0: continue
                sortBase = base[j].argsort()[::-1]
                base1 = num2base[sortBase[0]];
                base2 = num2base[sortBase[1]];
                if(base1 > base2): base1, base2 = base2, base1
                if np.argmax(z[j]) == 0: outBase = "%s%s" % (base1, base2)
                else: outBase = "%s%s" % (base1, base1)
                varZygosityName = v2Zygosity2Name[np.argmax(z[j])];
                varTypeName = v2Type2Name[np.argmax(t[j])];
                varLength = v2Length2Name[np.argmax(l[j])];
                print >> call_fh, " ".join(posBatch[j].split("-")), outBase, varZygosityName, varTypeName, varLength

    logging.info("Total time elapsed: %.2f s" % (time.time() - predictStart))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Predict using Clairvoyante" )

    parser.add_argument('--tensor_fn', type=str, default = None,
            help="Tensor input")

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a checkpoint for testing or continue training")

    parser.add_argument('--call_fn', type=str, default = None,
            help="Output variant predictions")

    parser.add_argument('--v1', type=bool, default = False,
            help="Use Clairvoyante version 1")

    parser.add_argument('--slim', type=bool, default = False,
            help="Train using the slim version of Clairvoyante, optional")

    parser.add_argument('--show_ref', type=bool, default = False,
            help="Show reference calls, optional")

    args = parser.parse_args()

    Run(args)

