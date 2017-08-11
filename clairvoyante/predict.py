import sys
import time
import argparse
import param
import logging
import numpy as np

logging.basicConfig(format='%(message)s', level=logging.INFO)

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
    logging.info("Predicing the dataset ...")
    call_fh = open(args.call_fn, "w")
    predictStart = time.time()
    for num, XBatch, posBatch in utils.GetTensor( args.tensor_fn, param.predictBatchSize ):
        base, t = m.predict(XBatch)
        if num != len(base):
            sys.exit("Inconsistent shape between input tensor and output predictions %d/%d" % (num, len(base)))
        for j in range(len(base)):
            print >> call_fh, posBatch[j], np.argmax(base[j]), np.argmax(t[j])
    logging.info("Prediciton time elapsed: %.2f s" % (time.time() - predictStart))


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

    args = parser.parse_args()

    Run(args)

