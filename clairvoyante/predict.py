import sys
import time
import argparse
import param
import logging
import numpy as np
import utils as utils
import clairvoyante as cv

logging.basicConfig(format='%(message)s', level=logging.INFO)

def Run(args):
    # create a Clairvoyante
    logging.info("Loading model ...")
    utils.SetupEnv()
    m = cv.Clairvoyante()
    m.init()

    m.restoreParameters(args.chkpnt_fn)
    Test(args, m)


def Test(args, m):
    logging.info("Loading the dataset ...")
    total, XArrayCompressed, posArrayCompressed = \
    utils.GetAlnArray(args.tensor_fn,
                      args.bed_fn)

    logging.info("Predicing the dataset ...")
    call_fh = open(args.call_fn, "w")
    predictStart = time.time()
    predictBatchSize = param.predictBatchSize
    for i in range(0, total, predictBatchSize):
        XBatch, _, endFlag = utils.DecompressArray(XArrayCompressed, i, predictBatchSize, total)
        base, t = m.predict(XBatch)
        for j in (len(base)):
            print >> call_fh, posArray[i+j], np.argmax(base[i+j]), np.argmax(t[i+j])
    logging.info("Prediciton time elapsed: %.2f s" % (time.time() - predictStart))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Predict using Clairvoyante" )

    parser.add_argument('--tensor_fn', type=str, default = None,
            help="Tensor input")

    parser.add_argument('--bed_fn', type=str, default = None,
            help="High confident genome regions input in the BED format")

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a checkpoint for testing or continue training")

    parser.add_argument('--call_fn', type=str, default = None,
            help="Output variant predictions")

    args = parser.parse_args()

    Run(args)

