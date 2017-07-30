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
    m = cv.Clairvoyante()
    m.init()

    m.restoreParameters(args.chkpnt_fn)


def Test(args, m):
    logging.info("Loading the dataset ...")
    XArray, YArray, posArray = \
    utils.GetTrainingArray(args.tensor_fn,
                           args.var_fn,
                           args.bed_fn)

    logging.info("Testing on the dataset ...")
    predictStart = time.time()
    predictBatchSize = param.predictBatchSize
    bases, ts = m.predict(XArray[0:predictBatchSize])
    for i in range(predictBatchSize, len(XArray), predictBatchSize):
        base, t = m.predict(XArray[i:i+predictBatchSize])
        bases = np.append(bases, base, 0)
        ts = np.append(ts, t, 0)
    logging.info("Prediciton time elapsed: %.2f s" % (time.time() - predictStart))

    logging.info("Model evaluation on the dataset:")
    ed = np.zeros( (5,5), dtype=np.int )
    for predictV, annotateV in zip(ts, YArray[:,4:]):
        ed[np.argmax(annotateV)][np.argmax(predictV)] += 1

    for i in range(5):
        logging.info("\t".join([str(ed[i][j]) for j in range(5)]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Predict and compare using Clairvoyante" )

    parser.add_argument('--tensor_fn', type=str, default = None,
            help="Tensor input")

    parser.add_argument('--var_fn', type=str, default = None,
            help="Truth variants list input")

    parser.add_argument('--bed_fn', type=str, default = None,
            help="High confident genome regions input in the BED format")

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a checkpoint for testing or continue training")

    args = parser.parse_args()

    Run(args)

