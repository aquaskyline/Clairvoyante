import sys
import time
import argparse
import param
import logging
import pickle
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
    logging.info("Loading the dataset ...")

    if args.bin_fn != None:
        with open(args.bin_fn, "rb") as fh:
            total = pickle.load(fh)
            XArrayCompressed = pickle.load(fh)
            YArrayCompressed = pickle.load(fh)
            posArrayCompressed = pickle.load(fh)
    else:
        total, XArrayCompressed, YArrayCompressed, posArrayCompressed = \
        utils.GetTrainingArray(args.tensor_fn,
                               args.var_fn,
                               args.bed_fn)

    logging.info("Dataset size: %d" % total)
    logging.info("Testing on the dataset ...")
    predictBatchSize = param.predictBatchSize
    predictStart = time.time()
    if args.v1 == True:
        datasetPtr = 0
        XBatch, _, _ = utils.DecompressArray(XArrayCompressed, datasetPtr, predictBatchSize, total)
        bases, ts = m.predict(XBatch)
        datasetPtr += predictBatchSize
        while datasetPtr < total:
            XBatch, _, endFlag = utils.DecompressArray(XArrayCompressed, datasetPtr, predictBatchSize, total)
            base, t = m.predict(XBatch)
            bases = np.append(bases, base, 0)
            ts = np.append(ts, t, 0)
            datasetPtr += predictBatchSize
            if endFlag != 0:
                break
    else:
        datasetPtr = 0
        XBatch, _, _ = utils.DecompressArray(XArrayCompressed, datasetPtr, predictBatchSize, total)
        bases, zs, ts, ls = m.predict(XBatch)
        datasetPtr += predictBatchSize
        while datasetPtr < total:
            XBatch, _, endFlag = utils.DecompressArray(XArrayCompressed, datasetPtr, predictBatchSize, total)
            base, z, t, l = m.predict(XBatch)
            bases = np.append(bases, base, 0)
            zs = np.append(zs, z, 0)
            ts = np.append(ts, t, 0)
            ls = np.append(ls, l, 0)
            datasetPtr += predictBatchSize
            if endFlag != 0:
                break
    logging.info("Prediciton time elapsed: %.2f s" % (time.time() - predictStart))

    YArray, _, _ = utils.DecompressArray(YArrayCompressed, 0, total, total)
    if args.v1 == True:
        logging.info("Version 1 model, evaluation on base change:")
        allBaseCount = top1Count = top2Count = 0
        for predictV, annotateV in zip(bases, YArray[:,0:4]):
            allBaseCount += 1
            sortPredictV = predictV.argsort()[::-1]
            if sortPredictV[np.argmax(annotateV)] == 0:
                top1Count += 1
                top2Count += 1
            if sortPredictV[np.argmax(annotateV)] == 1:
                top2Count += 1
        logging.info("all/top1/top2: %d/%d/%d" % (allBaseCount, top1Count, top2Count))
        logging.info("Version 1 model, evaluation on variant type:")
        ed = np.zeros( (5,5), dtype=np.int )
        for predictV, annotateV in zip(ts, YArray[:,4:9]):
            ed[np.argmax(annotateV)][np.argmax(predictV)] += 1
        for i in range(5):
            logging.info("\t".join([str(ed[i][j]) for j in range(5)]))
    else:
        logging.info("Version 2 model, evaluation on base change:")
        allBaseCount = top1Count = top2Count = 0
        for predictV, annotateV in zip(bases, YArray[:,0:4]):
            allBaseCount += 1
            sortPredictV = predictV.argsort()[::-1]
            if sortPredictV[np.argmax(annotateV)] == 0:
                top1Count += 1
                top2Count += 1
            if sortPredictV[np.argmax(annotateV)] == 1:
                top2Count += 1
        logging.info("all/top1/top2: %d/%d/%d" % (allBaseCount, top1Count, top2Count))
        logging.info("Version 2 model, evaluation on Zygosity:")
        ed = np.zeros( (2,2), dtype=np.int )
        for predictV, annotateV in zip(zs, YArray[:,4:6]):
            ed[np.argmax(annotateV)][np.argmax(predictV)] += 1
        for i in range(2):
            logging.info("\t".join([str(ed[i][j]) for j in range(2)]))
        logging.info("Version 2 model, evaluation on variant type:")
        ed = np.zeros( (4,4), dtype=np.int )
        for predictV, annotateV in zip(ts, YArray[:,6:10]):
            ed[np.argmax(annotateV)][np.argmax(predictV)] += 1
        for i in range(4):
            logging.info("\t".join([str(ed[i][j]) for j in range(4)]))
        logging.info("Version 2 model, evaluation on indel length:")
        ed = np.zeros( (6,6), dtype=np.int )
        for predictV, annotateV in zip(ls, YArray[:,10:16]):
            ed[np.argmax(annotateV)][np.argmax(predictV)] += 1
        for i in range(6):
            logging.info("\t".join([str(ed[i][j]) for j in range(6)]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Predict and compare using Clairvoyante" )

    parser.add_argument('--bin_fn', type=str, default = None,
            help="Binary tensor input generated by tensor2Bin.py, tensor_fn, var_fn and bed_fn will be ignored")

    parser.add_argument('--tensor_fn', type=str, default = None,
            help="Tensor input")

    parser.add_argument('--var_fn', type=str, default = None,
            help="Truth variants list input")

    parser.add_argument('--bed_fn', type=str, default = None,
            help="High confident genome regions input in the BED format")

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a checkpoint for testing")

    parser.add_argument('--v1', type=bool, default = False,
            help="Use Clairvoyante version 1")

    parser.add_argument('--slim', type=bool, default = False,
            help="Train using the slim version of Clairvoyante, optional")

    args = parser.parse_args()

    Run(args)

