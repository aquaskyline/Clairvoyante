import sys
import time
import argparse
import param
import logging
import pickle
import numpy as np
import utils as utils
import clairvoyante_slim as cv

logging.basicConfig(format='%(message)s', level=logging.INFO)

def Run(args):
    # create a Clairvoyante
    logging.info("Initializing model ...")
    utils.SetupEnv()
    m = cv.Clairvoyante()
    m.init()

    if args.chkpnt_fn != None:
        m.restoreParameters(args.chkpnt_fn)
    TrainAll(args, m)


def TrainAll(args, m):
    logging.info("Loading the training dataset ...")
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

    logging.info("The size of training dataset: {}".format(total))

    # op to write logs to Tensorboard
    if args.olog_dir != None:
        summaryWriter = m.summaryFileWriter(args.olog_dir)

    # training and save the parameters, we train on all variant sites and validate on the last 20% variant sites
    logging.info("Start training ...")
    trainingStart = time.time()
    trainBatchSize = param.trainBatchSize
    predictBatchSize = param.predictBatchSize
    validationLosts = []
    logging.info("Start at learning rate: %.2e" % m.setLearningRate(args.learning_rate))
    epochStart = time.time()
    numValItems = int(total * 0.2 + 0.499)
    c = 0; maxLearningRateSwitch = param.maxLearningRateSwitch
    datasetPtr = 0
    i = 1 if args.chkpnt_fn == None else int(args.chkpnt_fn[-param.parameterOutputPlaceHolder:])+1
    while i < (1 + int(param.maxEpoch * total / trainBatchSize + 0.499)):
        XBatch, num, endFlag = utils.DecompressArray(XArrayCompressed, datasetPtr, trainBatchSize, total)
        YBatch, num2, endFlag2 = utils.DecompressArray(YArrayCompressed, datasetPtr, trainBatchSize, total)
        if num != num2 or endFlag != endFlag2:
            sys.exit("Inconsistency between decompressed arrays: %d/%d" % (num, num2))
        loss, summary = m.train(XBatch, YBatch)
        if args.olog_dir != None:
            summaryWriter.add_summary(summary, i)
        if endFlag != 0:
            validationLost = 0
            for j in range(0, numValItems, predictBatchSize):
                XBatch, _, _ = utils.DecompressArray(XArrayCompressed, j, predictBatchSize, numValItems)
                YBatch, _, _ = utils.DecompressArray(YArrayCompressed, j, predictBatchSize, numValItems)
                validationLost += m.getLoss( XBatch, YBatch )
            logging.info(" ".join([str(i), "Training lost:", str(loss/trainBatchSize), "Validation lost: ", str(validationLost/numValItems)]))
            logging.info("Epoch time elapsed: %.2f s" % (time.time() - epochStart))
            validationLosts.append( (validationLost, i) )
            c += 1
            flag = 0
            if c >= 6:
              if validationLosts[-6][0] - validationLosts[-5][0] > 0:
                  if validationLosts[-5][0] - validationLosts[-4][0] < 0:
                      if validationLosts[-4][0] - validationLosts[-3][0] > 0:
                          if validationLosts[-3][0] - validationLosts[-2][0] < 0:
                              if validationLosts[-2][0] - validationLosts[-1][0] > 0:
                                  flag = 1
              elif validationLosts[-6][0] - validationLosts[-5][0] < 0:
                  if validationLosts[-5][0] - validationLosts[-4][0] > 0:
                      if validationLosts[-4][0] - validationLosts[-3][0] < 0:
                          if validationLosts[-3][0] - validationLosts[-2][0] > 0:
                              if validationLosts[-2][0] - validationLosts[-1][0] < 0:
                                  flag = 1
              else:
                  flag = 1
            if flag == 1:
                if args.ochk_prefix != None:
                    parameterOutputPath = "%s-%%0%dd" % ( args.ochk_prefix, param.parameterOutputPlaceHolder )
                    m.saveParameters(parameterOutputPath % i)
                maxLearningRateSwitch -= 1
                if maxLearningRateSwitch == 0:
                  break
                logging.info("New learning rate: %.2e" % m.setLearningRate())
                c = 0
            epochStart = time.time()
            datasetPtr = 0
        i += 1
        datasetPtr += trainBatchSize

    logging.info("Training time elapsed: %.2f s" % (time.time() - trainingStart))

    # show the parameter set with the smallest validation loss
    validationLosts.sort()
    i = validationLosts[0][1]
    logging.info("Best validation lost at batch: %d" % i)

    logging.info("Testing on the training dataset ...")
    predictStart = time.time()
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
    logging.info("Prediciton time elapsed: %.2f s" % (time.time() - predictStart))

    logging.info("Model evaluation on the training dataset:")
    if True:
        YArray, _, _ = utils.DecompressArray(YArrayCompressed, 0, total, total)
        ed = np.zeros( (5,5), dtype=np.int )
        for predictV, annotateV in zip(ts, YArray[:,4:]):
            ed[np.argmax(annotateV)][np.argmax(predictV)] += 1

        for i in range(5):
            logging.info("\t".join([str(ed[i][j]) for j in range(5)]))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Training Clairvoyante" )

    parser.add_argument('--bin_fn', type=str, default = None,
            help="Binary tensor input generated by tensor2Bin.py, tensor_fn, var_fn and bed_fn will be ignored")

    parser.add_argument('--tensor_fn', type=str, default = None,
            help="Tensor input")

    parser.add_argument('--var_fn', type=str, default = None,
            help="Truth variants list input")

    parser.add_argument('--bed_fn', type=str, default = None,
            help="High confident genome regions input in the BED format")

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a checkpoint for testing or continue training")

    parser.add_argument('--learning_rate', type=float, default = param.initialLearningRate,
            help="Set the initial learning rate, default: %(default)s")

    parser.add_argument('--ochk_prefix', type=str, default = None,
            help="Prefix for checkpoint outputs at each learning rate change, optional")

    parser.add_argument('--olog_dir', type=str, default = None,
            help="Directory for tensorboard log outputs, optional")

    args = parser.parse_args()

    Run(args)

