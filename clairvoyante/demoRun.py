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
    logging.info("Initializing model ...")
    utils.SetupEnv()
    m = cv.Clairvoyante()
    m.init()

    TrainAll(args, m)
    Test22(args, m)


def TrainAll(args, m):
    logging.info("Loading the training dataset ...")
    total, XArrayCompressed, YArrayCompressed, posArrayCompressed = \
    utils.GetTrainingArray("../training/tensor_pi_chr21",
                           "../training/var_chr21",
                           "../training/bed")

    logging.info("The size of training dataset: {}".format(total))

    # op to write logs to Tensorboard
    if args.olog != None:
        summaryWriter = m.summaryFileWriter(args.olog)

    # training and save the parameters, we train on all variant sites and validate on the last 10% variant sites
    logging.info("Start training ...")
    trainingStart = time.time()
    trainBatchSize = param.trainBatchSize
    validationLosts = []
    logging.info("Start at learning rate: %.2e" % m.setLearningRate(args.learning_rate))
    c = 0; maxLearningRateSwitch = param.maxLearningRateSwitch
    epochStart = time.time()
    datasetPtr = 0
    numValItems = int(total * 0.1 + 0.499)
    valXArray, _, _ = utils.DecompressArray(XArrayCompressed, 0, numValItems, total)
    valYArray, _, _ = utils.DecompressArray(YArrayCompressed, 0, numValItems, total)
    logging.info("Number of variants for validation: %d" % len(valXArray))
    i = 1
    while i < (1 + int(param.maxEpoch * total / trainBatchSize + 0.499)):
        XBatch, num, endFlag = utils.DecompressArray(XArrayCompressed, datasetPtr, trainBatchSize, total)
        YBatch, num2, endFlag2 = utils.DecompressArray(YArrayCompressed, datasetPtr, trainBatchSize, total)
        if num != num2 or endFlag != endFlag2:
            sys.exit("Inconsistency between decompressed arrays: %d/%d" % (num, num2))
        loss, summary = m.train(XBatch, YBatch)
        if args.olog != None:
            summaryWriter.add_summary(summary, i)
        if endFlag != 0:
            validationLost = m.getLoss( valXArray, valYArray )
            logging.info(" ".join([str(i), "Training lost:", str(loss/trainBatchSize), "Validation lost: ", str(validationLost/numValItems)]))
            validationLosts.append( (validationLost, i) )
            logging.info("Epoch time elapsed: %.2f s" % (time.time() - epochStart))
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
                maxLearningRateSwitch -= 1
                if maxLearningRateSwitch == 0:
                  break
                logging.info("New learning rate: %.2e" % m.setLearningRate())
                c = 0
            c += 1
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
    predictBatchSize = param.predictBatchSize
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

    if True:
        YArray, _, _ = utils.DecompressArray(YArrayCompressed, 0, total, total)
        logging.info("Model evaluation on the training dataset:")
        ed = np.zeros( (5,5), dtype=np.int )
        for predictV, annotateV in zip(ts, YArray[:,4:]):
            ed[np.argmax(annotateV)][np.argmax(predictV)] += 1

        for i in range(5):
            logging.info("\t".join([str(ed[i][j]) for j in range(5)]))

def Test22(args, m):
    logging.info("Loading the chr22 dataset ...")
    total, XArrayCompressed, YArrayCompressed, posArrayCompressed = \
    utils.GetTrainingArray("../training/tensor_pi_chr22",
                           "../training/var_chr22",
                           "../training/bed")

    logging.info("Testing on the chr22 dataset ...")
    predictStart = time.time()
    predictBatchSize = param.predictBatchSize
    datasetPtr = 0
    XBatch, _, _ = utils.DecompressArray(XArrayCompressed, datasetPtr, predictBatchSize, total)
    bases, ts = m.predict(XBatch)
    while datasetPtr < total:
        XBatch, _, endFlag = utils.DecompressArray(XArrayCompressed, datasetPtr, predictBatchSize, total)
        base, t = m.predict(XBatch)
        bases = np.append(bases, base, 0)
        ts = np.append(ts, t, 0)
        datasetPtr += predictBatchSize
        if endFlag != 0:
            break
    logging.info("Prediciton time elapsed: %.2f s" % (time.time() - predictStart))

    if True:
        YArray, _, _ = utils.DecompressArray(YArrayCompressed, 0, total, total)
        logging.info("Model evaluation on the chr22 dataset:")
        ed = np.zeros( (5,5), dtype=np.int )
        for predictV, annotateV in zip(ts, YArray[:,4:]):
            ed[np.argmax(annotateV)][np.argmax(predictV)] += 1

        for i in range(5):
            logging.info("\t".join([str(ed[i][j]) for j in range(5)]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Training and testing Clairvoyante using demo dataset" )

    parser.add_argument('--learning_rate', type=float, default = param.initialLearningRate,
            help="Set the initial learning rate, default: %f" % param.initialLearningRate)

    parser.add_argument('--olog', type=str, default = None,
            help="Prefix for tensorboard log outputs, optional")

    args = parser.parse_args()

    Run(args)

