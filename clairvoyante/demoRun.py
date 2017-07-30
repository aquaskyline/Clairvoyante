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
    m = cv.Clairvoyante()
    m.init()

    TrainAll(args, m)
    Test22(args, m)


def TrainAll(args, m):
    # load the generate alignment tensors
    # use only variants overlapping with the high confident regions
    logging.info("Loading the training dataset ...")
    XArray, YArray, posArray = \
    utils.GetTrainingArray("../training/tensor_mix",
                           "../training/var_mul",
                           "../training/bed")

    logging.info("The shapes of training dataset:")
    logging.info("Input: {}".format(XArray.shape))
    logging.info("Output: {}".format(YArray.shape))

    # op to write logs to Tensorboard
    if args.olog != None:
        summaryWriter = m.summaryFileWriter(args.olog)

    # training and save the parameters, we train on all variant sites and validate on the last 10% variant sites
    logging.info("Start training ...")
    trainingStart = time.time()
    trainBatchSize = param.trainBatchSize
    validationLosts = []
    numValItems = int(len(XArray) * 0.1 + 0.499)
    logging.info("Start at learning rate: %.2e" % m.setLearningRate(args.learning_rate))

    c = 0; maxLearningRateSwitch = param.maxLearningRateSwitch
    epochStart = time.time()
    while i < range(1, 1 + int(param.maxEpoch * len(XArray) / trainBatchSize + 0.499)):
        XBatch, YBatch = utils.GetBatch(XArray, YArray, size=trainBatchSize)
        loss, summary = m.train(XBatch, YBatch)
        if args.olog != None:
            summaryWriter.add_summary(summary, i)
        if i % int(len(XArray) / trainBatchSize + 0.499) == 0:
            validationLost = m.getLoss( XArray[-numValItems:-1], YArray[-numValItems:-1] )
            logging.info(" ".join([str(i), "Training lost:", str(loss/trainBatchSize), "Validation lost: ", str(validationLost/trainBatchSize)]))
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
        i += 1

    logging.info("Training time elapsed: %.2f s" % (time.time() - trainingStart))

    # show the parameter set with the smallest validation loss
    validationLosts.sort()
    i = validationLosts[0][1]
    logging.info("Best validation lost at batch: %d" % i)

    logging.info("Testing on the training dataset ...")
    predictStart = time.time()
    predictBatchSize = param.predictBatchSize
    bases, ts = m.predict(XArray[0:predictBatchSize])
    for i in range(predictBatchSize, len(XArray), predictBatchSize):
        base, t = m.predict(XArray[i:i+predictBatchSize])
        bases = np.append(bases, base, 0)
        ts = np.append(ts, t, 0)
    logging.info("Prediciton time elapsed: %.2f s" % (time.time() - predictStart))

    logging.info("Model evaluation on the training dataset:")
    ed = np.zeros( (5,5), dtype=np.int )
    for predictV, annotateV in zip(ts, YArray[:,4:]):
        ed[np.argmax(annotateV)][np.argmax(predictV)] += 1

    for i in range(5):
        logging.info("\t".join([str(ed[i][j]) for j in range(5)]))

def Test22(args, m):
    logging.info("Loading the chr22 dataset ...")
    XArray2, YArray2, posArray2 = \
    utils.GetTrainingArray("../training/tensor_pi_chr22",
                           "../training/var_chr22",
                           "../testingData/chr22/CHROM22_v.3.3.2_highconf_noinconsistent.bed")

    logging.info("Testing on the chr22 dataset ...")
    predictStart = time.time()
    predictBatchSize = param.predictBatchSize
    bases, ts = m.predict(XArray2[0:predictBatchSize])
    for i in range(predictBatchSize, len(XArray2), predictBatchSize):
        base, t = m.predict(XArray2[i:i+predictBatchSize])
        bases = np.append(bases, base, 0)
        ts = np.append(ts, t, 0)
    logging.info("Prediciton time elapsed: %.2f s" % (time.time() - predictStart))

    logging.info("Model evaluation on the chr22 dataset:")
    ed = np.zeros( (5,5), dtype=np.int )
    for predictV, annotateV in zip(ts, YArray2[:,4:]):
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

