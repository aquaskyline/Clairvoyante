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

    if args.chkpnt_fn != None:
        m.restoreParameters(args.chkpnt_fn)
        if args.cont == True:
            TrainAll(args, m)
    else:
        TrainAll(args, m)


def TrainAll(args, m):
    # load the generate alignment tensors
    # use only variants overlapping with the high confident regions
    logging.info("Loading the training dataset ...")
    XArray, YArray, posArray = \
    utils.GetTrainingArray(args.tensor_fn,
                           args.var_fn,
                           args.bed_fn)

    logging.info("The shapes of training dataset:")
    logging.info("Input: {}".format(XArray.shape))
    logging.info("Output: {}".format(YArray.shape))

    # op to write logs to Tensorboard
    if args.olog != None:
        summaryWriter = m.summaryFileWriter(args.olog)

    # training and save the parameters, we train on all variant sites and validate on the last 20% variant sites
    logging.info("Start training ...")
    trainingStart = time.time()
    trainBatchSize = param.trainBatchSize
    predictBatchSize = param.predictBatchSize
    validationLosts = []
    numValItems = int(len(XArray) * 0.2 + 0.499)
    logging.info("Start at learning rate: %.2e" % m.setLearningRate(args.learning_rate))

    c = 0; maxLearningRateSwitch = 10
    epochStart = time.time()
    i = 1 if args.chkpnt_fn == None else int(args.chkpnt_fn[-param.parameterOutputPlaceHolder:])+1
    while i < range(1, 1 + int(param.maxEpoch * len(XArray) / trainBatchSize + 0.499)):
        XBatch, YBatch = utils.GetBatch(XArray, YArray, size=trainBatchSize)
        loss, summary = m.train(XBatch, YBatch)
        if args.olog != None:
            summaryWriter.add_summary(summary, i)
        if i % int(len(XArray) / trainBatchSize + 0.499) == 0:
            validationLost = 0
            for j in range(len(XArray)-numValItems, len(XArray), predictBatchSize):
                validationLost += m.getLoss( XArray[j:j+predictBatchSize], YArray[j:j+predictBatchSize] )
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
                if args.ochk != None:
                    parameterOutputPath = "%s-%%0%dd" % ( args.ochk, param.parameterOutputPlaceHolder )
                    m.saveParameters(parameterOutputPath % i)
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

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Training Clairvoyante" )

    parser.add_argument('--tensor_fn', type=str, default = None,
            help="Tensor input")

    parser.add_argument('--var_fn', type=str, default = None,
            help="Truth variants list input")

    parser.add_argument('--bed_fn', type=str, default = None,
            help="High confident genome regions input in the BED format")

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a checkpoint for testing or continue training")

    parser.add_argument('--learning_rate', type=float, default = param.initialLearningRate,
            help="Set the initial learning rate, default: %f" % param.initialLearningRate)

    parser.add_argument('--cont', type=bool, default = False,
            help="If a checkpoint is provided, continue on training the model, default: False")

    parser.add_argument('--ochk', type=str, default = None,
            help="Prefix for checkpoint outputs at each learning rate change, optional")

    parser.add_argument('--olog', type=str, default = None,
            help="Prefix for tensorboard log outputs, optional")

    args = parser.parse_args()

    Run(args)

