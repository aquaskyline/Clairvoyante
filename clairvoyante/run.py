import sys
import time
import argparse
import param
import numpy as np
import utils as utils
import clairvoyante as cv

def Run(args):
    # create a Clairvoyante
    print >> sys.stderr, "Initializing model ..."
    m = cv.Clairvoyante()
    m.init()

    if args.chkpnt_fn != None:
        m.restoreParameters(args.chkpnt_fn)
        if args.cont == True:
            TrainAll(args, m)
    else:
        TrainAll(args, m)
    Test22(args, m)


def TrainAll(args, m):
    # load the generate alignment tensors
    # use only variants overlapping with the high confident regions
    print >> sys.stderr, "Loading the training dataset ..."
    XArray, YArray, posArray = \
    utils.GetTrainingArray("../training/tensor_pi_mul",
                           "../training/var_mul",
                           "../training/bed")

    print >> sys.stderr, "The shapes of training dataset:"
    print >> sys.stderr, "Input: ", XArray.shape
    print >> sys.stderr, "Output: ", YArray.shape

    # op to write logs to Tensorboard
    logsPath = "../training/logs"
    summaryWriter = m.summaryFileWriter(logsPath)

    # training and save the parameters, we train on all variant sites and validate on the last 10% variant sites
    print >> sys.stderr, "Start training ..."
    trainingStart = time.time()
    trainBatchSize = param.trainBatchSize
    validationLosts = []
    numValItems = int(len(XArray) * 0.1 + 0.499)
    print >> sys.stderr, "Start at learning rate: %.2e" % m.setLearningRate(args.learning_rate)

    c = 0; maxLearningRateSwitch = 10
    epochStart = time.time()
    i = 1 if args.chkpnt_fn == None else int(args.chkpnt_fn[-param.parameterOutputPlaceHolder:])
    while i < range(1, 1 + int(param.maxEpoch * len(XArray) / trainBatchSize + 0.499)):
        XBatch, YBatch = utils.GetBatch(XArray, YArray, size=trainBatchSize)
        loss, summary = m.train(XBatch, YBatch)
        summaryWriter.add_summary(summary, i)
        if i % int(len(XArray) / trainBatchSize + 0.499) == 0:
            validationLost = m.getLoss( XArray[-numValItems:-1], YArray[-numValItems:-1] )
            print >> sys.stderr, i, "Training lost:", loss/trainBatchSize, "Validation lost: ", validationLost/numValItems
            validationLosts.append( (validationLost, i) )
            print >> sys.stderr, "Time elapsed for 1 epoch: %.2f s" % (time.time() - epochStart)
            flag = 0
            if c >= 4:
              if validationLosts[-4][0] - validationLosts[-3][0] > 0:
                  if validationLosts[-3][0] - validationLosts[-2][0] < 0:
                      if validationLosts[-2][0] - validationLosts[-1][0] > 0:
                          flag = 1
              elif validationLosts[-4][0] - validationLosts[-3][0] < 0:
                  if validationLosts[-3][0] - validationLosts[-2][0] > 0:
                      if validationLosts[-2][0] - validationLosts[-1][0] < 0:
                          flag = 1
              else:
                  flag = 1
            if flag == 1:
                parameterOutputPath = "../training/parameters/cv.params-%%0%dd" % param.parameterOutputPlaceHolder
                m.saveParameters(parameterOutputPath % i)
                maxLearningRateSwitch -= 1
                if maxLearningRateSwitch == 0:
                  break
                print >> sys.stderr, "New learning rate: %.2e" % m.setLearningRate()
                c = 0
            c += 1; i += 1
            epochStart = time.time()

    print >> sys.stderr, "Training time elapsed: %.2f s" % (time.time() - trainingStart)

    # pick the parameter set of the smallest validation loss
    validationLosts.sort()
    i = validationLosts[0][1]
    print >> sys.stderr, "Best validation lost at batch: %d" % i

    print >> sys.stderr, "Testing on the training dataset ..."
    predictStart = time.time()
    predictBatchSize = param.predictBatchSize
    bases, ts = m.predict(XArray[0:predictBatchSize])
    for i in range(predictBatchSize, len(XArray), predictBatchSize):
        base, t = m.predict(XArray[i:i+predictBatchSize])
        bases = np.append(bases, base, 0)
        ts = np.append(ts, t, 0)
    print >> sys.stderr, "Prediciton time elapsed: %.2f s" % (time.time() - predictStart)

    print >> sys.stderr, "Model evaluation on the training dataset:"
    ed = np.zeros( (5,5), dtype=np.int )
    for predictV, annotateV in zip(ts, YArray[:,4:]):
        ed[np.argmax(annotateV)][np.argmax(predictV)] += 1

    for i in range(5):
          print >> sys.stderr, i,"\t",
          for j in range(5):
              print >> sys.stderr, ed[i][j],"\t",
          print >> sys.stderr


def Test22(args, m):
    print >> sys.stderr, "Loading the chr22 dataset ..."
    XArray2, YArray2, posArray2 = \
    utils.GetTrainingArray("../training/tensor_pi_chr22",
                           "../training/var_chr22",
                           "../testingData/chr22/CHROM22_v.3.3.2_highconf_noinconsistent.bed")

    print >> sys.stderr, "Testing on the chr22 dataset ..."
    predictStart = time.time()
    predictBatchSize = param.predictBatchSize
    bases, ts = m.predict(XArray2[0:predictBatchSize])
    for i in range(predictBatchSize, len(XArray2), predictBatchSize):
        base, t = m.predict(XArray2[i:i+predictBatchSize])
        bases = np.append(bases, base, 0)
        ts = np.append(ts, t, 0)
    print >> sys.stderr, "Prediciton time elapsed: %.2f s" % (time.time() - predictStart)

    print >> sys.stderr, "Model evaluation on the chr22 dataset:"
    ed = np.zeros( (5,5), dtype=np.int )
    for predictV, annotateV in zip(ts, YArray2[:,4:]):
        ed[np.argmax(annotateV)][np.argmax(predictV)] += 1

    for i in range(5):
          print >> sys.stderr, i,"\t",
          for j in range(5):
              print >> sys.stderr, ed[i][j],"\t",
          print >> sys.stderr

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Training and testing Clairvoyante using demo dataset" )

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a checkpoint for testing or continue training")

    parser.add_argument('--learning_rate', type=float, default = param.initialLearningRate,
            help="Set the initial learning rate, default: %f" % param.initialLearningRate)

    parser.add_argument('--cont', type=bool, default = False,
            help="If a checkpoint is provided, continue on training the model, default: False")

    args = parser.parse_args()

    Run(args)

