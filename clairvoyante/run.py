import sys
import numpy as np
import time
import param
import logging

# import clairvoyante
logging.basicConfig(format='%(message)s', level=logging.INFO)
import utils as utils
import clairvoyante as cv

# load the generate alignment tensors
# use only variants overlapping with the high confident regions defined in `CHROM21_v.3.3.2_highconf_noinconsistent.bed`

logging.info("Loading training dataset ...")
XArray, YArray, posArray = \
utils.GetTrainingArray("../training/tensor_pi_mul",
                       "../training/var_mul",
                       "../training/bed")

logging.info("The shapes of training dataset:")
logging.info("Input: {}".format(XArray.shape))
logging.info("Output: {}".format(YArray.shape))

# create a Clairvoyante
logging.info("Initializing model ...")
m = cv.Clairvoyante()
m.init()

# op to write logs to Tensorboard
logsPath = "../training/logs"
summaryWriter = m.summaryFileWriter(logsPath)

# training and save the parameters, we train on all variant sites and validate on the last 10% variant sites
logging.info("Start training ...")
trainingStart = time.time()
trainBatchSize = param.trainBatchSize
validationLosts = []
numValItems = int(len(XArray) * 0.1 + 0.499)

c = 0; maxLearningRateSwitch = 10
epochStart = time.time()
for i in range(1, 1 + int(param.maxEpoch * len(XArray) / trainBatchSize + 0.499)):
    XBatch, YBatch = utils.GetBatch(XArray, YArray, size=trainBatchSize)
    loss, summary = m.train(XBatch, YBatch)
    summaryWriter.add_summary(summary, i)
    if i % int(len(XArray) / trainBatchSize + 0.499) == 0:
        validationLost = m.getLoss( XArray[-numValItems:-1], YArray[-numValItems:-1] )
        logging.info(" ".join([i, "Training lost:", loss/trainBatchSize, "Validation lost: ", validationLost/trainBatchSize]))
        validationLosts.append( (validationLost, i) )
        logging.info("Epoch time elapsed: %.2f s" % (time.time() - epochStart))
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
            m.saveParameters('../training/parameters/cv.params-%05d' % i)
            maxLearningRateSwitch -= 1
            if maxLearningRateSwitch == 0:
              break
            logging.info("New learning rate: %.2e" % m.setLearningRate())
            c = 0
        c += 1
        epochStart = time.time()

logging.info("Training time elapsed: %.2f s" % (time.time() - trainingStart))

# pick the parameter set of the smallest validation loss
validationLosts.sort()
i = validationLosts[0][1]
logging.info("Best validation lost at batch: %d" % i)
#m.restoreParameters('../training/parameters/cv.params-%05d' % i)

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

