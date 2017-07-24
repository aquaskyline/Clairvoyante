import sys
import numpy as np
import time
import param

#import clairvoyante as cv
print >> sys.stderr, "Importing Clairvoyante ..."
import utils as utils
import clairvoyante as cv

# load the generate alignment tensors
# use only variants overlapping with the high confident regions defined in `CHROM21_v.3.3.2_highconf_noinconsistent.bed`

print >> sys.stderr, "Loading training data ..."
XArray, YArray, posArray = \
utils.GetTrainingArray("../training/tensor_chr21",
                       "../training/var_chr21",
                       "../testingData/chr21/CHROM21_v.3.3.2_highconf_noinconsistent.bed",
                       "chr21")

print >> sys.stderr, "The shapes of training data:"
print >> sys.stderr, "Input: ", XArray.shape
print >> sys.stderr, "Output: ", YArray.shape

# create a Clairvoyante
print >> sys.stderr, "Initializing model ..."
m = cv.Clairvoyante()
m.init()

# op to write logs to Tensorboard
logsPath = "../training/logs"
summaryWriter = m.summaryFileWriter(logsPath)

# training and save the parameters, we train on the first 80% SNP sites and validate on other 20% SNP sites
print >> sys.stderr, "Start training ..."
trainingStart = time.time()
trainBatchSize = param.trainBatchSize
validationLosts = []
XLen = len(XArray)
XIdx = int(XLen * 0.8)
epoch = 0
epochStart = time.time()
for i in range(1, int(XIdx/trainBatchSize)*30+1):
    XBatch, YBatch = utils.GetBatch(XArray[:XIdx], YArray[:XIdx], size=trainBatchSize)
    loss, summary = m.train(XBatch, YBatch)
    if i % int(XIdx / trainBatchSize) == 0:
        validationLost = m.getLoss( XArray[XIdx:-1], YArray[XIdx:-1] )
        print >> sys.stderr, i, "Training lost:", loss/trainBatchSize, "Validation lost: ", validationLost/(XLen-XIdx)
        m.saveParameters('../training/parameters/cv.params-%05d' % i)
        validationLosts.append( (validationLost, i) )
        if epoch and epoch % 10 == 0:
            print >> sys.stderr, "New learning rate: %.2e" % m.setLearningRate()
        epoch += 1
        print >> sys.stderr, "Epoch time elapsed: ", time.time() - epochStart
        epochStart = time.time()

        # Write summary log
        summaryWriter.add_summary(summary, i)

print >> sys.stderr, "Training time elapsed: ", time.time() - trainingStart

# pick the parameter set of the smallest validation loss
validationLosts.sort()
i = validationLosts[0][1]
print >> sys.stderr, "Best validation lost at batch: %d" % i
#m.restoreParameters('../training/parameters/cv.params-%05d' % i)

print >> sys.stderr, "Loading prediction dataset ..."
XArray2, YArray2, posArray2 = \
utils.GetTrainingArray("../training/tensor_chr22",
                       "../training/var_chr22",
                       "../testingData/chr22/CHROM22_v.3.3.2_highconf_noinconsistent.bed",
                       "chr22")

print >> sys.stderr, "Predicting ..."
predictStart = time.time()
predictBatchSize = param.predictBatchSize
bases, ts = m.predict(XArray2[0:predictBatchSize])
for i in range(predictBatchSize, len(XArray2), predictBatchSize):
    base, t = m.predict(XArray2[i:i+predictBatchSize])
    bases = np.append(bases, base, 0)
    ts = np.append(ts, t, 0)
print >> sys.stderr, "Prediciton time elapsed: ", time.time() - predictStart

print >> sys.stderr, "Model evaluation:"
ed = []
for pos, predictV, annotateV in zip(np.array(posArray2), ts, YArray2[:,4:]):
    ed.append( (pos, np.argmax(predictV), np.argmax(annotateV)) )
ed = np.array(ed)

from collections import Counter
for i in range(5):
    cnt = Counter(ed[ed[:,2]==i,1])
    print >> sys.stderr, i,"\t",
    for j in range(5):
        print >> sys.stderr, cnt.get(j,0),"\t",
    print >> sys.stderr

print >> sys.stderr, "Recall rate for het-call (regardless called variant types):", 1.0*sum((ed[:,1]!=4) & (ed[:,2]==0))/sum(ed[:,2]==0)
print >> sys.stderr, "Recall rate for het-call (called variant type = het):",       1.0*sum((ed[:,1]==0) & (ed[:,2]==0))/sum(ed[:,2]==0)
print >> sys.stderr, "PPV for het-call (regardless called variant types):", 1.0*sum((ed[:,1]==0) & (ed[:,2]!=4))/sum(ed[:,1]==0)
print >> sys.stderr, "PPV for het-call (called variant type = het):",       1.0*sum((ed[:,1]==0) & (ed[:,2]==0))/sum(ed[:,1]==0)
print >> sys.stderr, "Recall rate for hom-call (regardless called variant types):", 1.0*sum((ed[:,1]!=4) & (ed[:,2]==1))/sum(ed[:,2]==1)
print >> sys.stderr, "Recall rate for hom-call (called variant type = hom):",       1.0*sum((ed[:,1]==1) & (ed[:,2]==1))/sum(ed[:,2]==1)
print >> sys.stderr, "PPV for hom-call (regardless called variant types):", 1.0*sum((ed[:,1]==1) & (ed[:,2]!=4))/sum(ed[:,1]==1)
print >> sys.stderr, "PPV for hom-call (called variant type = hom):",       1.0*sum((ed[:,1]==1) & (ed[:,2]==1))/sum(ed[:,1]==1)
print >> sys.stderr, "Recall rate for all calls:", 1.0*sum((ed[:,1]!=4) & (ed[:,2]!=4))/sum(ed[:,2]!=4)
print >> sys.stderr, "PPV for all calls:",         1.0*sum((ed[:,1]!=4) & (ed[:,2]!=4))/sum(ed[:,1]!=4)

