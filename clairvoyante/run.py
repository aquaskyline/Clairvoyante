import sys
sys.path.append('../')

#import clairvoyante as cv
import clairvoyante.utils as utils
import clairvoyante.clairvoyante as cv

# load the generate alignment tensors 
# we use only variants overlapping with the regions defined in `CHROM21_v.3.3.2_highconf_noinconsistent.bed`

XArray, YArray, posArray = \
utils.GetTrainingArray("../training/tensor_chr21", 
                       "../training/var_chr21", 
                       "../testingData/chr21/CHROM21_v.3.3.2_highconf_noinconsistent.bed",
                       "chr21")

print XArray.shape
print YArray.shape

# create a Clairvoyante
m = cv.Clairvoyante()
m.init()

# training and save the parameters, we train on the first 80% SNP sites and validate on other 20% SNP sites
batchSize = 500
validationLosts = []
XLen = len(XArray)
XIdx = int(XLen * 0.8)
epoch = 0
for i in range(1, int(XIdx/batchSize)*30+1):
    XBatch, YBatch = utils.GetBatch(XArray[:XIdx], YArray[:XIdx], size=batchSize)
    loss = m.train(XBatch, YBatch)
    if i % int(XIdx / batchSize) == 0:
        validationLost = m.getLoss( XArray[XIdx:-1], YArray[XIdx:-1] )
        print >> sys.stderr, i,\
                 "Training lost:", loss/batchSize,\
                 "Validation lost: ", validationLost/(XLen-XIdx)
        m.saveParameters('../training/parameters/cv.params-%05d' % i)
        validationLosts.append( (validationLost, i) )
        if epoch and epoch % 10 == 0:
            nl = m.setLearningRate()
            print >> sys.stderr, "New learning rate: %.2e" % nl
        epoch += 1
        
# pick the parameter set of the smallest validation loss
validationLosts.sort()
i = validationLosts[0][1]
print i
#m.restoreParameters('../training/parameters/cv.params-%05d' % i)

XArray2, YArray2, posArray2 = \
utils.GetTrainingArray("../training/tensor_chr22", 
                       "../training/var_chr22", 
                       "../testingData/chr22/CHROM22_v.3.3.2_highconf_noinconsistent.bed",
                       "chr22")

batchSize = 1000
bases, ts = m.predict(XArray2[0:batchSize])

for i in range(batchSize, len(XArray2), batchSize):
    base, t = m.predict(XArray2[i:i+batchSize])
    bases = np.append(bases, base, 0)
    ts = np.append(ts, t, 0)

ed = []
for pos, predictV, annotateV in zip(np.array(posArray2), ts, YArray2[:,4:]):
    ed.append( (pos, np.argmax(predictV), np.argmax(annotateV)) )
ed = np.array(ed)

from collections import Counter

for i in range(5):
    cnt = Counter(ed[ed[:,2]==i,1])
    print i,"\t",
    for j in range(5):
        print cnt.get(j,0),"\t",
    print

print "Recall rate for het-call (regardless called variant types):", 1.0*sum((ed[:,1]!=4) & (ed[:,2]==0))/sum(ed[:,2]==0)
print "Recall rate for het-call (called variant type = het):",       1.0*sum((ed[:,1]==0) & (ed[:,2]==0))/sum(ed[:,2]==0)
print
print "PPV for het-call (regardless called variant types):", 1.0*sum((ed[:,1]==0) & (ed[:,2]!=4))/sum(ed[:,1]==0)
print "PPV for het-call (called variant type = het):",       1.0*sum((ed[:,1]==0) & (ed[:,2]==0))/sum(ed[:,1]==0)
print
print "Recall rate for hom-call (regardless called variant types):", 1.0*sum((ed[:,1]!=4) & (ed[:,2]==1))/sum(ed[:,2]==1)
print "Recall rate for hom-call (called variant type = hom):",       1.0*sum((ed[:,1]==1) & (ed[:,2]==1))/sum(ed[:,2]==1)
print
print "PPV for hom-call (regardless called variant types):", 1.0*sum((ed[:,1]==1) & (ed[:,2]!=4))/sum(ed[:,1]==1)
print "PPV for hom-call (called variant type = hom):",       1.0*sum((ed[:,1]==1) & (ed[:,2]==1))/sum(ed[:,1]==1)
print
print "Recall rate for all calls:", 1.0*sum((ed[:,1]!=4) & (ed[:,2]!=4))/sum(ed[:,2]!=4) 
print "PPV for all calls:",         1.0*sum((ed[:,1]!=4) & (ed[:,2]!=4))/sum(ed[:,1]!=4) 

