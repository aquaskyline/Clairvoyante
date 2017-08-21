import tensorflow as tf
import selu
import param

class Clairvoyante(object):

    def __init__(self, inputShape = (2*param.flankingBaseNum+1, 4, param.matrixNum),
                       outputShape1 = (4, ), outputShape2 = (2, ), outputShape3 = (4, ), outputShape4 = (6, ),
                       kernelSize1 = (1, 4), kernelSize2 = (3, 4), kernelSize3 = (5, 4),
                       numFeature1 = 8, numFeature2 = 16, numFeature3 = 32,
                       hiddenLayerUnits4 = 36, hiddenLayerUnits5 = 18,
                       initialLearningRate = param.initialLearningRate,
                       learningRateDecay = param.learningRateDecay,
                       dropoutRateFC4 = param.dropoutRateFC4, dropoutRateFC5 = param.dropoutRateFC5):
        self.inputShape = inputShape
        self.outputShape1 = outputShape1; self.outputShape2 = outputShape2; self.outputShape3 = outputShape3; self.outputShape4 = outputShape4
        self.kernelSize1 = kernelSize1; self.kernelSize2 = kernelSize2; self.kernelSize3 = kernelSize3
        self.numFeature1 = numFeature1; self.numFeature2 = numFeature2; self.numFeature3 = numFeature3
        self.hiddenLayerUnits4 = hiddenLayerUnits4; self.hiddenLayerUnits5 = hiddenLayerUnits5
        self.learningRateVal = initialLearningRate
        self.learningRateDecay = learningRateDecay
        self.dropoutRateFC4Val = dropoutRateFC4; self.dropoutRateFC5Val = dropoutRateFC5
        self.g = tf.Graph()
        self._buildGraph()
        self.session = tf.Session(graph = self.g)

    def _buildGraph(self):
        with self.g.as_default():
            XPH = tf.placeholder(tf.float32, [None, self.inputShape[0], self.inputShape[1], self.inputShape[2]], name='XPH')
            self.XPH = XPH

            YPH = tf.placeholder(tf.float32, [None, self.outputShape1[0] + self.outputShape2[0] + self.outputShape3[0] + self.outputShape4[0]], name='YPH')
            self.YPH = YPH

            learningRatePH = tf.placeholder(tf.float32, shape=[], name='learningRatePH')
            self.learningRatePH = learningRatePH

            phasePH = tf.placeholder(tf.bool, shape=[], name='phasePH')
            self.phasePH = phasePH

            dropoutRateFC4PH = tf.placeholder(tf.float32, shape=[], name='dropoutRateFC4PH')
            self.dropoutRateFC4PH = dropoutRateFC4PH

            dropoutRateFC5PH = tf.placeholder(tf.float32, shape=[], name='dropoutRateFC5PH')
            self.dropoutRateFC5PH = dropoutRateFC5PH

            conv1 = tf.layers.conv2d(inputs=XPH,
                                     filters=self.numFeature1,
                                     kernel_size=self.kernelSize1,
                                     kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
                                     padding="same",
                                     activation=selu.selu,
                                     name='conv1')

            conv2 = tf.layers.conv2d(inputs=conv1,
                                     filters=self.numFeature2,
                                     kernel_size=self.kernelSize2,
                                     kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
                                     padding="same",
                                     activation=selu.selu,
                                     name='conv2')

            conv3 = tf.layers.conv2d(inputs=conv2,
                                     filters=self.numFeature3,
                                     kernel_size=self.kernelSize3,
                                     kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
                                     padding="same",
                                     activation=selu.selu,
                                     name='conv3')

            flat_size = self.inputShape[0] * self.inputShape[1] * self.numFeature3
            conv3_flat =  tf.reshape(conv3, [-1,  flat_size])

            fc4 = tf.layers.dense(inputs=conv3_flat,
                                 units=self.hiddenLayerUnits4,
                                 kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
                                 activation=selu.selu,
                                 name='fc4')

            dropout4 = selu.dropout_selu(fc4, dropoutRateFC4PH, training=phasePH, name='dropout4')

            fc5 = tf.layers.dense(inputs=dropout4,
                                 units=self.hiddenLayerUnits5,
                                 kernel_initializer = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False),
                                 activation=selu.selu,
                                 name='fc5')

            dropout5 = selu.dropout_selu(fc5, dropoutRateFC5PH, training=phasePH, name='dropout5')

            epsilon = tf.constant(value=1e-10)
            YBaseChangeSigmoid = tf.layers.dense(inputs=dropout5, units=self.outputShape1[0], activation=tf.nn.sigmoid, name='YBaseChangeSigmoid')
            YZygosityFC = tf.layers.dense(inputs=dropout5, units=self.outputShape2[0], activation=selu.selu, name='YZygosityFC')
            YZygosityLogits = tf.add(YZygosityFC, epsilon, name='YZygosityLogits')
            YZygositySoftmax = tf.nn.softmax(YZygosityLogits, name='YZygositySoftmax')
            YVarTypeFC = tf.layers.dense(inputs=dropout5, units=self.outputShape3[0], activation=selu.selu, name='YVarTypeFC')
            YVarTypeLogits = tf.add(YVarTypeFC, epsilon, name='YVarTypeLogits')
            YVarTypeSoftmax = tf.nn.softmax(YVarTypeLogits, name='YVarTypeSoftmax')
            YIndelLengthFC = tf.layers.dense(inputs=dropout5, units=self.outputShape4[0], activation=selu.selu, name='YIndelLengthFC')
            YIndelLengthLogits = tf.add(YIndelLengthFC, epsilon, name='YIndelLengthLogits')
            YIndelLengthSoftmax = tf.nn.softmax(YIndelLengthLogits, name='YIndelLengthSoftmax')
            self.YBaseChangeSigmoid = YBaseChangeSigmoid
            self.YZygositySoftmax = YZygositySoftmax
            self.YVarTypeSoftmax = YVarTypeSoftmax
            self.YIndelLengthSoftmax = YIndelLengthSoftmax

            loss1 = tf.reduce_sum(tf.pow(YBaseChangeSigmoid - tf.slice(YPH,[0,0],[-1,self.outputShape1[0]], name='YBaseChangeGetTruth'), 2, name='YBaseChangeMSE'), name='YBaseChangeReduceSum')
            YZygosityCrossEntropy = tf.nn.log_softmax(YZygosityLogits, name='YZygosityLogSoftmax')\
                                    * -tf.slice(YPH, [0,self.outputShape1[0]], [-1,self.outputShape2[0]], name='YZygosityGetTruth')
            loss2 = tf.reduce_sum(YZygosityCrossEntropy, name='YZygosityReduceSum')
            YVarTypeCrossEntropy = tf.nn.log_softmax(YVarTypeLogits, name='YVarTypeLogSoftmax')\
                                   * -tf.slice(YPH, [0,self.outputShape1[0]+self.outputShape2[0]], [-1,self.outputShape3[0]], name='YVarTypeGetTruth')
            loss3 = tf.reduce_sum(YVarTypeCrossEntropy, name='YVarTypeReduceSum')
            YIndelLengthCrossEntropy = tf.nn.log_softmax(YIndelLengthLogits, name='YIndelLengthLogSoftmax')\
                                       * -tf.slice(YPH, [0,self.outputShape1[0]+self.outputShape2[0]+self.outputShape3[0]], [-1,self.outputShape4[0]], name='YIndelLengthGetTruth')
            loss4 = tf.reduce_sum(YIndelLengthCrossEntropy, name='YIndelLengthReduceSum')
            lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name ]) * learningRatePH
            loss = loss1 + loss2 + loss3 + loss4 + lossL2
            self.loss = loss

            # add summaries
            tf.summary.scalar('learning_rate', learningRatePH)
            tf.summary.scalar("loss1", loss1)
            tf.summary.scalar("loss2", loss2)
            tf.summary.scalar("loss3", loss3)
            tf.summary.scalar("loss4", loss4)
            tf.summary.scalar("loss", loss)

            # For report or debug. Fetching histogram summary is slow, GPU utilization will be low if enabled.
            #for var in tf.trainable_variables():
            #    tf.summary.histogram(var.op.name, var)
            self.merged_summary_op = tf.summary.merge_all()

            self.training_op = tf.train.AdamOptimizer(learning_rate=learningRatePH).minimize(loss)
            self.init_op = tf.global_variables_initializer()

    def init(self):
        self.session.run( self.init_op )

    def close(self):
        self.session.close()

    def train(self, batchX, batchY):
        #for i in range(len(batchX)):
        #    tf.image.per_image_standardization(batchX[i])
        loss, _, summary = self.session.run( (self.loss, self.training_op, self.merged_summary_op),
                                              feed_dict={self.XPH:batchX, self.YPH:batchY,
                                                         self.learningRatePH:self.learningRateVal,
                                                         self.phasePH:True,
                                                         self.dropoutRateFC4PH:self.dropoutRateFC4Val,
                                                         self.dropoutRateFC5PH:self.dropoutRateFC5Val})
        return loss, summary

    def trainNoRT(self, batchX, batchY):
        #for i in range(len(batchX)):
        #    tf.image.per_image_standardization(batchX[i])
        self.trainLossRTVal, _, self.trainSummaryRTVal = self.session.run( (self.loss, self.training_op, self.merged_summary_op),
                                              feed_dict={self.XPH:batchX, self.YPH:batchY,
                                                         self.learningRatePH:self.learningRateVal,
                                                         self.phasePH:True,
                                                         self.dropoutRateFC4PH:self.dropoutRateFC4Val,
                                                         self.dropoutRateFC5PH:self.dropoutRateFC5Val})

    def getLoss(self, batchX, batchY):
        #for i in range(len(batchX)):
        #    tf.image.per_image_standardization(batchX[i])
        loss = self.session.run( self.loss, feed_dict={self.XPH:batchX, self.YPH:batchY,
                                                       self.learningRatePH:0.0,
                                                       self.phasePH:False,
                                                       self.dropoutRateFC4PH:0.0,
                                                       self.dropoutRateFC5PH:0.0})
        return loss

    def getLossNoRT(self, batchX, batchY):
        #for i in range(len(batchX)):
        #    tf.image.per_image_standardization(batchX[i])
        self.getLossLossRTVal = self.session.run( self.loss, feed_dict={self.XPH:batchX, self.YPH:batchY,
                                                                        self.learningRatePH:0.0,
                                                                        self.phasePH:False,
                                                                        self.dropoutRateFC4PH:0.0,
                                                                        self.dropoutRateFC5PH:0.0})

    def setLearningRate(self, learningRate=None):
        if learningRate == None:
            self.learningRateVal = self.learningRateVal * self.learningRateDecay
        else:
            self.learningRateVal = learningRate
        return self.learningRateVal

    def saveParameters(self, fn):
        with self.g.as_default():
            self.saver = tf.train.Saver()
            self.saver.save(self.session, fn)

    def restoreParameters(self, fn):
        with self.g.as_default():
            self.saver = tf.train.Saver()
            self.saver.restore(self.session, fn)

    def summaryFileWriter(self, logsPath):
        summaryWriter = tf.summary.FileWriter(logsPath, graph=self.g)
        return summaryWriter

    def predict(self, XArray):
        #for i in range(len(batchX)):
        #    tf.image.per_image_standardization(XArray[i])
        base, zygosity, varType, indelLength = self.session.run( (self.YBaseChangeSigmoid, self.YZygositySoftmax, self.YVarTypeSoftmax, self.YIndelLengthSoftmax),
                                                                  feed_dict={self.XPH:XArray,
                                                                             self.learningRatePH:0.0,
                                                                             self.phasePH:False,
                                                                             self.dropoutRateFC4PH:0.0,
                                                                             self.dropoutRateFC5PH:0.0})
        return base, zygosity, varType, indelLength

    def __del__(self):
        self.session.close()

