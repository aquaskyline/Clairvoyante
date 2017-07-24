import tensorflow as tf
import selu
import param

class Clairvoyante(object):

    def __init__(self, inputShape = (2*param.flankingBaseNum+1, 4, param.matrixNum),
                       outputShape1 = (4, ), outputShape2 = (5, ),
                       kernelSize1 = (2, 4), kernelSize2 = (3, 4),
                       pollSize1 = (param.flankingBaseNum, 1), pollSize2 = (3, 1),
                       filterNum = 48,
                       hiddenLayerUnitNumber = 48):
        self.inputShape = inputShape
        self.outputShape1 = outputShape1; self.outputShape2 = outputShape2
        self.kernelSize1 = kernelSize1; self.kernelSize2 = kernelSize2
        self.pollSize1 = pollSize1; self.pollSize2 = pollSize2
        self.filterNum = filterNum
        self.hiddenLayerUnitNumber = hiddenLayerUnitNumber
        self.learningRateVal = param.initialLearningRate
        self.learningRateDecay = param.learningRateDecay
        self.g = tf.Graph()
        self._buildGraph()
        self.session = tf.Session(graph = self.g)

    def _buildGraph(self):
        with self.g.as_default():
            with tf.name_scope('Inputs'):
                XPH = tf.placeholder(tf.float32, [None, self.inputShape[0], self.inputShape[1], self.inputShape[2]])
                self.XPH = XPH

                YPH = tf.placeholder(tf.float32, [None, self.outputShape1[0] + self.outputShape2[0]])
                self.YPH = YPH

                learningRatePH = tf.placeholder(tf.float32, shape=[])
                self.learningRatePH = learningRatePH

                phasePH = tf.placeholder(tf.bool, shape=[])
                self.phasePH = phasePH

            with tf.name_scope('Layers'):
                with tf.name_scope('conv1'):
                    conv1 = tf.layers.conv2d(inputs=XPH,
                                             filters=self.filterNum,
                                             kernel_size=self.kernelSize1,
                                             kernel_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                                             padding="same",
                                             activation=selu.selu)

                with tf.name_scope('pool1'):
                    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                                    pool_size=self.pollSize1,
                                                    strides=1)

                with tf.name_scope('conv2'):
                    conv2 = tf.layers.conv2d(inputs=pool1,
                                             filters=self.filterNum,
                                             kernel_size=self.kernelSize2,
                                             kernel_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                                             padding="same",
                                             activation=selu.selu)

                with tf.name_scope('pool2'):
                    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                                    pool_size=self.pollSize2,
                                                    strides=1)

                flat_size = ( (param.flankingBaseNum*2+1) - (self.pollSize1[0] - 1) - (self.pollSize2[0] - 1))
                flat_size *= ( 4 - (self.pollSize1[1] - 1) - (self.pollSize2[1] - 1))
                flat_size *= self.filterNum
                conv2_flat =  tf.reshape(pool2, [-1,  flat_size])

                with tf.name_scope('fc3'):
                    h3 = tf.layers.dense(inputs=conv2_flat,
                                         units=self.hiddenLayerUnitNumber,
                                         kernel_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                                         activation=selu.selu)

                with tf.name_scope("dropout3"):
                    dropout3 = selu.dropout_selu(h3, param.dropoutRate, training=phasePH)

                with tf.name_scope('fc4'):
                    h4 = tf.layers.dense(inputs=dropout3,
                                         units=self.hiddenLayerUnitNumber,
                                         kernel_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                                         activation=selu.selu)

                with tf.name_scope("dropout4"):
                    dropout4 = selu.dropout_selu(h4, param.dropoutRate, training=phasePH)

                with tf.name_scope('fc5'):
                    h5 = tf.layers.dense(inputs=dropout4,
                                         units=self.hiddenLayerUnitNumber,
                                         kernel_initializer = tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                                         activation=selu.selu)

                with tf.name_scope("dropout5"):
                    dropout5 = selu.dropout_selu(h5, param.dropoutRate, training=phasePH)

                with tf.name_scope('Outputs'):
                    Y1 = tf.layers.dense(inputs=dropout5, units=self.outputShape1[0], activation=tf.nn.sigmoid)
                    Y2 = tf.layers.dense(inputs=dropout5, units=self.outputShape2[0], activation=selu.selu)
                    Y3 = tf.nn.softmax(Y2)
                    self.Y1 = Y1
                    self.Y3 = Y3

                with tf.name_scope("Losses"):
                    loss1 = tf.reduce_sum( tf.pow( Y1 - tf.slice(YPH,[0,0],[-1,self.outputShape1[0]] ), 2) )
                    loss2 = tf.reduce_sum( tf.nn.softmax_cross_entropy_with_logits( logits=Y2,
                                                                                    labels=tf.slice( YPH, [0,self.outputShape1[0]],
                                                                                                          [-1,self.outputShape2[0]] ) ) )
                    loss = loss1 + loss2
                    self.loss = loss

            # add summaries
            tf.summary.scalar('learning_rate', learningRatePH)
            tf.summary.scalar("loss1", loss1)
            tf.summary.scalar("loss2", loss2)
            tf.summary.scalar("loss", loss)
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
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
        loss = 0
        loss, _, summary = self.session.run( (self.loss, self.training_op, self.merged_summary_op),
                                        feed_dict={self.XPH:batchX, self.YPH:batchY, self.learningRatePH:self.learningRateVal,
                                                    self.phasePH:True})
        return loss, summary

    def getLoss(self, batchX, batchY):
        #for i in range(len(batchX)):
        #    tf.image.per_image_standardization(batchX[i])
        loss = 0
        loss  = self.session.run( self.loss, feed_dict={self.XPH:batchX, self.YPH:batchY, self.phasePH:False})
        return loss

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
        summaryWriter = tf.summary.FileWriter(logsPath, graph=tf.get_default_graph())
        return summaryWriter

    def predict(self, XArray):
        #for i in range(len(batchX)):
        #    tf.image.per_image_standardization(XArray[i])
        base, varType  = self.session.run( (self.Y1, self.Y3), feed_dict={self.XPH:XArray, self.phasePH:False})
        return base, varType

    def __del__(self):
        self.session.close()

