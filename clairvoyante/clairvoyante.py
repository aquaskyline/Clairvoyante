import tensorflow as tf
import param

class Clairvoyante(object):

    def __init__(self, inputShape = (2*param.flankingBaseNum+1, 4, param.matrixNum),
                       outputShape1 = (4, ), outputShape2 = (5, ),
                       kernelSize1 = (2, 4), kernelSize2 = (3, 4),
                       pollSize1 = (param.flankingBaseNum, 1), pollSize2 = (3, 1),
                       filterNum = 32,
                       hiddenLayerUnitNumber = 32):
        self.inputShape = inputShape
        self.outputShape1 = outputShape1
        self.outputShape2 = outputShape2
        self.kernelSize1 = kernelSize1
        self.kernelSize2 = kernelSize2
        self.pollSize1 = pollSize1
        self.pollSize2 = pollSize2
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

            with tf.name_scope('Layers'):
                with tf.name_scope('conv1'):
                    conv1 = tf.layers.conv2d(
                    inputs=XPH,
                    filters=self.filterNum,
                    kernel_size=self.kernelSize1,
                    padding="same",
                    activation=tf.nn.elu)

                with tf.name_scope('pool1'):
                    pool1 = tf.layers.max_pooling2d(
                    inputs=conv1,
                    pool_size=self.pollSize1,
                    strides=1)

                with tf.name_scope('conv2'):
                    conv2 = tf.layers.conv2d(
                    inputs=pool1,
                    filters=self.filterNum,
                    kernel_size=self.kernelSize2,
                    padding="same",
                    activation=tf.nn.elu)

                with tf.name_scope('pool2'):
                    pool2 = tf.layers.max_pooling2d(
                    inputs=conv2,
                    pool_size=self.pollSize2,
                    strides=1)

                flat_size = ( (param.flankingBaseNum*2+1) - (self.pollSize1[0] - 1) - (self.pollSize2[0] - 1))
                flat_size *= ( 4 - (self.pollSize1[1] - 1) - (self.pollSize2[1] - 1))
                flat_size *= self.filterNum
                conv2_flat =  tf.reshape(pool2, [-1,  flat_size])

                with tf.name_scope('fc1'):
                    h1 = tf.layers.dense(inputs=conv2_flat, units=self.hiddenLayerUnitNumber, activation=tf.nn.elu)
                    dropout1 = tf.layers.dropout(inputs=h1, rate=param.dropoutRate)

                with tf.name_scope('fc2'):
                    h2 = tf.layers.dense(inputs=dropout1, units=self.hiddenLayerUnitNumber, activation=tf.nn.elu)
                    dropout2 = tf.layers.dropout(inputs=h2, rate=param.dropoutRate)

                with tf.name_scope('fc3'):
                    h3 = tf.layers.dense(inputs=dropout2, units=self.hiddenLayerUnitNumber, activation=tf.nn.elu)
                    dropout3 = tf.layers.dropout(inputs=h3, rate=param.dropoutRate)

                with tf.name_scope('Outputs'):
                    Y1 = tf.layers.dense(inputs=dropout3, units=self.outputShape1[0], activation=tf.nn.sigmoid)
                    Y2 = tf.layers.dense(inputs=dropout3, units=self.outputShape2[0], activation=tf.nn.elu)
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

            self.training_op = tf.train.AdamOptimizer(learning_rate=learningRatePH).minimize(loss)
            self.init_op = tf.global_variables_initializer()

    def init(self):
        self.session.run( self.init_op )

    def close(self):
        self.session.close()

    def train(self, batchX, batchY):
        loss, _ = self.session.run( (self.loss, self.training_op), feed_dict={self.XPH:batchX, self.YPH:batchY, self.learningRatePH:self.learningRateVal}) 
        return loss

    def getLoss(self, batchX, batchY):
        loss  = self.session.run( self.loss, feed_dict={self.XPH:batchX, self.YPH:batchY})    
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

    def predict(self, Xarray):
        with self.g.as_default():
            base, varType_  = self.session.run( (self.Y1, self.Y3), feed_dict={self.XPH:Xarray})
            return base, varType

    def __del__(self):
        self.session.close()

    def variableSummaries(self, var):
        # Attach a lot of summaries to a Tensor (for TensorBoard visualization).
        with tf.name_scope('Summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

