NUM_THREADS = 12
maxEpoch = 10000
parameterOutputPlaceHolder = 6

# Tensor related parameters, please use the same values for creating tensor, model training and variant calling
flankingBaseNum = 16        # Please change this value in the dataPrepScripts at the same time
matrixNum = 4               # Please change this value in the dataPrepScripts at the same time
bloscBlockSize = 500

# Model hyperparameters
trainBatchSize = 10000
predictBatchSize = 1000
initialLearningRate = 0.001
learningRateDecay = 0.1
maxLearningRateSwitch = 3
trainingDatasetPercentage = 0.9

# Clairvoyante v3 specific
l2RegularizationLambda = 0.001
l2RegularizationLambdaDecay = 0.1
dropoutRateFC4 = 0.5
dropoutRateFC5 = 0.0

# Clairvoyante v2 specific
dropoutRate = 0.05

# Global functions
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import sys
        raise sys.exit('Boolean value expected.')
