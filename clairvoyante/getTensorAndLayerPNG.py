import sys
import os
import argparse
import math
import numpy as np
import param
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def Prepare(args):
    import utils_v2 as utils # v3 network is using v2 utils
    if args.slim == True:
        import clairvoyante_v3_slim as cv
    else:
        import clairvoyante_v3 as cv

    utils.SetupEnv()
    m = cv.Clairvoyante()
    m.init()

    m.restoreParameters(args.chkpnt_fn)

    total, XArrayCompressed, YArrayCompressed, posArrayCompressed = \
    utils.GetTrainingArray(args.tensor_fn, args.var_fn, None)

    return m, utils, total, XArrayCompressed, YArrayCompressed, posArrayCompressed


def GetActivations(layer, batchX, m):
    # Version 3 network
    units = m.session.run(layer, feed_dict={m.XPH:batchX,
                                            m.phasePH:False,
                                            m.dropoutRateFC4PH:0.0,
                                            m.dropoutRateFC5PH:0.0,
                                            m.l2RegularizationLambdaPH:0.0})
    return units


def PlotFiltersConv(ofn, units, interval=1, xsize=18, ysize=20, xts=1, yts=1, lts=2):
    matplotlib.rc('xtick', labelsize=xts)
    matplotlib.rc('ytick', labelsize=yts)
    matplotlib.rc('axes', titlesize=lts)
    filters = units.shape[3]
    xlen = units.shape[2]
    plot = plt.figure(1, figsize=(xsize,ysize))
    nColumns = 8
    nRows = math.ceil(filters / nColumns) + 1
    for i in range(filters):
        plt.subplot(nRows, nColumns, i+1)
        plt.title('Filter ' + str(i))
        plt.xticks(np.arange(0, xlen, interval), ['A','C','G','T'])
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap=plt.cm.bwr)
    cax = plt.axes([0.92, 0.4, 0.01, 0.3])
    plt.colorbar(cax=cax)
    plot.savefig(ofn, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(plot)


def PlotFiltersFC(ofn, units, interval=10, xsize=18, ysize=4, xts=1, yts=1, lts=2):
    matplotlib.rc('xtick', labelsize=xts)
    matplotlib.rc('ytick', labelsize=yts)
    matplotlib.rc('axes', titlesize=lts)
    plot = plt.figure(1, figsize=(xsize,ysize))
    cell = units.shape[1]
    plt.xticks(np.arange(0, cell, interval))
    plt.yticks(np.arange(0, 1, 1), [''])
    plt.title(str(cell) + ' units')
    plt.imshow(np.reshape(units[0,:], (-1,cell)), interpolation="nearest", cmap=plt.cm.bwr)
    cax = plt.axes([0.45, 0.05, 0.2, 0.05])
    plt.colorbar(cax=cax, orientation="horizontal")
    plot.savefig(ofn, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(plot)


def PlotOutputArray(ofn, unitsX, unitsY, interval=1, xsize=8, ysize=2, xts=1, yts=1, lts=2):
    matplotlib.rc('xtick', labelsize=xts)
    matplotlib.rc('ytick', labelsize=yts)
    matplotlib.rc('axes', titlesize=lts)
    plot = plt.figure(1, figsize=(xsize,ysize))
    cell = unitsX.shape[1]
    plt.subplot(2,1,1)
    plt.xticks(np.arange(0, cell, interval), ["A","C","G","T","HET","HOM","REF","SNP","INS","DEL","0","1","2","3","4",">4"])
    plt.yticks(np.arange(0, 1, 1), [''])
    plt.title("Predicted")
    plt.imshow(np.reshape(unitsX[0,:], (-1,cell)), interpolation="nearest", cmap=plt.cm.bwr)
    plt.subplot(2,1,2)
    plt.xticks(np.arange(0, cell, interval), ["A","C","G","T","HET","HOM","REF","SNP","INS","DEL","0","1","2","3","4",">4"])
    plt.yticks(np.arange(0, 1, 1), [''])
    plt.title("Truth")
    plt.imshow(np.reshape(unitsY[0,:], (-1,cell)), interpolation="nearest", cmap=plt.cm.bwr)
    plt.colorbar(orientation="horizontal", pad=0.5)
    plot.savefig(ofn, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(plot)


def PlotTensor(ofn, XArray):
    plot = plt.figure(figsize=(15, 8));
    plt.subplot(4,1,1); plt.xticks(np.arange(0, 33, 1)); plt.yticks(np.arange(0, 4, 1), ['A','C','G','T'])
    plt.imshow(XArray[0,:,:,0].transpose(), vmin=0, vmax=50, interpolation="nearest", cmap=plt.cm.hot); plt.colorbar()
    plt.subplot(4,1,2); plt.xticks(np.arange(0, 33, 1)); plt.yticks(np.arange(0, 4, 1), ['A','C','G','T'])
    plt.imshow(XArray[0,:,:,1].transpose(), vmin=-50, vmax=50, interpolation="nearest", cmap=plt.cm.bwr); plt.colorbar()
    plt.subplot(4,1,3); plt.xticks(np.arange(0, 33, 1)); plt.yticks(np.arange(0, 4, 1), ['A','C','G','T'])
    plt.imshow(XArray[0,:,:,2].transpose(), vmin=-50, vmax=50, interpolation="nearest", cmap=plt.cm.bwr); plt.colorbar()
    plt.subplot(4,1,4); plt.xticks(np.arange(0, 33, 1)); plt.yticks(np.arange(0, 4, 1), ['A','C','G','T'])
    plt.imshow(XArray[0,:,:,3].transpose(), vmin=-50, vmax=50, interpolation="nearest", cmap=plt.cm.bwr); plt.colorbar()
    plot.savefig(ofn, dpi=300, transparent=True, bbox_inches='tight')
    plt.close(plot)


def CreatePNGs(args, m, utils, total, XArrayCompressed, YArrayCompressed, posArrayCompressed):
    for i in range(total):
        XArray, _, _ = utils.DecompressArray(XArrayCompressed, i, 1, total)
        YArray, _, _ = utils.DecompressArray(YArrayCompressed, i, 1, total)
        posArray, _, _ = utils.DecompressArray(posArrayCompressed, i, 1, total)
        varName = posArray[0]
        varName = "-".join(varName.split(":"))
        print >> sys.stderr, "Plotting %s..." % (varName)
        # Create folder
        if not os.path.exists(varName):
            os.makedirs(varName)
        # Plot tensors
        PlotTensor(varName+"/tensor.png", XArray)
        # Plot conv1
        units = GetActivations(m.conv1, XArray, m)
        PlotFiltersConv(varName+"/conv1.png", units, 1, 8, 9, 5, 5, 8)
        # Plot conv2
        units = GetActivations(m.conv2, XArray, m)
        PlotFiltersConv(varName+"/conv2.png", units, 1, 8, 18, 6, 6, 9)
        # Plot conv3
        units = GetActivations(m.conv3, XArray, m)
        PlotFiltersConv(varName+"/conv3.png", units, 1, 8, 24, 7, 7, 10)
        # Plot fc4
        units = GetActivations(m.fc4, XArray, m)
        PlotFiltersFC(varName+"/fc4.png", units, 10, 16, 1, 9, 9, 10)
        # Plot fc5
        units = GetActivations(m.fc5, XArray, m)
        PlotFiltersFC(varName+"/fc5.png", units, 10, 4, 1, 4, 4, 5)
        # Plot Predicted and Truth Y
        unitsX = [GetActivations(m.YBaseChangeSigmoid, XArray, m),\
                  GetActivations(m.YZygositySoftmax, XArray, m),\
                  GetActivations(m.YVarTypeSoftmax, XArray, m),\
                  GetActivations(m.YIndelLengthSoftmax, XArray, m)]
        unitsX =  np.concatenate(unitsX, axis=1)
        unitsY = np.reshape(YArray[0], (1,-1))
        PlotOutputArray(varName+"/output.png", unitsX, unitsY, 1, 4, 2, 4, 4, 5)


def ParseArgs():
    parser = argparse.ArgumentParser(
            description="Visualize tensors and hidden layers in PNG" )

    parser.add_argument('--tensor_fn', type=str, default = "vartensors",
            help="Tensor input")

    parser.add_argument('--var_fn', type=str, default = None,
            help="Truth variants input")

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a checkpoint for testing or continue training")

    parser.add_argument('--slim', type=param.str2bool, nargs='?', const=True, default = False,
            help="Train using the slim version of Clairvoyante, default: False")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    return args


def main():
    args = ParseArgs()
    m, utils, total, XArrayCompressed, YArrayCompressed, posArrayCompressed = Prepare(args)
    CreatePNGs(args, m, utils, total, XArrayCompressed, YArrayCompressed, posArrayCompressed)


if __name__ == "__main__":
    main()


