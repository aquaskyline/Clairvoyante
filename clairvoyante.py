#!/usr/bin/env python

import os
import sys
import importlib


if sys.version_info.major >= 3:
    clv_path = os.path.dirname(os.path.abspath(__file__))+os.sep+'clairvoyante'
    sys.path.insert(1, clv_path)

def mod(dir, name):
    if sys.argv[1] == name:
        r = importlib.import_module("%s.%s" % (dir, name))
        sys.argv = sys.argv[1:]
        sys.argv[0] += (".py")
        r.main()
        sys.exit(0)

cl = ["callVarBamParallel", "callVarBam", "callVar", "calTrainDevDiff", "demoRun", "evaluateListOfModels", "evaluate", "getEmbedding", "getTensorAndLayerPNG", "tensor2Bin", "trainNonstop", "train", "trainWithoutValidationNonstop"]
dp = ["ChooseItemInBed", "CombineMultipleDatasetsForTraining", "CountNumInBed", "CreateTensor", "ExtractVariantCandidates", "GetTruth", "PairWithNonVariants", "RandomSampling"]

def main():
    if len(sys.argv) <= 1:
        print ("Clairvoyante submodule invocator:")
        print ("  Usage: clairvoyante.py SubmoduleName [Options of the submodule]")
        print ("")
        print ("Available data preparation submodules:")
        for n in dp: print ("  - %s" % n)
        print ("")
        print ("Available clairvoyante submodules:")
        for n in cl: print ("  - %s" % n)
        print ("")
        print ("Data preparation scripts:")
        print ("%s/dataPrepScripts" % os.path.dirname(os.path.abspath(sys.argv[0])))
        print ("")
        print ("Clairvoyante scripts:")
        print ("%s/clairvoyante" % os.path.dirname(os.path.abspath(sys.argv[0])))
        print ("")
        sys.exit(0)

    for n in cl: mod("clairvoyante", n)
    for n in dp: mod("dataPrepScripts", n)

if __name__ == "__main__":
    main()
