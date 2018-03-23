flankingBaseNum = 16
matrixNum = 4
expandReferenceRegion = 1000000


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        import sys
        raise sys.exit('Boolean value expected.')
