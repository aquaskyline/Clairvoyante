import sys
import argparse
import logging
import pickle
import param

logging.basicConfig(format='%(message)s', level=logging.INFO)

def Run(args):
    if args.v2 == True or args.v3 == True:
        import utils_v2 as utils
    utils.SetupEnv()
    Convert(args, utils)


def Convert(args, utils):
    logging.info("Loading the dataset ...")
    total, XArrayCompressed, YArrayCompressed, posArrayCompressed = \
    utils.GetTrainingArray(args.tensor_fn,
                           args.var_fn,
                           args.bed_fn)

    logging.info("Writing to binary ...")
    fh = open(args.bin_fn, "wb")
    pickle.dump(total,fh)
    pickle.dump(XArrayCompressed,fh)
    pickle.dump(YArrayCompressed,fh)
    pickle.dump(posArrayCompressed,fh)


def main():
    parser = argparse.ArgumentParser(
            description="Generate a binary format input tensor" )

    parser.add_argument('--tensor_fn', type=str, default = "vartensors",
            help="Tensor input")

    parser.add_argument('--var_fn', type=str, default = "truthvars",
            help="Truth variants list input")

    parser.add_argument('--bed_fn', type=str, default = None,
            help="High confident genome regions input in the BED format")

    parser.add_argument('--bin_fn', type=str, default = None,
            help="Output a binary tensor file")

    parser.add_argument('--v3', type=param.str2bool, nargs='?', const=True, default = True,
            help="Use Clairvoyante version 3")

    parser.add_argument('--v2', type=param.str2bool, nargs='?', const=True, default = False,
            help="Use Clairvoyante version 2")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)

if __name__ == "__main__":
    main()
