import os
import sys
import shlex
import argparse
import subprocess

def Run(args):
    CombineDatasets(args)


def CombineDatasets(args):

    ocant_fpo = open(args.tensor_can_out, "wb")
    ocant_fh = subprocess.Popen(shlex.split("gzip -c -1"), stdin=subprocess.PIPE, stdout=ocant_fpo, stderr=sys.stderr, bufsize=8388608)
    ovart_fpo = open(args.tensor_var_out, "wb")
    ovart_fh = subprocess.Popen(shlex.split("gzip -c -1"), stdin=subprocess.PIPE, stdout=ovart_fpo, stderr=sys.stderr, bufsize=8388608)
    ovar_fpo = open(args.var_out, "wb")
    ovar_fh = subprocess.Popen(shlex.split("gzip -c -1"), stdin=subprocess.PIPE, stdout=ovar_fpo, stderr=sys.stderr, bufsize=8388608)
    obed_fpo = open(args.bed_out, "wb")
    obed_fh = subprocess.Popen(shlex.split("gzip -c -1"), stdin=subprocess.PIPE, stdout=obed_fpo, stderr=sys.stderr, bufsize=8388608)

    prefix = "a"
    with open(args.input_list, "r") as l:
        for d in l:
            s = d.strip().split()

            print >> sys.stderr, "Combining %s ..." % (s[0])
            f = subprocess.Popen(shlex.split("gzip -fdc %s" % (s[0]) ), stdout=subprocess.PIPE, bufsize=8388608)
            for row in f.stdout: ocant_fh.stdin.write(prefix); ocant_fh.stdin.write(row);
            f.stdout.close(); f.wait()

            print >> sys.stderr, "Combining %s ..." % (s[1])
            f = subprocess.Popen(shlex.split("gzip -fdc %s" % (s[1]) ), stdout=subprocess.PIPE, bufsize=8388608)
            for row in f.stdout: ovart_fh.stdin.write(prefix); ovart_fh.stdin.write(row);
            f.stdout.close(); f.wait()

            print >> sys.stderr, "Combining %s ..." % (s[2])
            f = subprocess.Popen(shlex.split("gzip -fdc %s" % (s[2]) ), stdout=subprocess.PIPE, bufsize=8388608)
            for row in f.stdout: ovar_fh.stdin.write(prefix); ovar_fh.stdin.write(row);
            f.stdout.close(); f.wait()

            print >> sys.stderr, "Combining %s ..." % (s[3])
            f = subprocess.Popen(shlex.split("gzip -fdc %s" % (s[3]) ), stdout=subprocess.PIPE, bufsize=8388608)
            for row in f.stdout: obed_fh.stdin.write(prefix); obed_fh.stdin.write(row);
            f.stdout.close(); f.wait()

            prefix = chr(ord(prefix)+1)

    ocant_fh.stdin.close(); ocant_fh.wait(); ocant_fpo.close()
    ovart_fh.stdin.close(); ovart_fh.wait(); ovart_fpo.close()
    ovar_fh.stdin.close(); ovar_fh.wait(); ovar_fpo.close()
    obed_fh.stdin.close(); obed_fh.wait(); obed_fpo.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Combine datasets for model training" )

    parser.add_argument('--input_list', type=str, default = None,
            help="A list with 4 columns of filenames. <Tensors generated at randome genome positions genearted by ExtractVariantCandidates.py+CreateTensor.py> <Variant tensors generated GetTruth.py+CreateTensor.py> <Truth variants genearted by GetTruth.py> <Usable genome regions in BED format>")

    parser.add_argument('--tensor_can_out', type=str, default = None,
            help="Combined tensors generated at randome genome positions")

    parser.add_argument('--tensor_var_out', type=str, default = None,
            help="Combined variant tensors output")

    parser.add_argument('--var_out', type=str, default = None,
            help="Combined Truth variants list output")

    parser.add_argument('--bed_out', type=str, default = None,
            help="Combined usable genome regions output")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    Run(args)

