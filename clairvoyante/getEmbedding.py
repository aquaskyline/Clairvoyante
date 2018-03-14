import sys
import time
import argparse
import param
import logging
import pickle
import math
import numpy as np

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import os
home_dir = os.path.expanduser('~')
sys.path.append(home_dir+"/.local/lib/python2.7/site-packages/")

def prepare_data(args):
    import utils_v2 as utils # v3 network is using v2 utils
    if args.slim == True:
        import clairvoyante_v3_slim as cv
    else:
        import clairvoyante_v3 as cv

    utils.SetupEnv()
    m = cv.Clairvoyante()
    m.init()

    m.restoreParameters(args.chkpnt_fn)

    if args.bin_fn != None:
        with open(args.bin_fn, "rb") as fh:
            total = pickle.load(fh)
            XArrayCompressed = pickle.load(fh)
            YArrayCompressed = pickle.load(fh)
            posArrayCompressed = pickle.load(fh)
    else:
        total, XArrayCompressed, YArrayCompressed, posArrayCompressed = \
        utils.GetTrainingArray(args.tensor_fn,
                               args.var_fn,
                               args.bed_fn)

    return m, utils, total, XArrayCompressed, YArrayCompressed, posArrayCompressed

def write_metadata(args, fn, labels):
    _dir = os.path.dirname(fn)
    _dir += "/"
    _dir += os.path.basename(os.path.normpath(args.olog_dir))
    _fn = _dir + "/" + os.path.basename(fn)
    if not tf.gfile.Exists(_dir):
        tf.gfile.MakeDirs(_dir)
    with open(_fn, 'w') as f:
        for label in labels:
            f.write("{}\n".format(label))

def get_embeddings(m, XArray):
    embeddings1, embeddings2, embeddings3, embeddings4 = m.session.run((m.embedding1, m.embedding2, m.embedding3, m.embedding4), feed_dict={m.XPH:XArray, m.phasePH:False, m.dropoutRateFC4PH:0.0, m.dropoutRateFC5PH:0.0, m.l2RegularizationLambdaPH:0.0, m.learningRatePH:0.0})
    return embeddings1, embeddings2, embeddings3, embeddings4

def get_labels(YBatch):
    dict2 = {0:'HET', 1:'HOM'}
    dict3 = {0: 'REF', 1:'SNP', 2:'INS', 3:'DEL'}
    dict4 = {0:'0', 1:'1', 2:'2', 3:'3', 4:'4', 5:'>4'}
    labels1 = []
    labels2 = []
    labels3 = []
    labels4 = []
    labels1.append("A\tC\tG\tT")
    for l in YBatch[:, 0:4]:
        labels1.append("%f\t%f\t%f\t%f" % (l[0], l[1], l[2], l[3]))
    for l in YBatch[:, 4:6]:
        for i in xrange(2):
            if l[i] == 1:
                labels2.append(dict2[i])
    for l in YBatch[:, 6:10]:
        for i in xrange(4):
            if l[i] == 1:
                labels3.append(dict3[i])
    for l in YBatch[:, 10:16]:
        for i in xrange(6):
            if l[i] == 1:
                labels4.append(dict4[i])
    return labels1, labels2, labels3, labels4


def visualize_embedding(args, m, utils, total, XArrayCompressed, YArrayCompressed,
                        olog_dir, embed_count):
    XBatch, XNum, XEndFlag = utils.DecompressArray(XArrayCompressed, 0, embed_count, total)
    YBatch, YNum, YEndFlag = utils.DecompressArray(YArrayCompressed, 0, embed_count, total)

    embeddings1, embeddings2, embeddings3, embeddings4 = get_embeddings(m, XBatch)
    labels1, labels2, labels3, labels4 = get_labels(YBatch)

    embedding1_values = embeddings1
    embedding1_labels = labels1
    embedding1_values = np.asarray(embedding1_values)
    embedding1_var = tf.Variable(embedding1_values, name="BaseChange")
    embedding2_values = embeddings2
    embedding2_labels = labels2
    embedding2_values = np.asarray(embedding2_values)
    embedding2_var = tf.Variable(embedding2_values, name="Zygosity")
    embedding3_values = embeddings3
    embedding3_labels = labels3
    embedding3_values = np.asarray(embedding3_values)
    embedding3_var = tf.Variable(embedding3_values, name="VarType")
    embedding4_values = embeddings4
    embedding4_labels = labels4
    embedding4_values = np.asarray(embedding4_values)
    embedding4_var = tf.Variable(embedding4_values, name="IndelLength")


    metadata1_path = os.path.join(olog_dir, 'BaseChange.tsv')
    write_metadata(args, metadata1_path, embedding1_labels)
    metadata2_path = os.path.join(olog_dir, 'Zygosity.tsv')
    write_metadata(args, metadata2_path, embedding2_labels)
    metadata3_path = os.path.join(olog_dir, 'VarType.tsv')
    write_metadata(args, metadata3_path, embedding3_labels)
    metadata4_path = os.path.join(olog_dir, 'IndelLength.tsv')
    write_metadata(args, metadata4_path, embedding4_labels)

    checkpoint_path = os.path.join(olog_dir, 'model.ckpt')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, checkpoint_path, 1)

    config = projector.ProjectorConfig()
    embedding1 = config.embeddings.add()
    embedding1.tensor_name = embedding1_var.name
    embedding1.metadata_path = metadata1_path
    embedding2 = config.embeddings.add()
    embedding2.tensor_name = embedding2_var.name
    embedding2.metadata_path = metadata2_path
    embedding3 = config.embeddings.add()
    embedding3.tensor_name = embedding3_var.name
    embedding3.metadata_path = metadata3_path
    embedding4 = config.embeddings.add()
    embedding4.tensor_name = embedding4_var.name
    embedding4.metadata_path = metadata4_path
    summary_writer = tf.summary.FileWriter(olog_dir, sess.graph)
    projector.visualize_embeddings(summary_writer, config)


def get_arguements():
    parser = argparse.ArgumentParser(
            description="Get Embedding for TensorBoard Visualization" )

    parser.add_argument('--bin_fn', type=str, default = None,
            help="Binary tensor input generated by tensor2Bin.py, tensor_fn, var_fn and bed_fn will be ignored")

    parser.add_argument('--tensor_fn', type=str, default = "vartensors",
            help="Tensor input")

    parser.add_argument('--var_fn', type=str, default = "truthvars",
            help="Truth variants list input")

    parser.add_argument('--bed_fn', type=str, default = None,
            help="High confident genome regions input in the BED format")

    parser.add_argument('--chkpnt_fn', type=str, default = None,
            help="Input a checkpoint for testing or continue training")

    parser.add_argument('--olog_dir', type=str, default = None,
            help="Directory for tensorboard log outputs")

    parser.add_argument('--slim', type=param.str2bool, nargs='?', const=True, default = False,
            help="Train using the slim version of Clairvoyante, default: False")

    parser.add_argument('--count', type=int, default = 10000,
            help="Number of variants to be visualized, default: 10000")

    args = parser.parse_args()

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        sys.exit(1)

    return args


def main():
    args = get_arguements()
    m, utils, total, XArrayCompressed, YArrayCompressed, posArrayCompressed = prepare_data(args)
    visualize_embedding(args, m, utils, total, XArrayCompressed, YArrayCompressed, args.olog_dir, args.count)


if __name__ == "__main__":
    main()
