import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

from imageio import imwrite
from scipy.special import softmax
from tensorflow.contrib.training import HParams
from tqdm import tqdm

from igpt.src.model import model
from igpt.src.utils import iter_data, count_parameters

class RunConfig:s
    # data and I/O
    data_path : str = "/root/downloads/imagenet"
    ckpt_path : str = "downloads/model/" + "model.ckpt-1000000"
    color_cluster_path : str = "downloads/clusters/" + "kmeans_centers.npy"
    save_dir : str = "results/"

    # model
    n_embd : int = 512 
    n_head : int = 8
    n_layer : int = 24
    n_px : int = 32 # image height or width in pixels
    n_vocab : int = 512 # possible values for each pixel

    bert : bool = False # evaluates the model, requires a checkpoint and dataset
    bert_mask_prob : float = 0.15 
    clf : bool = False # add a learnable classification head

    # parallelism
    n_sub_batch : int = 8 # per-gpu batch size
    n_gpu : int = 1 # number of gpus to distribute training across

    # mode 
    eval : bool = False # evaluates the model, requires a checkpoint and dataset
    sample : bool = False # samples from the model, requires a checkpoint and clusters

    # reproducibility
    seed : int = 42 # random seed for random, numpy and tensorflow

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def load_data(data_path):
    trX = np.load(f'{data_path}_trX.npy')
    trY = np.load(f'{data_path}_trY.npy')
    vaX = np.load(f'{data_path}_vaX.npy')
    vaY = np.load(f'{data_path}_vaY.npy')
    teX = np.load(f'{data_path}_teX.npy')
    teY = np.load(f'{data_path}_teY.npy')
    return (trX, trY), (vaX, vaY), (teX, teY)


def set_hparams(args):
    return HParams(
        n_ctx=args.n_px*args.n_px,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        n_vocab=args.n_vocab,
        bert=args.bert,
        bert_mask_prob=args.bert_mask_prob,
        clf=args.clf,
    )


def create_model(x, y, n_gpu, hparams):
    gen_logits = []
    gen_loss = []
    clf_loss = []
    tot_loss = []
    accuracy = []

    trainable_params = None
    for i in range(n_gpu):
        with tf.device("/gpu:%d" % i):
            results = model(hparams, x[i], y[i], reuse=(i != 0))

            gen_logits.append(results["gen_logits"])
            gen_loss.append(results["gen_loss"])
            clf_loss.append(results["clf_loss"])

            if hparams.clf:
                tot_loss.append(results["gen_loss"] + results["clf_loss"])
            else:
                tot_loss.append(results["gen_loss"])

            accuracy.append(results["accuracy"])

            if i == 0:
                trainable_params = tf.trainable_variables()
                print("trainable parameters:", count_parameters())

    return trainable_params, gen_logits, gen_loss, clf_loss, tot_loss, accuracy


def reduce_mean(gen_loss, clf_loss, tot_loss, accuracy, n_gpu):
    with tf.device("/gpu:0"):
        for i in range(1, n_gpu):
            gen_loss[0] += gen_loss[i]
            clf_loss[0] += clf_loss[i]
            tot_loss[0] += tot_loss[i]
            accuracy[0] += accuracy[i]
        gen_loss[0] /= n_gpu
        clf_loss[0] /= n_gpu
        tot_loss[0] /= n_gpu
        accuracy[0] /= n_gpu


def evaluate(sess, evX, evY, X, Y, gen_loss, clf_loss, accuracy, n_batch, desc, permute=False):
    metrics = []
    for xmb, ymb in iter_data(evX, evY, n_batch=n_batch, truncate=True, verbose=True):
        metrics.append(sess.run([gen_loss[0], clf_loss[0], accuracy[0]], {X: xmb, Y: ymb}))
    eval_gen_loss, eval_clf_loss, eval_accuracy = [np.mean(m) for m in zip(*metrics)]
    print(f"{desc} gen: {eval_gen_loss:.4f} clf: {eval_clf_loss:.4f} acc: {eval_accuracy:.2f}")


# naive sampler without caching
def sample(sess, X, gen_logits, n_sub_batch, n_gpu, n_px, n_vocab, clusters, save_dir, primer):
    samples = np.zeros([n_gpu * n_sub_batch, n_px * n_px], dtype=np.int32)
    # samples is array where we collect generate pixels, the shape is : [BATCH, (32*32)]
    primers = n_sub_batch * [primer]
    if primers is not None:
        # we add primers to the samples
        samples[:, :len(primer)] = primers
                        
    for i in tqdm(range(len(primer), n_px * n_px), ncols=80, leave=False):
        np_gen_logits = sess.run(gen_logits, {X: samples})
        for j in range(n_gpu):
            p = softmax(np_gen_logits[j][:, i, :], axis=-1)  # logits to probas
            for k in range(n_sub_batch):
                c = np.random.choice(n_vocab, p=p[k])  # choose based on probas
                samples[j * n_sub_batch + k, i] = c
    
    # dequantize
    samples = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [32, 32, 3]).astype(np.uint8) for s in samples]

    # write to png
    for i in range(n_gpu * n_sub_batch):
        imwrite(f"{args.save_dir}/sample_{i}.png", samples[i])


def main(args, primer=None):
    set_seed(args.seed)

    n_batch = args.n_sub_batch * args.n_gpu

    if args.data_path.endswith("cifar10"):
        n_class = 10
    elif args.data_path.endswith("imagenet"):
        n_class = 1000
    else:
        raise ValueError("Dataset not supported.")

    X = tf.placeholder(tf.int32, [n_batch, args.n_px * args.n_px])
    Y = tf.placeholder(tf.float32, [n_batch, n_class])

    x = tf.split(X, args.n_gpu, 0)
    y = tf.split(Y, args.n_gpu, 0)

    hparams = set_hparams(args)
    trainable_params, gen_logits, gen_loss, clf_loss, tot_loss, accuracy = create_model(x, y, args.n_gpu, hparams)
    reduce_mean(gen_loss, clf_loss, tot_loss, accuracy, args.n_gpu)

    saver = tf.train.Saver(var_list=[tp for tp in trainable_params if not 'clf' in tp.name])
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        sess.run(tf.global_variables_initializer())

        saver.restore(sess, args.ckpt_path)

        if args.eval:
            (trX, trY), (vaX, vaY), (teX, teY) = load_data(args.data_path)
            evaluate(sess, trX[:len(vaX)], trY[:len(vaY)], X, Y, gen_loss, clf_loss, accuracy, n_batch, "train")
            evaluate(sess, vaX, vaY, X, Y, gen_loss, clf_loss, accuracy, n_batch, "valid")
            evaluate(sess, teX, teY, X, Y, gen_loss, clf_loss, accuracy, n_batch, "test")

        if args.sample:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            clusters = np.load(args.color_cluster_path)
            sample(sess, X, gen_logits, args.n_sub_batch, args.n_gpu, args.n_px, args.n_vocab, clusters, args.save_dir, primer)