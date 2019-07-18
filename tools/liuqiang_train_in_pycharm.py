# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi He, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.train_val import get_training_roidb, train_net
from model.config import cfg, cfg_from_file, cfg_from_list, get_output_dir, get_output_tb_dir
from datasets.factory import get_imdb
import datasets.imdb
import argparse
import pprint
import numpy as np
import sys

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1


def combined_roidb(imdb_names):
    """
    Combine multiple roidbs
    """

    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print('Loaded dataset `{:s}` for training'.format(imdb.name))
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]

    roidb = roidbs[0]
    if len(roidbs) > 1:
        for r in roidbs[1:]:
            roidb.extend(r)
        tmp = get_imdb(imdb_names.split('+')[1])
        imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb


if __name__ == '__main__':

    cfg_file = '/mnt/data/tf-faster-rcnn-master/experiments/cfgs/vgg16.yml'
    set_cfgs = ['ANCHOR_SCALES', '[8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'TRAIN.STEPSIZE', '[40000]']
    imdb_name = 'voc_2007_trainval'
    imdbval_name = 'voc_2007_test'
    max_iters = 20000
    net = 'vgg16'
    tag = None
    max_per_image = 100
    weight = '/mnt/data//tf-faster-rcnn-master/data/imagenet_weights/vgg16.ckpt'

    if cfg_file is not None:
        cfg_from_file(cfg_file)
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)

    print('Using config:')
    pprint.pprint(cfg)

    np.random.seed(cfg.RNG_SEED)

    # train set
    imdb, roidb = combined_roidb(imdb_name)
    print('{:d} roidb entries'.format(len(roidb)))

    # output directory where the models are saved
    output_dir = get_output_dir(imdb, tag)
    print('Output will be saved to `{:s}`'.format(output_dir))

    # tensorboard directory where the summaries are saved during training
    tb_dir = get_output_tb_dir(imdb, tag)
    print('TensorFlow summaries will be saved to `{:s}`'.format(tb_dir))

    # also add the validation set, but with no flipping images
    orgflip = cfg.TRAIN.USE_FLIPPED
    cfg.TRAIN.USE_FLIPPED = True
    _, valroidb = combined_roidb(imdbval_name)
    print('{:d} validation roidb entries'.format(len(valroidb)))
    cfg.TRAIN.USE_FLIPPED = orgflip

    # load network
    if net == 'vgg16':
        net = vgg16()
    elif net == 'res50':
        net = resnetv1(num_layers=50)
    elif net == 'res101':
        net = resnetv1(num_layers=101)
    elif net == 'res152':
        net = resnetv1(num_layers=152)
    elif net == 'mobile':
        net = mobilenetv1()
    else:
        raise NotImplementedError

    train_net(net, imdb, roidb, valroidb, output_dir, tb_dir,
              pretrained_model=weight,
              max_iters=max_iters)
