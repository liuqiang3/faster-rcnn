# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys
from datasets.self_dataset import Dataset

import tensorflow as tf
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

def get_imdb(imdb_name):
    imdb_name, image_set = imdb_name.split('_')
    imdb= Dataset(imdb_name,image_set)
    print(imdb)
    return imdb

if __name__ == '__main__':
  cfg_file = '/mnt/data/tf-faster-rcnn-master/experiments/cfgs/vgg16.yml'
  model = '/mnt/data/tf-faster-rcnn-master/output/vgg16/us_trainval/default/vgg16_faster_rcnn_iter_200.ckpt'
  weight = '/mnt/data/tf-faster-rcnn-master/output/vgg16/us_trainval/default/vgg16_faster_rcnn_iter_200.ckpt'
  imdb_name = 'us_trainval'
  imdbval_name = 'us_trainval'
  comp_mode = False
  net = 'vgg16'
  tag = None
  max_per_image = 1
  set_cfgs = ['ANCHOR_SCALES', '[8,16,32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'TRAIN.STEPSIZE', '[40000]']



  if cfg_file is not None:
    cfg_from_file(cfg_file)
  if set_cfgs is not None:
    cfg_from_list(set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  # if has model, get the name from it
  # if does not, then just use the initialization weights
  if model:
    filename = os.path.splitext(os.path.basename(model))[0]
  else:
    filename = os.path.splitext(os.path.basename(weight))[0]

  tag = tag
  tag = tag if tag else 'default'
  filename = tag + '/' + filename

  imdb = get_imdb(imdb_name)
  imdb.competition_mode(comp_mode)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth=True

  # init session
  sess = tf.Session(config=tfconfig)
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

  # load model
  net.create_architecture("TEST", imdb.num_classes, tag='default',
                          anchor_scales=cfg.ANCHOR_SCALES,
                          anchor_ratios=cfg.ANCHOR_RATIOS)

  print(('Loading model check point from {:s}').format(model))
  saver = tf.train.Saver()
  saver.restore(sess, model)
  print('Loaded.')


  test_net(sess, net, imdb, filename, max_per_image=max_per_image)

  sess.close()
