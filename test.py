from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import pdb
import sys
sys.path.append('E:\\UT-Austin\\Capsnet')

import traceback
from absl import flags
from absl import logging
from monty.collections import AttrDict
import sklearn.cluster
import tensorflow as tf

from stacked_capsule_autoencoders.capsules.configs import data_config
from stacked_capsule_autoencoders.capsules.configs import model_config

flags.DEFINE_string('dataset', 'mnist', 'lalala')


def main(_=None):
    FLAGS = flags.FLAGS  # pylint: disable=invalid-name,redefined-outer-name
    config = FLAGS
    FLAGS.__dict__['config'] = config
    dataset = data_config.get(FLAGS)
    print("here is main")
    print(dataset)


if __name__ == '__main__':
    try:
        logging.set_verbosity(logging.INFO)
        tf.app.run()
    except Exception as err:  # pylint: disable=broad-except
        FLAGS = flags.FLAGS

        last_traceback = sys.exc_info()[2]
        traceback.print_tb(last_traceback)
        print(err)
        pdb.post_mortem(last_traceback)
