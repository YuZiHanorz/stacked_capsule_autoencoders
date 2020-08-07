# coding=utf-8
# Copyright 2019 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""Evaluation script."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import pdb
import sys
import traceback

from absl import flags
from absl import logging
from monty.collections import AttrDict
import sklearn.cluster
import tensorflow as tf

from stacked_capsule_autoencoders.capsules.configs import data_config
from stacked_capsule_autoencoders.capsules.configs import model_config
from stacked_capsule_autoencoders.capsules.eval import cluster_classify
from stacked_capsule_autoencoders.capsules.eval import collect_results
from stacked_capsule_autoencoders.capsules.plot import make_tsne_plot
from stacked_capsule_autoencoders.capsules.train import tools
from stacked_capsule_autoencoders.capsules import capsule as _capsule
from tensorflow.python.client import timeline

flags.DEFINE_string('snapshot', '', 'Checkpoint file.')
flags.DEFINE_string(
    'tsne_figure_name', 'tsne.png', 'Filename for the TSNE '
    'figure. It will be saved in the checkpoint folder.')

# These two flags are necessary for model loading. Don't change them!
flags.DEFINE_string('dataset', 'mnist', 'Don\'t change!')
flags.DEFINE_string('model', 'scae', 'Don\'t change!.')


def main(_=None):
    FLAGS = flags.FLAGS  # pylint: disable=invalid-name,redefined-outer-name
    config = FLAGS
    FLAGS.__dict__['config'] = config
    # Build the graph
    with tf.Graph().as_default():

        dataset = tf.data.Dataset.from_tensors(tf.ones([32, 256]))
        dataset = dataset.repeat().batch(200)
        it = dataset.make_one_shot_iterator()
        data_batch = it.get_next()

        model = _capsule.ModelTest(config.n_obj_caps,
                                   2,
                                   config.n_part_caps,
                                   n_caps_params=config.n_obj_caps_params,
                                   n_hiddens=128,
                                   learn_vote_scale=True,
                                   deformations=True,
                                   noise_type='uniform',
                                   noise_scale=4.,
                                   similarity_transform=False)
        #data_dict = data_config.get(FLAGS)
        #trainset = data_dict.trainset
        #validset = data_dict.validset

        # Optimisation target
        #validset = tools.maybe_convert_dataset(validset)
        #trainset = tools.maybe_convert_dataset(trainset)
        #print(validset['image'])
        #print(trainset['image'])
        tensor = model(data_batch)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        #saver = tf.train.Saver()
        #saver.restore(sess, FLAGS.snapshot)
    n_batches = 20
    print('Collecting: 0/{}'.format(n_batches), end='')

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    for i in range(n_batches):
        print('\rCollecting: {}/{}'.format(i + 1, n_batches), end='')

        if i == 10:
            print('')
            print('herehereherehere it starts')
            sess.run(tensor, options=run_options, run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            print('herehereherehere it ends')
            with open('timeline.json', 'w') as f:
                f.write(ctf)
            print('')
        else:
            sess.run(tensor)


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
