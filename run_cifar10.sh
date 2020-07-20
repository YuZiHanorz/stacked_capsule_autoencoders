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

#!/bin/bash
# Runs SCAE on 40x40 MNIST where part templates and their mixing probabilisties
# are learned separately.
set -e
set -x

export PYTHONPATH=$PYTHONPATH:/work/07521/zhyu1214/maverick2/Capsule
#source stacked_capsule_autoencoders/setup_virtualenv.sh
python -m stacked_capsule_autoencoders.train\
  --name=cifar10\
  --model=scae\
  --dataset=cifar10\
  --max_train_steps=300000\
  --batch_size=128\
  --lr=1e-4\
  --decay_steps=3000\
  --use_lr_schedule=True\
  --canvas_size=32\
  --template_size=14\
  --n_part_caps=32\
  --n_channels=3\
  --n_obj_caps=64\
  --colorize_templates=True\
  --use_alpha_channel=True\
  --prior_within_example_sparsity_weight=0.17\
  --prior_between_example_sparsity_weight=0.1\
  --posterior_within_example_sparsity_weight=1.39\
  --posterior_between_example_sparsity_weight=7.32\
  --color_nonlin='sigmoid'\
  --template_nonlin='sigmoid'\
  --n_heads=2\
  --n_dims=64\
  --n_output_dims=128\
  --prep='sobel'\
  "$@"
