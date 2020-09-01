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
set -e
set -x

export PYTHONPATH=$PYTHONPATH:/work/07521/zhyu1214/maverick2/Capsule
#source stacked_capsule_autoencoders/setup_virtualenv.sh
python -m stacked_capsule_autoencoders.test2\
  --dataset=svhn\
  --snapshot=stacked_capsule_autoencoders/checkpoints/svhn/model.ckpt-300001\
  --batch_size=100\
  --canvas_size=32\
  --template_size=14\
  --n_part_caps=24\
  --n_channels=3\
  --n_obj_caps=32\
  --colorize_templates=True\
  --use_alpha_channel=True\
  --prior_within_example_sparsity_weight=0.22\
  --prior_between_example_sparsity_weight=0.1\
  --posterior_within_example_sparsity_weight=8.62\
  --posterior_between_example_sparsity_weight=0.26\
  --n_heads=2\
  --n_dims=64\
  --n_output_dims=128\
  --prep='sobel'\
  --part_ll_weight=2.56\
  "$@"
