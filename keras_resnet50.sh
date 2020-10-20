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
#nvprof --log-file cnn_dram_read_throughput_cifar10.csv --metrics dram_read_throughput \
#nvprof --log-file cnn_dram_read_transaction_cifar10.csv --metrics dram_read_transactions \
#nvprof --log-file cnn_dram_write_throughput_cifar10.csv --metrics dram_write_throughput \
#nvprof --log-file cnn_dram_write_transaction_cifar10.csv --metrics dram_write_transactions \
nvprof --log-file cnn_dram_write_transaction_cifar10_100.csv --metrics dram_write_transactions \
python -m stacked_capsule_autoencoders.keras_resnet50
