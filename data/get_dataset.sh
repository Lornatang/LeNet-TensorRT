#!/bin/bash

# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

echo "Downloading MNIST dataset ..."
mkdir -p MNIST/processed
mkdir -p MNIST/raw
wget -P MNIST/processed/ https://github.com/Lornatang/LeNet-TensorRT/releases/download/v0.1/test.pt
wget -P MNIST/processed/ https://github.com/Lornatang/LeNet-TensorRT/releases/download/v0.1/training.pt
wget -P MNIST/raw/ http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -P MNIST/raw/ http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
wget -P MNIST/raw/ http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -P MNIST/raw/ http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
echo "Done downloading."
