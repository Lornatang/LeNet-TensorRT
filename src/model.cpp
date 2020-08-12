/*
 * Copyright (c) 2020, Lorna Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../include/model.h"

using namespace nvinfer1;
using namespace std;

static Logger gLogger;

static const char *WEIGHTS = "/opt/tensorrt_models/torch/lenet/lenet.wts";

// Custom create LeNet neural network engine
ICudaEngine *create_lenet_engine(unsigned int max_batch_size, IBuilder *builder, DataType data_type,
                                 IBuilderConfig *config) {
  INetworkDefinition *model = builder->createNetworkV2(0);

  // Create input tensor of shape {1, 1, 28, 28} with name INPUT_NAME
  ITensor *data = model->addInput("input", data_type, Dims3{1, 28, 28});
  assert(data);

  std::map<std::string, Weights> weights = load_weights(WEIGHTS);

  // Add convolution layer with 6 outputs and a 5x5 filter.
  IConvolutionLayer *conv1 =
      model->addConvolutionNd(*data, 6, DimsHW{5, 5}, weights["conv1.weight"], weights["conv1.bias"]);
  assert(conv1);
  conv1->setStrideNd(DimsHW{1, 1});
  conv1->setPaddingNd(DimsHW{2, 2});

  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu1 = model->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
  assert(relu1);

  // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
  IPoolingLayer *pool1 = model->addPoolingNd(*relu1->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
  assert(pool1);
  pool1->setStrideNd(DimsHW{2, 2});

  // Add convolution layer with 6 outputs and a 5x5 filter.
  IConvolutionLayer *conv2 =
      model->addConvolutionNd(*pool1->getOutput(0), 16, DimsHW{5, 5}, weights["conv2.weight"], weights["conv2.bias"]);
  assert(conv2);
  conv2->setStrideNd(DimsHW{1, 1});

  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu2 = model->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
  assert(relu2);

  // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
  IPoolingLayer *pool2 = model->addPoolingNd(*relu2->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
  assert(pool2);
  pool2->setStrideNd(DimsHW{2, 2});

  // Add fully connected layer with 400 outputs.
  IFullyConnectedLayer *fc1 =
      model->addFullyConnected(*pool2->getOutput(0), 120, weights["fc1.weight"], weights["fc1.bias"]);
  assert(fc1);

  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu3 = model->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
  assert(relu3);

  // Add fully connected layer with 84 outputs.
  IFullyConnectedLayer *fc2 =
      model->addFullyConnected(*relu3->getOutput(0), 84, weights["fc2.weight"], weights["fc2.bias"]);
  assert(fc2);

  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu4 = model->addActivation(*fc2->getOutput(0), ActivationType::kRELU);
  assert(relu4);

  // Add fully connected layer with 10 outputs.
  IFullyConnectedLayer *fc3 =
      model->addFullyConnected(*relu4->getOutput(0), 10, weights["fc3.weight"], weights["fc3.bias"]);
  assert(fc3);

  // Add activation layer using the ReLU algorithm.
  ISoftMaxLayer *prob = model->addSoftMax(*fc3->getOutput(0));
  assert(prob);
  prob->getOutput(0)->setName("label");
  model->markOutput(*prob->getOutput(0));

  // Build engine
  builder->setMaxBatchSize(max_batch_size);
  config->setMaxWorkspaceSize(16_MiB);
  config->setFlag(BuilderFlag::kFP16);
  ICudaEngine *engine = builder->buildEngineWithConfig(*model, *config);

  // Don't need the model any more
  model->destroy();

  // Release host memory
  for (auto &memory : weights) free((void *)(memory.second.values));

  return engine;
}
