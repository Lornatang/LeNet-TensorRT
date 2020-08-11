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

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include "NvInfer.h"
#include "common.h"
#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"

// stuff we know about the network and the input/output blobs
static const int INPUT_C = 1;
static const int INPUT_H = 28;
static const int INPUT_W = 28;
static const int NUMBER_CLASSES = 10;

const char *INPUT_NAME = "image";
const char *OUTPUT_NAME = "label";

using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> load_weights(const std::string file) {
  printMessage(0);
  std::cout << "Loading weights from `" << file << "`." << std::endl;
  std::map<std::string, Weights> weights;

  // Open weights file
  std::ifstream input(file);
  assert(input.is_open() && "Unable to load weight file.");

  // Read number of weight blobs
  int32_t count;
  input >> count;
  assert(count > 0 && "Invalid weight map file.");

  while (count--) {
    Weights wt{DataType::kFLOAT, nullptr, 0};
    uint32_t size;

    // Read name and type of blob
    std::string name;
    input >> name >> std::dec >> size;
    wt.type = DataType::kFLOAT;

    // Load blob
    uint32_t *val = reinterpret_cast<uint32_t *>(malloc(sizeof(val) * size));
    for (uint32_t x = 0, y = size; x < y; ++x) input >> std::hex >> val[x];

    wt.values = val;
    wt.count = size;
    weights[name] = wt;
  }

  return weights;
}

// Custom create LeNet neural network engine
ICudaEngine *create_engine(unsigned int maxBatchSize, IBuilder *builder, DataType datatype, IBuilderConfig *config) {
  INetworkDefinition *model = builder->createNetworkV2(0);

  // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
  ITensor *data = model->addInput(INPUT_NAME, datatype, Dims3{INPUT_C, INPUT_H, INPUT_W});
  assert(data);

  std::map<std::string, Weights> weights = loadWeights("/opt/tensorrt_models/torch/lenet/lenet.wts");

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

  // Add fully connected layer with 500 outputs.
  IFullyConnectedLayer *fc1 =
      model->addFullyConnected(*pool2->getOutput(0), 120, weights["fc1.weight"], weights["fc1.bias"]);
  assert(fc1);

  // Add activation layer using the ReLU algorithm.
  IActivationLayer *relu3 = model->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
  assert(relu3);

  IFullyConnectedLayer *fc2 =
      model->addFullyConnected(*relu3->getOutput(0), 84, weights["fc2.weight"], weights["fc2.bias"]);
  assert(fc2);

  // Add second fully connected layer with 20 outputs.
  IActivationLayer *relu4 = model->addActivation(*fc2->getOutput(0), ActivationType::kRELU);
  assert(relu4);

  // Add fully connected layer with 500 outputs.
  IFullyConnectedLayer *fc3 =
      model->addFullyConnected(*relu4->getOutput(0), NUMBER_CLASSES, weights["fc3.weight"], weights["fc3.bias"]);
  assert(fc3);

  // Add activation layer using the ReLU algorithm.
  ISoftMaxLayer *prob = model->addSoftMax(*fc3->getOutput(0));
  assert(prob);
  prob->getOutput(0)->setName(OUTPUT_NAME);
  model->markOutput(*prob->getOutput(0));

  // Build engine
  builder->setMaxBatchSize(maxBatchSize);
  config->setMaxWorkspaceSize(16_MiB);
  config->setFlag(BuilderFlag::kFP16);
  ICudaEngine *engine = builder->buildEngineWithConfig(*model, *config);

  // Don't need the model any more
  model->destroy();

  // Release host memory
  for (auto &memory : weights) free((void *)(memory.second.values));

  return engine;
}

void serializeEngine(unsigned int maxBatchSize, IHostMemory **modelStream) {
  // Create builder
  IBuilder *builder = createInferBuilder(gLogger);
  IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an
  // engine
  printMessage(0);
  std::cout << "Currently creating an inference engine." << std::endl;
  ICudaEngine *engine = create_engine(maxBatchSize, builder, DataType::kFLOAT, config);
  assert(engine != nullptr);

  // Serialize the engine
  (*modelStream) = engine->serialize();

  // Close everything down
  engine->destroy();
  builder->destroy();
}

void inference(IExecutionContext &context, float *input, float *output, int batchSize) {
  const ICudaEngine &engine = context.getEngine();

  // Pointers to input and output device buffers to pass to engine.
  // Engine requires exactly IEngine::getNbBindings() number of buffers.
  assert(engine.getNbBindings() == 2);
  void *buffers[2];

  // In order to bind the buffers, we need to know the names of the input and
  // output tensors. Note that indices are guaranteed to be less than
  // IEngine::getNbBindings()
  const int inputIndex = engine.getBindingIndex(INPUT_NAME);
  const int outputIndex = engine.getBindingIndex(OUTPUT_NAME);

  // Create GPU buffers on device
  CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float)));
  CHECK(cudaMalloc(&buffers[outputIndex], batchSize * NUMBER_CLASSES * sizeof(float)));

  // Create stream
  cudaStream_t stream;
  CHECK(cudaStreamCreate(&stream));

  // DMA input batch data to device, infer on the batch asynchronously, and DMA
  // output back to host
  CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_C * INPUT_H * INPUT_W * sizeof(float),
                        cudaMemcpyHostToDevice, stream));
  context.enqueue(batchSize, buffers, stream, nullptr);
  CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * NUMBER_CLASSES * sizeof(float),
                        cudaMemcpyDeviceToHost, stream));
  cudaStreamSynchronize(stream);

  // Release stream and buffers
  cudaStreamDestroy(stream);
  CHECK(cudaFree(buffers[inputIndex]));
  CHECK(cudaFree(buffers[outputIndex]));
}

void printHelpInfo() {
  printMessage(2);
  std::cerr << "Invalid arguments!" << std::endl;
  std::cout << "Usage: " << std::endl;
  std::cout << "  ./lenet --engine  // Generate TensorRT inference model." << std::endl;
  std::cout << "  ./lenet --image ../examples/0.jpg  // Reasoning on the picture." << std::endl;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printHelpInfo();
    return -1;
  }

  // create a model using the API directly and serialize it to a stream
  char *trtModelStream{nullptr};
  size_t size{0};

  if (std::string(argv[1]) == "--engine") {
    IHostMemory *modelStream{nullptr};
    serializeEngine(1, &modelStream);
    assert(modelStream != nullptr);

    std::ofstream engine("/opt/tensorrt_models/torch/lenet/lenet.engine");
    if (!engine) {
      printMessage(2);
      std::cerr << "Could not open plan output file" << std::endl;
      printMessage(0);
      std::cout << "Please refer to the documentation how to generate an inference engine." << std::endl;
      return -1;
    }
    engine.write(reinterpret_cast<const char *>(modelStream->data()), modelStream->size());

    printMessage(0);
    std::cout << "The inference engine is saved to `/opt/tensorrt_models/torch/lenet/lenet.engine`!" << std::endl;

    modelStream->destroy();
    return 1;
  } else if (std::string(argv[1]) == "--image") {
    printMessage(0);
    std::cout << "Read from`/opt/tensorrt_models/torch/lenet/lenet.engine` inference engine." << std::endl;
    std::ifstream file("/opt/tensorrt_models/torch/lenet/lenet.engine", std::ios::binary);
    if (file.good()) {
      file.seekg(0, file.end);
      size = file.tellg();
      file.seekg(0, file.beg);
      trtModelStream = new char[size];
      assert(trtModelStream);
      file.read(trtModelStream, size);
      file.close();
    }
  } else
    return -1;

  IRuntime *runtime = createInferRuntime(gLogger);
  assert(runtime != nullptr);
  ICudaEngine *engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
  assert(engine != nullptr);
  IExecutionContext *context = engine->createExecutionContext();
  assert(context != nullptr);

  // Load dataset labels
  std::string label_file = "/opt/tensorrt_models/data/mnist.txt";
  std::vector<std::string> labels_;
  std::ifstream labels(label_file.c_str());
  // CHECK(labels) << "Unable to open labels file " << label_file;
  std::string line;
  while (std::getline(labels, line)) labels_.push_back(std::string(line));

  // Read a digit file
  float data[INPUT_C * INPUT_H * INPUT_W];
  cv::Mat raw_image, image;

  printMessage(0);
  std::cout << "Read image from `" << argv[2] << "`!" << std::endl;

  raw_image = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
  if (raw_image.empty()) {
    printMessage(2);
    std::cerr << "Open image error!" << std::endl;
    return -2;
  }

  printMessage(0);
  std::cout << "Resize image size to 28 * 28." << std::endl;
  cv::resize(raw_image, image, cv::Size(INPUT_H, INPUT_W));

  for (int i = 0; i < INPUT_C * INPUT_H * INPUT_W; i++) data[i] = image.data[i];

  // Run inference
  float prob[NUMBER_CLASSES];

  printMessage(0);
  std::cout << "Inference......" << std::endl;
  for (int i = 0; i < 1000; i++) {
    auto start = std::chrono::system_clock::now();
    inference(*context, data, prob, 1);
    auto end = std::chrono::system_clock::now();
  }

  // Destroy the engine
  context->destroy();
  engine->destroy();
  runtime->destroy();

  // Calculate the probability of the top 5 categories
  std::vector<float> probs;
  std::vector<std::pair<float, int> > pairs;
  std::vector<int> results;

  for (int n = 0; n < NUMBER_CLASSES; n++) probs.push_back(prob[n]);

  // Sort the categories in the array
  for (size_t i = 0; i < probs.size(); ++i) pairs.push_back(std::make_pair(probs[i], i));
  std::partial_sort(pairs.begin(), pairs.begin() + 5, pairs.end(), PairCompare);

  // Formatted output and display
  std::cout << std::left << std::setw(12) << "--------" << std::right << std::setw(12) << "-----------" << std::endl;
  std::cout << std::left << std::setw(12) << "Category" << std::right << std::setw(12) << "probability" << std::endl;
  std::cout << std::left << std::setw(12) << "--------" << std::right << std::setw(12) << "-----------" << std::endl;
  for (int i = 0; i < 5; ++i) {
    results.push_back(pairs[i].second);
    std::cout << std::left << std::setw(12) << labels_[pairs[i].second] << std::right << std::setw(9)
              << prob[pairs[i].second] * 100 << " %." << std::endl;
  }
  std::cout << std::endl;

  return 0;
}