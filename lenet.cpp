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

#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "common.h"
#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>

// stuff we know about the network and the input/output blobs
static const int INPUT_H = 32;
static const int INPUT_W = 32;
static const int NUMBER_CLASSES = 10;

const char *INPUT_NAME = "image";
const char *OUTPUT_NAME = "label";

using namespace nvinfer1;

static Logger gLogger;

// Load weights from files shared with TensorRT samples.
// TensorRT weight files have a simple space delimited format:
// [type] [size] <data x size in hex>
std::map<std::string, Weights> loadWeights(const std::string file)
{
    std::cout << "Loading weights: " << file << std::endl;
    std::map<std::string, Weights> weightMap;

    // Open weights file
    std::ifstream input(file);
    assert(input.is_open() && "Unable to load weight file.");

    // Read number of weight blobs
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");

    while (count--)
    {
        Weights wt{DataType::kFLOAT, nullptr, 0};
        uint32_t size;

        // Read name and type of blob
        std::string name;
        input >> name >> std::dec >> size;
        wt.type = DataType::kFLOAT;

        // Load blob
        uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
        for (uint32_t x = 0, y = size; x < y; ++x)
        {
            input >> std::hex >> val[x];
        }
        wt.values = val;
        
        wt.count = size;
        weightMap[name] = wt;
    }

    return weightMap;
}

// Custom create LeNet neural network engine
ICudaEngine *createEngine(unsigned int maxBatchSize, IBuilder* builder, DataType datatype) {
    // batch size equal 1
    INetworkDefinition* model = builder->createNetwork();

    // Create input tensor of shape { 1, 1, 32, 32 } with name INPUT_BLOB_NAME
    ITensor *data = model->addInput(INPUT_NAME, datatype, Dims3{1, INPUT_H, INPUT_W});
    assert(data);
 
    std::map<std::string, Weights> weights = loadWeights("/opt/tensorrt_models/torch/lenet/lenet.wts");

    // Add convolution layer with 6 outputs and a 5x5 filter.
    IConvolutionLayer *conv1 = model->addConvolution(*data, 6, DimsHW{5, 5}, weights["conv1.weight"], weights["conv1.bias"]);
    assert(conv1);
    conv1->setStride(DimsHW{1, 1});

    // Add activation layer using the ReLU algorithm.
    IActivationLayer *relu1 = model->addActivation(*conv1->getOutput(0), ActivationType::kRELU);
    assert(relu1);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer *pool1 = model->addPooling(*relu1->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool1);
    pool1->setStride(DimsHW{2, 2});

    // Add convolution layer with 6 outputs and a 5x5 filter.
    IConvolutionLayer *conv2 = model->addConvolution(*pool1->getOutput(0), 16, DimsHW{5, 5}, weights["conv2.weight"], weights["conv2.bias"]);
    assert(conv2);
    conv2->setStride(DimsHW{1, 1});

    // Add activation layer using the ReLU algorithm.
    IActivationLayer *relu2 = model->addActivation(*conv2->getOutput(0), ActivationType::kRELU);
    assert(relu2);

    // Add max pooling layer with stride of 2x2 and kernel size of 2x2.
    IPoolingLayer *pool2 = model->addPooling(*relu2->getOutput(0), PoolingType::kAVERAGE, DimsHW{2, 2});
    assert(pool2);
    pool2->setStride(DimsHW{2, 2});

    // Add fully connected layer with 500 outputs.
    IFullyConnectedLayer *fc1 = model->addFullyConnected(*pool2->getOutput(0), 120, weights["fc1.weight"], weights["fc1.bias"]);
    assert(fc1);

    // Add activation layer using the ReLU algorithm.
    IActivationLayer *relu3 = model->addActivation(*fc1->getOutput(0), ActivationType::kRELU);
    assert(relu3);

    IFullyConnectedLayer *fc2 = model->addFullyConnected(*relu3->getOutput(0), 84, weights["fc2.weight"], weights["fc2.bias"]);
    assert(fc2);
    
     // Add second fully connected layer with 20 outputs.
    IActivationLayer *relu4 = model->addActivation(*fc2->getOutput(0), ActivationType::kRELU);
    assert(relu4);

    // Add fully connected layer with 500 outputs.
    IFullyConnectedLayer *fc3 = model->addFullyConnected(*relu4->getOutput(0), NUMBER_CLASSES, weights["fc3.weight"], weights["fc3.bias"]);
    assert(fc3);

    // Add activation layer using the ReLU algorithm.
    ISoftMaxLayer *prob = model->addSoftMax(*fc3->getOutput(0));
    assert(prob);
    prob->getOutput(0)->setName(OUTPUT_NAME);
    model->markOutput(*prob->getOutput(0));

    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(16_MiB);
    ICudaEngine* engine = builder->buildCudaEngine(*model);

    // Don't need the model any more
    model->destroy();

    // Release host memory
    for (auto& memory : weights)
    {
        free((void*) (memory.second.values));
    }

    return engine;
}

void serializeEngine(unsigned int maxBatchSize, IHostMemory **modelStream) {
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void inference(IExecutionContext& context, float *input, float *output, int batchSize) {
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_NAME);

    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * NUMBER_CLASSES * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueue(batchSize, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * NUMBER_CLASSES * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc < 2 && argv[1] != "--engine" && argv[1] != "--image") {
        std::cerr << "Usage:" << std::endl;
        std::cerr << "\t./lenet --engine   // Generate TensorRT inference model." << std::endl;
        std::cerr << "\t./lenet --image ../examples/0.jpg   // Reasoning on the picture." << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    // Subtract mean from image
    float data[INPUT_H * INPUT_W];

    if (std::string(argv[1]) == "--engine") {
        IHostMemory* modelStream{nullptr};
        serializeEngine(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream engine("/opt/tensorrt_models/torch/lenet/lenet.engine");
        if (!engine)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        engine.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        std::cout << "Model engine has created!" << std::endl;
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "--image") {
        std::ifstream file("/opt/tensorrt_models/torch/lenet/lenet.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();

            cv::Mat image = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                std::cerr << "Open image error!" << std::endl;
                return -2;
            }
            cv::resize(image, image, cv::Size(INPUT_H,INPUT_W));

            // Print ASCII representation of digit image
            std::cout << "\nInput:\n" << std::endl;
            for (int i = 0; i < INPUT_H * INPUT_W; i++) {
                data[i] = image.data[i];
                std::cout << (" .:-=+*#%@"[ image.data[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");
            }
        }
    } else 
        return -1;
    

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference
    float prob[NUMBER_CLASSES];
    for (int i = 0; i < 1000; i++) {
        auto start = std::chrono::system_clock::now();
        inference(*context, data, prob, 1);
        auto end = std::chrono::system_clock::now();
    }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    std::cout << "\nPrediction:\n\n";
    unsigned int category;
    for (unsigned int i = 0; i < NUMBER_CLASSES; i++) {
        if(prob[i] > 0.5) 
            category = i;
    }

    std::cout << "Category: " << "`" <<category <<  "`." << " Probability: "<< prob[category] * 100 << "%." << std::endl;

    return 0;
}