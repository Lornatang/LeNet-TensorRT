# LeNet-TensorRT

## Overview

Use the PyTorch framework to build a network model and train the data set, and hand it over to TensorRT for inference.

### Table of contents

- [LeNet-TensorRT](#lenet-tensorrt)
  - [Overview](#overview)
    - [Table of contents](#table-of-contents)
    - [About TensorRT](#about-tensorrt)
    - [About LeNet-TensorRT](#about-lenet-tensorrt)
    - [Installation](#installation)
      - [Clone and install requirements](#clone-and-install-requirements)
      - [Download tensorrt weights and tensorrt engine](#download-tensorrt-weights-and-tensorrt-engine)
      - [Download MNIST datasets](#download-mnist-datasets)
    - [Usage](#usage)
      - [Train](#train)
      - [Inference](#inference)
    - [Credit](#credit)
      - [Gradient-Based Learning Applied to Document Recognition](#gradient-based-learning-applied-to-document-recognition)

### About TensorRT

TensorRT is a C++ library for high performance inference on NVIDIA GPUs and deep learning accelerators.
More detail see [https://developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt)

### About LeNet-TensorRT

This repo uses the TensorRT API to build an engine for a model trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
It creates the network layer by layer, sets up weights and inputs/outputs, and then performs inference.
Both of these samples use the same model weights, handle the same input, and expect similar output.

### Installation

#### Clone and install requirements

```bash
git clone https://github.com/Lornatang/LeNet-TensorRT.git
cd LeNet-TensorRT/
pip install -r requirements.txt
```

**In addition, the following conditions should also be met for TensorRT:**

- Cmake >= 3.10.2
- OpenCV >= 4.4.0
- TensorRT >= 7.0

#### Download tensorrt weights and tensorrt engine

```bash
cd weights/
bash download.sh
```

#### Download MNIST datasets

```bash
cd data/
bash get_dataset.sh
```

### Usage

#### Train

```bash
python train.py data
```

#### Inference

1. Compile this sample by running `make` in the `<TensorRT root directory>/build` directory. The binary named `lenet` will be created in the `<TensorRT root directory>/build/bin` directory.

    ```bash
    cd <TensorRT root directory>
    mkdir build
    cd build
    cmake ..
    make
    ```

    Where `<TensorRT root directory>` is where you clone LeNet-TensorRT.

2. Run the sample to perform inference on the digit:

    ```bash
    ./lenet --engine  // Generate TensorRT inference model.
    ./lenet --image ../examples/0.jpg  // Inference on the picture.
    ```

3. Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following; ASCII rendering of the input image with digit 0:

    ```text
    # ./lenet --image ../examples/0.jpg 
    [INFO(9:18:9)]: Read from`/opt/tensorrt_models/torch/lenet/lenet.engine` inference engine.
    [INFO(9:18:9)]: Read image from `../examples/0.jpg`!
    [INFO(9:18:9)]: Read image successful! 
    [INFO(9:18:9)]: Adjust image size to 32 * 32.
    [INFO(9:18:9)]: Adjust image size successful.

    [INFO(9:18:9)]: Input:

                                    
                                    
                                    
                                    
                    ::::          
                    -+**+:         
                    :#@@@@*         
                :%@@@@@#:        
                -%@@@#++#%=       
                :+%@@%@%+=#@*.      
                -#@@**%#=-=%*.      
            -*@@#--+=: :##:      
            :*%@#= .:.  .#%=.     
            :+%%*=:      .#@*.     
            +%#=.        .#@*:     
            :%%=.         .#@*:     
            *@*:          .#@*:     
        :%@+:          .#@+.     
        -@#-.         .=%#-      
        -@*:         .=##=.      
        -@*:        .=##-.       
        -@*.      .-+##-         
        -@#-    ..=#%#-          
        -@@*-::=*#%@#=.          
        -@@@%#%@@@%+:            
        .#@@@@@@#*:.             
            :+****+-.               
            .::::.                 
                                    
                                    
                                    
                                    

    [INFO(9:18:10)]: Inference......
    [INFO(9:18:10)]: Result: 
        Category: 0
        Probability: 100%.

    ```
  
4. Install into the system directory(optional)

    ```bash
    sudo make install
    # Create lenet engine
    lenet --engine
    # Test the picture
    lenet --image ~/Desktop/test.jpg
    ```

### Credit

#### Gradient-Based Learning Applied to Document Recognition

_YANN LECUN, MEMBER, IEEE, LEON BOTTOU, YOSHUA BENGIO, AND PATRICK HAFFNER_ </br>

**Abstract** </br>
Multilayer neural networks trained with the back-propagation
algorithm constitute the best example of a successful gradientbased learning technique. Given an appropriate network
architecture, gradient-based learning algorithms can be used
to synthesize a complex decision surface that can classify
high-dimensional patterns, such as handwritten characters, with
minimal preprocessing. This paper reviews various methods
applied to handwritten character recognition and compares them
on a standard handwritten digit recognition task. Convolutional
neural networks, which are specifically designed to deal with
the variability of two dimensional (2-D) shapes, are shown to
outperform all other techniques.
Real-life document recognition systems are composed of multiple
modules including field extraction, segmentation, recognition,
and language modeling. A new learning paradigm, called graph
transformer networks (GTN’s), allows such multimodule systems
to be trained globally using gradient-based methods so as to
minimize an overall performance measure.
Two systems for online handwriting recognition are described.
Experiments demonstrate the advantage of global training, and
the flexibility of graph transformer networks.
A graph transformer network for reading a bank check is
also described. It uses convolutional neural network character
recognizers combined with global training techniques to provide
record accuracy on business and personal checks. It is deployed
commercially and reads several million checks per day.

[[Paper]](https://pdfs.semanticscholar.org/62d7/9ced441a6c78dfd161fb472c5769791192f6.pdf)

```bash
@article{LeNet,
  title={Gradient-Based Learning Applied to Document Recognition},
  author={YANN LECUN, MEMBER, IEEE, LEON BOTTOU, YOSHUA BENGIO, AND PATRICK HAFFNE},
  journal = {IEEE},
  year={1998}
}
```
