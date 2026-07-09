# CHIP: Efficient Homomorphic Encryption-Based CNN Batch Inference Using Channel-Interleaved Packing with Small Rotation Key Set

This repository contains the implementation of the paper **CHIP: Efficient Homomorphic Encryption-Based CNN Batch Inference Using Channel-Interleaved Packing with Small Rotation Key Set**.
The paper was accepted at the IEEE CSF 2026, available [here]().

## Prerequests
1. Git & Git LFS (run ```sudo apt install git-lfs``` to isntall)

## Project Structure
- data: Parameter files for Fast Convolution Algorithm
- examples: Core Imprelementation of CHIP
- model: Pre-trained model weights and CIFAR10/100 samples


## Installation
```
# Clone the project
git clone https://github.com/whcjimmy/CHIP
cd CHIP

# Pull the HEAAN docker
docker pull cryptolabinc/heaan:0.2.0-x86_64-hexl-ubuntu22.04

# Run the docker container
docker run -v ./:/app -it cryptolabinc/heaan:0.2.0-x86_64-hexl-ubuntu22.04 bash

# --- The following commands should be executed inside the container ---

# Install requested libraries
apt update 
apt install -y cmake git libhdf5-dev

# Install HighFive
cd /app
git clone https://github.com/highfive-devs/highfive
cd /app/highfive/deps
git clone https://github.com/catchorg/Catch2 catch2
cd /app/highfive
mkdir build && cd build 
cmake .. && make -j8
make install

# Build the CHIP project
cd /app/examples
mkdir build && cd build 
cmake .. && make -j8
```

## Usage
1. ```ex8_resenet``` is the program for ResNet-20 and ResNet-32
2. ```ex9_vgg11``` is for VGG-11
3. ```ex10_vgg16``` is for VGG-16
4. ```ex13_resnet18``` is for ResNet-18

All programs require one parameter: the starting batch index. For example, to run ResNet-20 starting from batch sample 0, the command is: ```./bin/ex8_resnet 0```.

## Citation
```
@inproceedings{CHIP,
  title={{CHIP: Efficient Homomorphic Encryption-Based CNN Batch Inference Using Channel-Interleaved Packing with Small Rotation Key Set}},
  author={ Wang, Huan-Chih and Wu, Ja-Ling},
  booktitle={IEEE CSF},
  year={2026}
}
```
