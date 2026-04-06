# Spiking Neural Network (SNN) Training on CIFAR-10

Implementation of Spiking VGG-16 for CIFAR-10 classification using the SpikingJelly framework, a PyTorch-based deep learning framework for Spiking Neural Networks.

## Overview

This project implements a Spiking Neural Network (SNN) based on the VGG-16 architecture, adapted for CIFAR-10 (32×32 images). Unlike traditional Artificial Neural Networks (ANNs) that use continuous activation values, SNNs communicate through discrete spikes over multiple timesteps, mimicking biological neural computation.

### What is a Spiking Neural Network?

| Aspect | Traditional ANN | Spiking Neural Network (SNN) |
|--------|-----------------|------------------------------|
| **Data Flow** | Input → ReLU → Output | Input → IFNode → Spikes |
| **Processing** | Single forward pass | Repeated over T timesteps |
| **Activation Values** | Continuous (0.0 to 1.0) | Binary spikes (0 or 1) |
| **Activation Function** | ReLU | Integrate-and-Fire neurons |


### SpikingJelly Framework

[SpikingJelly](https://github.com/fangwei123456/spikingjelly) is a PyTorch-based framework for deep learning with Spiking Neural Networks. It provides:

- **Spiking Neuron Models**: IF (Integrate-and-Fire), LIF (Leaky Integrate-and-Fire), etc.
- **Surrogate Gradient Functions**: Enable backpropagation through non-differentiable spike functions
- **Step Modes**: Single-step (`'s'`) and multi-step (`'m'`) processing
- **Pre-built Architectures**: Spiking versions of VGG, ResNet, etc.
- **ANN-to-SNN Conversion**: Convert trained ANNs to SNNs ([ann2snn](https://github.com/fangwei123456/spikingjelly/tree/master/spikingjelly/activation_based/ann2snn))

This project trains the SNN **from scratch** using surrogate gradient learning.

## Model Architecture

### Spiking VGG-16-BN for CIFAR-10

The architecture is adapted from the [SpikingJelly ImageNet VGG](https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/model/spiking_vgg.py) with modifications for CIFAR-10:

| Component | SNN ImageNet VGG | SNN CIFAR-10 VGG (Ours) |
|-----------|--------------|---------------------|
| Input Size | 224×224 | 32×32 |
| AdaptiveAvgPool | (7, 7) | (1, 1) |
| Classifier | 25088→4096→4096→1000 | 512→10 |
| Activation | IFNode | IFNode |


## Usage

### Training Model

```bash
python train_snn.py
```

### Training Paramters
```bash
Timesteps (T): 4
Epochs: 200
Batch Size: 64
Learning Rate: 0.01 (with Cosine Annealing)
Optimizer: SGD with momentum (0.9)
Weight Decay: 5e-4
Spiking Neuron: IFNode
Surrogate Function: ATan
```
### Evaluating Models

Evaluate a trained model on clean CIFAR-10 test data:

```bash
python main.py
```

The evaluation script:

Loads the trained model
Wraps it with SNNWrapper for compatibility with standard evaluation functions
Evaluates on CIFAR-10 test set
Extracts correctly classified samples (balanced per class)

## Project Structure
```bash
snn-cifar10/
│
├── model_architecture/              # Model architecture implementations
│   ├── __init__.py                  # Package initialization
│   ├── spiking_vgg_cifar.py         # Spiking VGG for CIFAR-10
│   └── VGG_cifar.py                 # Original CNN VGG (reference)
│
├── checkpoint/                      # Saved model checkpoints
│   └── spiking_vgg16_bn_cifar.pth   # Trained SNN model
│
├── data/                            # Datasets (auto-downloaded)
│   └── cifar-10-batches-py/         # CIFAR-10 dataset
│
├── train_snn.py                     # Training script
├── main.py                          # Evaluation script
├── utils.py                         # Utility functions and data loaders
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Results

### Clean CIFAR-10 Test Performance

| Model | Test Accuracy | Timesteps (T) | Training Time | Parameters |
|-------|---------------|---------------|---------------|------------|
| Spiking VGG16-BN | 89.59% | 4 | 162.55 min | ~15M |

## Notes

- **Timesteps (T):** Must be the same during training and testing. The model learns temporal dynamics specific to T=4.

- **Membrane Reset:** Always call `functional.reset_net(model)` after processing each batch.

- **Step Mode:** Use `functional.set_step_mode(model, 'm')` for multi-step processing with `[T, N, C, H, W]` input.t.

## Reference

- [SpikingJelly Documentation](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/basic_concept.html)
- [Train Large Scale SNN Tutorial](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/train_large_scale_snn.html)
