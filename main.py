import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional

import utils
from model_architecture.spiking_vgg_cifar import spiking_vgg16_bn_cifar


class SNNWrapper(nn.Module):
    """
    Wrapper that handles time dimension internally.
    Allows using existing utils.py functions without modification.
    """
    
    def __init__(self, snn_model, T=4):
        super(SNNWrapper, self).__init__()
        self.snn = snn_model
        self.T = T
    
    def forward(self, x):
        # Add time dimension: [N, C, H, W] → [T, N, C, H, W]
        x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        
        # Forward through SNN
        out_seq = self.snn(x_seq)
        
        # Average over time
        out = out_seq.mean(0)
        
        # Reset membrane
        functional.reset_net(self.snn)
        
        return out


def main():
    modelDir = "./checkpoint/spiking_vgg16_bn_cifar.pth"

    # Parameters
    batchSize = 64
    numClasses = 10
    T = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create model
    snn_model = spiking_vgg16_bn_cifar(
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True
    )
    
    functional.set_step_mode(snn_model, 'm')

    # Load trained weights (NO module prefix handling needed!)
    checkpoint = torch.load(modelDir, map_location=device)
    snn_model.load_state_dict(checkpoint['model'])
    
    # Wrap and setup
    model = SNNWrapper(snn_model, T=T)
    model = model.to(device)
    model = model.eval()
    
    print(f"Model loaded from: {modelDir}")
    print(f"Checkpoint accuracy: {checkpoint.get('acc', 'N/A'):.2f}%")
    print(f"Timesteps (T): {T}")

    # Load validation data
    valLoader = utils.GetCIFAR10Validation(batchSize)

    # Evaluate using existing utils functions
    valAcc = utils.validateD(valLoader, model, device)
    print(f"CIFAR-10 Validation Accuracy: {valAcc * 100:.2f}%")

    # Get correctly classified samples
    totalSamplesRequired = 1000
    correctLoader = utils.GetCorrectlyIdentifiedSamplesBalanced(
        model, totalSamplesRequired, valLoader, numClasses
    )

    correctAcc = utils.validateD(correctLoader, model, device)
    print(f"CIFAR-10 Clean Correct Loader Acc: {correctAcc * 100:.2f}%")

    # Summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: Spiking VGG16-BN")
    print(f"Timesteps: {T}")
    print(f"Validation Accuracy: {valAcc * 100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()