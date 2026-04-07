import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional

import utils
from model_architecture.spiking_vgg_voter import spiking_vgg16_bn_voter


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
    modelDir = "./checkpoint/spiking_vgg16_bn_voter.pth"

    # Parameters
    batchSize = 64
    numClasses = 2
    imgH = 40
    imgW = 50
    T = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"GPUs: {torch.cuda.device_count()}")

    # Create model
    print("\n==> Building Spiking VGG16-BN for Voter Dataset...")
    snn_model = spiking_vgg16_bn_voter(
        imgH=imgH,
        imgW=imgW,
        num_classes=numClasses,
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True
    )
    
    functional.set_step_mode(snn_model, 'm')

    # Load trained weights (NO module prefix handling needed!)
    print("\n==> Loading checkpoint...")
    checkpoint = torch.load(modelDir, map_location=device)
    snn_model.load_state_dict(checkpoint['model'])
    
    # Wrap and setup
    model = SNNWrapper(snn_model, T=T)
    model = model.to(device)
    model = model.eval()
    
    print(f"Model loaded from: {modelDir}")
    print(f"Checkpoint accuracy: {checkpoint.get('acc', 'N/A'):.2f}%")
    print(f"Timesteps (T): {T}")
    print(f"Image size: {imgH}×{imgW} (Grayscale)")
    print(f"Num classes: {numClasses}")

    # Load validation data
    print("\n==> Loading Voter validation dataset...")
    valLoader = utils.GetVoterValidation(batchSize)
    print(f"Validation set loaded with batch size: {batchSize}")

    # Evaluate using existing utils functions
    print("\n==> Evaluating on full validation set...")
    valAcc = utils.validateD(valLoader, model, device)
    print(f"Voter Validation Accuracy: {valAcc * 100:.2f}%")

    # Get correctly classified samples
    print("\n==> Collecting correctly identified samples (balanced)...")
    totalSamplesRequired = 1000
    correctLoader = utils.GetCorrectlyIdentifiedSamplesBalanced(
        model, totalSamplesRequired, valLoader, numClasses
    )

    correctAcc = utils.validateD(correctLoader, model, device)
    print(f"Voter Clean Correct Loader Accuracy: {correctAcc * 100:.2f}%")

    # Summary
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Model: Spiking VGG16-BN")
    print(f"Dataset: Voter (Grayscale 40×50)")
    print(f"Number of Classes: {numClasses}")
    print(f"Timesteps: {T}")
    print(f"Validation Accuracy: {valAcc * 100:.2f}%")
    print(f"Clean Samples Accuracy: {correctAcc * 100:.2f}%")
    print("="*50)


if __name__ == "__main__":
    main()