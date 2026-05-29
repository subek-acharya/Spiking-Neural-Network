import torch
import torch.nn as nn
from spikingjelly.activation_based import surrogate, neuron, functional

import utils
from model_architecture.spiking_vgg_voter import spiking_vgg16_bn_voter
from model_architecture.spiking_resnet_voter import spiking_resnet20_voter


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


def create_spiking_vgg(imgH, imgW, num_classes, device):
    model = spiking_vgg16_bn_voter(
        imgH=imgH, imgW=imgW, num_classes=num_classes,
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True
    )
    functional.set_step_mode(model, 'm')
    return model


def create_spiking_resnet(imgH, imgW, num_classes, device):
    model = spiking_resnet20_voter(
        imgH=imgH, imgW=imgW, num_classes=num_classes,
        spiking_neuron=neuron.IFNode,
        surrogate_function=surrogate.ATan(),
        detach_reset=True
    )
    functional.set_step_mode(model, 'm')
    return model


def load_model(create_fn, checkpoint_path, imgH, imgW, num_classes, T, device):
    snn_model = create_fn(imgH, imgW, num_classes, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    snn_model.load_state_dict(checkpoint['model'])
    model = SNNWrapper(snn_model, T=T)
    model = model.to(device)
    model.eval()
    return model


def evaluate_model(model, batch_size, device):
    loaders = {
        "Val_OnlyBubbles": utils.GetVoterValidation(batch_size),
        "Val_Combined": utils.GetVoterValidationCombined(batch_size),
        "Train_OnlyBubbles": utils.GetVoterTraining(batch_size),
        "Train_Combined": utils.GetVoterTrainingCombined(batch_size),
    }
    
    results = {}
    for name, loader in loaders.items():
        results[name] = utils.validateD(loader, model, device)
    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parameters
    batch_size = 64
    num_classes = 2
    imgH, imgW = 40, 50
    T = 4
    
    # Model checkpoints
    vgg_checkpoint = "./checkpoint/spiking_vgg16_bn_voter.pth"
    resnet_checkpoint = "./checkpoint/spiking_resnet20_voter.pth"
    
    # Evaluate Spiking VGG
    print("Spiking VGG16-BN:")
    vgg_model = load_model(create_spiking_vgg, vgg_checkpoint, imgH, imgW, num_classes, T, device)
    vgg_results = evaluate_model(vgg_model, batch_size, device)
    for name, acc in vgg_results.items():
        print(f"  {name}: {acc:.4f}")
    
    # Evaluate Spiking ResNet
    print("\nSpiking ResNet20:")
    resnet_model = load_model(create_spiking_resnet, resnet_checkpoint, imgH, imgW, num_classes, T, device)
    resnet_results = evaluate_model(resnet_model, batch_size, device)
    for name, acc in resnet_results.items():
        print(f"  {name}: {acc:.4f}")


if __name__ == "__main__":
    main()