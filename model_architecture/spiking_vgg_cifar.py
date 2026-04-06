"""
Spiking VGG for CIFAR-10
Modified from SpikingJelly's spiking_vgg.py minimal changes
Same architecture style, adapted for 32×32 images
https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/activation_based/model/spiking_vgg.py
"""

import torch
import torch.nn as nn
from copy import deepcopy
from spikingjelly.activation_based import functional, neuron, layer


__all__ = [
    "SpikingVGGCIFAR",
    "spiking_vgg11_cifar",
    "spiking_vgg11_bn_cifar",
    "spiking_vgg13_cifar",
    "spiking_vgg13_bn_cifar",
    "spiking_vgg16_cifar",
    "spiking_vgg16_bn_cifar",
    "spiking_vgg19_cifar",
    "spiking_vgg19_bn_cifar",
]


class SpikingVGGCIFAR(nn.Module):
    """
    Spiking VGG for CIFAR-10 (32×32 images)
    
    Changes from ImageNet version:
    1. AdaptiveAvgPool2d((7,7)) → AdaptiveAvgPool2d((1,1))
    2. Large classifier → Simple Linear(512, num_classes)
    3. Default num_classes=10 instead of 1000
    """
    
    def __init__(
        self,
        cfg,
        batch_norm=False,
        norm_layer=None,
        num_classes=10,  # ← Changed from 1000
        init_weights=True,
        spiking_neuron: callable = None,
        **kwargs,
    ):
        super(SpikingVGGCIFAR, self).__init__()
        
        self.features = self.make_layers(
            cfg=cfg,
            batch_norm=batch_norm,
            norm_layer=norm_layer,
            neuron=spiking_neuron,
            **kwargs,
        )
        
        # ← Changed: (7,7) → (1,1) for CIFAR-10
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        
        # ← Changed: Simple classifier instead of 3-layer classifier
        # For CIFAR-10: 32→16→8→4→2→1 after 5 MaxPools, so 512×1×1 = 512 features
        self.classifier = layer.Linear(512, num_classes)
        
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        
        if self.avgpool.step_mode == "s":
            x = torch.flatten(x, 1)
        elif self.avgpool.step_mode == "m":
            x = torch.flatten(x, 2)
        
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    @staticmethod
    def make_layers(
        cfg, batch_norm=False, norm_layer=None, neuron: callable = None, **kwargs
    ):
        if norm_layer is None:
            norm_layer = layer.BatchNorm2d
        layers = []
        in_channels = 3
        for v in cfg:
            if v == "M":
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = layer.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, norm_layer(v), neuron(**deepcopy(kwargs))]
                else:
                    layers += [conv2d, neuron(**deepcopy(kwargs))]
                in_channels = v
        return nn.Sequential(*layers)


# Same configs as ImageNet version
cfgs = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M",
    ],
    "E": [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, 256, "M",
        512, 512, 512, 512, "M",
        512, 512, 512, 512, "M",
    ],
}


# ------- Factory Functions --------------

def _spiking_vgg_cifar(
    cfg,
    batch_norm,
    spiking_neuron: callable = None,
    **kwargs,
):
    model = SpikingVGGCIFAR(
        cfg=cfgs[cfg],
        batch_norm=batch_norm,
        spiking_neuron=spiking_neuron,
        **kwargs,
    )
    return model


def spiking_vgg11_cifar(spiking_neuron: callable = None, **kwargs):
    return _spiking_vgg_cifar("A", False, spiking_neuron, **kwargs)


def spiking_vgg11_bn_cifar(spiking_neuron: callable = None, **kwargs):
    return _spiking_vgg_cifar("A", True, spiking_neuron, **kwargs)


def spiking_vgg13_cifar(spiking_neuron: callable = None, **kwargs):
    return _spiking_vgg_cifar("B", False, spiking_neuron, **kwargs)


def spiking_vgg13_bn_cifar(spiking_neuron: callable = None, **kwargs):
    return _spiking_vgg_cifar("B", True, spiking_neuron, **kwargs)


def spiking_vgg16_cifar(spiking_neuron: callable = None, **kwargs):
    return _spiking_vgg_cifar("D", False, spiking_neuron, **kwargs)


def spiking_vgg16_bn_cifar(spiking_neuron: callable = None, **kwargs):
    return _spiking_vgg_cifar("D", True, spiking_neuron, **kwargs)


def spiking_vgg19_cifar(spiking_neuron: callable = None, **kwargs):
    return _spiking_vgg_cifar("E", False, spiking_neuron, **kwargs)


def spiking_vgg19_bn_cifar(spiking_neuron: callable = None, **kwargs):
    return _spiking_vgg_cifar("E", True, spiking_neuron, **kwargs)
