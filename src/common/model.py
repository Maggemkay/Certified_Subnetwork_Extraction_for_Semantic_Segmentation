"""Model loading, hook points (circuit components) and the two hook helpers:
ActivationRecorder (keeps activations for gradients) and ChannelMasker (runs a circuit).

Components are the output channels of the hook points below, concatenated in
HOOK_POINTS order into one vector of NUM_COMPONENTS = 4,352 entries.
"""

import numpy as np
import torch
from torchvision import transforms
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights, DeepLabV3_ResNet101_Weights, deeplabv3_resnet50, deeplabv3_resnet101,
)

# name -> (sub-module path, number of output channels)
HOOK_POINTS = {
    "layer1": ("backbone.layer1", 256),
    "layer2": ("backbone.layer2", 512),
    "layer3": ("backbone.layer3", 1024),
    "layer4": ("backbone.layer4", 2048),
    "aspp":   ("classifier.0", 256),   # ASPP output (after its 1x1 projection)
    "head":   ("classifier.3", 256),   # ReLU after the head's 3x3 conv
}
LAYER_GROUPS = {
    "backbone":      ["layer1", "layer2", "layer3", "layer4"],
    "backbone+head": ["layer1", "layer2", "layer3", "layer4", "aspp", "head"],
    "head":          ["aspp", "head"],
}

# Position of each hook point inside the concatenated component vector.
LAYER_SLICES, NUM_COMPONENTS = {}, 0
for _name, (_, _channels) in HOOK_POINTS.items():
    LAYER_SLICES[_name] = slice(NUM_COMPONENTS, NUM_COMPONENTS + _channels)
    NUM_COMPONENTS += _channels

_MODELS = {
    "deeplabv3_resnet50": (deeplabv3_resnet50, DeepLabV3_ResNet50_Weights.DEFAULT),
    "deeplabv3_resnet101": (deeplabv3_resnet101, DeepLabV3_ResNet101_Weights.DEFAULT),
}

# Native-resolution preprocessing: weights.transforms() would resize to 520 on
# the short side and misalign predictions with the VOC masks.
_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(name, device):
    builder, weights = _MODELS[name]
    model = builder(weights=weights)
    # Only the main output is used. Keep the dropped head's size for the parameter print-out.
    model.removed_aux_params = sum(p.numel() for p in model.aux_classifier.parameters())
    model.aux_classifier = None
    return model.eval().to(device)


def preprocess(image, device):
    """PIL image -> normalised [1, 3, H, W] tensor at native resolution."""
    return _preprocess(image).unsqueeze(0).to(device)


def certifiable_components(group):
    """The certifiable components U of a LAYER_GROUPS entry: its layer names, the positions of U in the
    NUM_COMPONENTS vector, and each layer's slice within U."""
    layers = LAYER_GROUPS[group]
    certifiable_index = np.concatenate([np.arange(LAYER_SLICES[name].start, LAYER_SLICES[name].stop)
                                        for name in layers])
    certifiable_slices, start = {}, 0
    for name in layers:
        certifiable_slices[name] = slice(start, start + HOOK_POINTS[name][1])
        start += HOOK_POINTS[name][1]
    return layers, certifiable_index, certifiable_slices


class _Hooks:
    """Context manager base: registers one forward hook per hook point, removes them on exit."""

    def __init__(self, model, names):
        self.model = model
        self.names = list(names)
        self._handles = []

    def _hook(self, name):
        raise NotImplementedError

    def __enter__(self):
        for name in self.names:
            module = self.model.get_submodule(HOOK_POINTS[name][0])
            self._handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *exc):
        for handle in self._handles:
            handle.remove()
        self._handles = []


class ActivationRecorder(_Hooks):
    """Keeps each hook point's output (still in the autograd graph) in `self.acts`,
    so that torch.autograd.grad can differentiate a target with respect to it."""

    def __init__(self, model, names=HOOK_POINTS):
        super().__init__(model, names)
        self.acts = {}

    def _hook(self, name):
        def hook(module, inputs, output):
            assert output.shape[1] == HOOK_POINTS[name][1], (name, output.shape)
            self.acts[name] = output
        return hook


@torch.inference_mode()
def forward(model, x, masks=None):
    """[21, H, W] logits of the full model (masks=None) or of the circuit given by `masks`."""
    if masks is None:
        return model(x)["out"][0]
    with ChannelMasker(model, masks):
        return model(x)["out"][0]


class ChannelMasker(_Hooks):
    """Runs the model as a circuit: multiplies each hook point's output by a 0/1
    channel mask. `masks` maps layer name -> [channels] tensor; layers that are
    not given keep all channels (i.e. are not part of U)."""

    def __init__(self, model, masks):
        super().__init__(model, masks.keys())
        device = next(model.parameters()).device
        self.masks = {name: torch.as_tensor(m, dtype=torch.float32, device=device).view(1, -1, 1, 1)
                      for name, m in masks.items()}

    def _hook(self, name):
        def hook(module, inputs, output):
            return output * self.masks[name]
        return hook
