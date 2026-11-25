"""Size measures of a circuit under the binary-mask convention.

Binary-mask convention: a circuit is a 0/1 mask over the model's weights, 1 = "this
is used", 0 = "this is removed". A zero weight counts as removed; nothing is deleted or pruned for size on disk.

A circuit zeroes channels of the hook points in U (the certifiable components, LAYERS). Every weight that can then be
removed without changing the output is 0 in the weight mask. For a zeroed channel j of a hook point:
    producers (write into channel j): filter j of the last conv before the hook point and entry j (weight and bias) of
        the BatchNorm after it, because nothing else reads them:
            layer1-layer4  the stage's LAST block: conv3 filter j and bn3 entry j. Filter j of the EARLIER blocks (and
                           block 0's downsample) stays 1: inside a stage the residual stream is only zeroed at the stage
                           output, so those filters still feed the later blocks of the stage.
            aspp           the ASPP projection conv's filter j and its BatchNorm entry j
            head           the head's 3x3 conv filter j and its BatchNorm entry j
    consumers (read channel j): input slice j of every conv that reads the hook point's output. They multiply zero,
        so removing them changes nothing:
            layer1-layer3  the next stage's first block: conv1 and the downsample conv
            layer4         every ASPP branch conv (the 1x1, the three dilated 3x3, the pooling branch's 1x1)
            aspp           the head's 3x3 conv
            head           the final 1x1 classifier conv (its bias stays 1)
No cascade: if a whole stage is zeroed, the next block's BatchNorm still adds its bias, so its output is a constant, not
zero, and its weights stay 1. Every other weight stays 1 (stem, block interiors, ASPP projection inputs, ...).

Two denominators:
    whole model        every parameter of the model as used here (auxiliary classifier removed): for comparisons with
                       other papers, e.g. Gullstrand et al.'s "percentage of the full model".
    reachable weights  every weight tied to a certifiable channel: its writers (producers) and its readers (consumers),
                       incl. the ASPP input slices. These are the weights a circuit CAN set to 0 (all of U zeroed), so
                       the share runs from 0 to 100 % of what a circuit could remove. They are NOT the backbone: most of
                       them are the ASPP convs reading layer4.
A weight tensor that is both a producer and a consumer (the head's 3x3 conv with LAYERS = "backbone+head") is counted
once per weight: weight [o, i, ...] is 0 if filter o or input slice i is removed.
"""

import numpy as np
import torch

from common import model as M

REACHABLE_WEIGHTS_LABEL = ("reachable weights = weights tied to a certifiable channel (the hook-point outputs of U): "
                           "its writers in the model and its readers, incl. the ASPP input slices")


class WeightMaskCounter:
    """Counts the weights a circuit keeps under the binary-mask convention (module docstring), for the hook points
    `layers` (the layers of U) of `model`."""

    def __init__(self, model, layers):
        self.layers = list(layers)
        parameters = dict(model.named_parameters())
        self.whole_model_weight_count = sum(p.numel() for p in parameters.values())
        # parameter name -> [producer hook point or None (dim 0 = its channels), consumer hook point or None (dim 1)]
        self.rules = {}
        for layer in self.layers:
            producers, consumers = _tied_parameters(model, layer)
            for name in producers:
                self.rules.setdefault(name, [None, None])[0] = layer
            for name in consumers:
                self.rules.setdefault(name, [None, None])[1] = layer
        self.shapes = {name: tuple(parameters[name].shape) for name in self.rules}
        for name, (producer, consumer) in self.rules.items():  # the channel dims must match the hook points
            shape = self.shapes[name]
            assert producer is None or shape[0] == M.HOOK_POINTS[producer][1], (name, shape, producer)
            assert consumer is None or shape[1] == M.HOOK_POINTS[consumer][1], (name, shape, consumer)
        empty = {layer: np.zeros(M.HOOK_POINTS[layer][1], dtype=bool) for layer in self.layers}
        self.reachable_weight_count = self.removed_weight_count(empty)

    def removed_weight_count(self, masks):
        """Number of weights that are 0 in the weight mask of the circuit `masks` ({layer of U: 0/1 channel mask})."""
        return sum(_removed_in_tensor(self.shapes[name], producer, consumer, masks)
                   for name, (producer, consumer) in self.rules.items())

    def size_measures(self, masks):
        """The size measures of the circuit `masks`, by their summary.csv names:
            subnet_size_layer_normalized_ratio_<layer>  share of the layer's channels kept, per layer of U
            subnet_size_group_channel_count             channels of U kept, all layers pooled (absolute)
            subnet_size_group_pooled_ratio              that count over |U| (relative; big layers dominate)
            subnet_size_kept_weight_count               weights that are 1 in the weight mask
            subnet_size_whole_model_weight_ratio        that count over the whole model's weights
            subnet_size_reachable_weight_ratio          the reachable weights kept, over all reachable weights
        (subnet_size_layer_normalized_ratio_mean, the equal-weight mean of the per-layer values, comes from s04.)"""
        removed = self.removed_weight_count(masks)
        out = {f"subnet_size_layer_normalized_ratio_{layer}": float(np.count_nonzero(masks[layer]) / len(masks[layer]))
               for layer in self.layers}
        group_channel_count = sum(int(np.count_nonzero(masks[layer])) for layer in self.layers)
        out["subnet_size_group_channel_count"] = group_channel_count
        out["subnet_size_group_pooled_ratio"] = group_channel_count / sum(len(masks[layer]) for layer in self.layers)
        out["subnet_size_kept_weight_count"] = self.whole_model_weight_count - removed
        out["subnet_size_whole_model_weight_ratio"] = (self.whole_model_weight_count - removed) / self.whole_model_weight_count
        out["subnet_size_reachable_weight_ratio"] = (self.reachable_weight_count - removed) / self.reachable_weight_count
        return out

    def reachable_weight_breakdown(self):
        """Where the reachable weights are, for the log: [(description, count)], summing to reachable_weight_count."""
        empty = {layer: np.zeros(M.HOOK_POINTS[layer][1], dtype=bool) for layer in self.layers}
        groups = {}
        for name, (producer, consumer) in self.rules.items():
            place = "in the backbone" if name.startswith("backbone.") else (
                "in the ASPP branch convs (input slices reading layer4)" if name.startswith("classifier.0.convs.")
                else "in the ASPP projection and the head")
            role = ("writers and readers" if producer and consumer else "writers (producers)" if producer
                    else "readers (consumers)")
            key = f"{role} {place}"
            groups[key] = groups.get(key, 0) + _removed_in_tensor(self.shapes[name], producer, consumer, empty)
        return list(groups.items())


def _removed_in_tensor(shape, producer, consumer, masks):
    """Weights of one parameter tensor that are 0: weight [o, i, ...] is 0 if filter o is a removed producer channel
    (dim 0, hook point `producer`) or input slice i a removed consumer channel (dim 1, hook point `consumer`)."""
    kept = int(np.count_nonzero(masks[producer])) if producer else shape[0]
    if len(shape) > 1:
        kept *= (int(np.count_nonzero(masks[consumer])) if consumer else shape[1]) * int(np.prod(shape[2:]))
    return int(np.prod(shape)) - kept


def _tied_parameters(model, layer):
    """Parameter names of the producers and the consumers of hook point `layer` (module docstring)."""
    if layer.startswith("layer"):
        stage = int(layer[len("layer"):])
        last = f"backbone.{layer}.{len(model.get_submodule(f'backbone.{layer}')) - 1}"
        producers = [f"{last}.conv3.weight", f"{last}.bn3.weight", f"{last}.bn3.bias"]
        if stage < 4:
            first = f"backbone.layer{stage + 1}.0"
            consumers = [f"{first}.conv1.weight", f"{first}.downsample.0.weight"]
        else:
            consumers = [f"classifier.0.convs.{name}.weight"
                         for name, module in model.get_submodule("classifier.0.convs").named_modules()
                         if isinstance(module, torch.nn.Conv2d)]
        return producers, consumers
    if layer == "aspp":
        return (["classifier.0.project.0.weight", "classifier.0.project.1.weight", "classifier.0.project.1.bias"],
                ["classifier.1.weight"])
    if layer == "head":
        return ["classifier.1.weight", "classifier.2.weight", "classifier.2.bias"], ["classifier.4.weight"]
    raise ValueError(f"unknown hook point {layer!r}")
