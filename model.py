import copy
import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.init as init
from torch.optim import Adam, RMSprop, SGD

import swe_ops


def swe_vgg16_bn_cifar(in_channels: int = 3, num_classes: int = 100, hidden: int = 100):
    net = nn.ModuleList()

    feature_extractor = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(32),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )
    net.append(feature_extractor)
    net.append(nn.Flatten())

    hidden_dim = int(hidden / 3)
    net.append(
        swe_ops.SWEFullyConnected(
            in_features=64 * 8 * 8,
            out_features=hidden_dim,
            can_expand=True,
            use_bias=True,
            activation="relu",
            use_batch_norm=False,
        )
    )
    net.append(
        swe_ops.SWEFullyConnected(
            in_features=hidden_dim,
            out_features=hidden_dim,
            can_expand=True,
            use_bias=True,
            activation="relu",
            use_batch_norm=False,
        )
    )
    net.append(
        swe_ops.SWEFullyConnected(
            in_features=hidden_dim,
            out_features=hidden_dim,
            can_expand=True,
            use_bias=True,
            activation="relu",
            use_batch_norm=False,
        )
    )
    net.append(
        swe_ops.SWEFullyConnected(
            in_features=hidden_dim,
            out_features=num_classes,
            can_expand=False,
            use_bias=False,
            activation="none",
            use_batch_norm=False,
        )
    )

    next_layers = {0: [1], 1: [2], 2: [3], 3: [4], 4: [5]}
    layers_to_expand = [2, 3, 4]
    layer_expansion_groups: Dict[int, List[int]] = {}

    return net, next_layers, layers_to_expand, layer_expansion_groups


class Regressor(swe_ops.SWENetwork):
    def __init__(self, config, inp: int, hid: int, outp: int):
        super().__init__()
        self.config = config
        self.verbose = config.verbose
        self.device = config.device
        self.grow_ratio = config.grow_ratio

        self.inp = inp
        self.hid = hid
        self.outp = outp

        self.net, self.next_layers, self.layers_to_expand, self.layer_expansion_groups = (
            swe_vgg16_bn_cifar(hidden=self.hid, num_classes=self.outp)
        )
        if self.verbose:
            print("[INFO] network architecture:", self.next_layers)

        self.lr = 0.001
        self.create_optimizer()
        self.criterion = nn.CrossEntropyLoss()

    def clone_self(self, number_new_neurons: int = 0):
        clone = copy.deepcopy(self)
        clone.opt = None
        return clone

    def create_optimizer(self):
        params = [param for param in self.parameters() if param.requires_grad]

        if self.config.optimizer == "Adam":
            self.opt = Adam(params, lr=self.lr, betas=(self.config.beta1, self.config.beta2), weight_decay=self.config.weight_decay)
        elif self.config.optimizer == "SGD":
            self.opt = SGD(
                params,
                lr=self.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unsupported optimizer {self.config.optimizer}")

    def set_lr(self, lr: float):
        self.lr = lr
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def decay_lr(self, factor: float):
        for param_group in self.opt.param_groups:
            param_group["lr"] *= factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.net:
            x = layer(x)
        return x

    def expand_forward(self, x: torch.Tensor, neurons_per_layer: Dict[int, int], train_only: int = 0) -> torch.Tensor:
        for idx, layer in enumerate(self.net):
            if isinstance(layer, swe_ops.SWEModule) and layer.can_expand and idx == train_only:
                x = layer.swe_forward(x, neurons_per_layer[idx])
            else:
                x = layer(x)
        return x

    def distributor_forward(self, x: torch.Tensor, y: torch.Tensor, total_neurons: int = 0) -> torch.Tensor:
        for layer in self.net:
            if isinstance(layer, swe_ops.SWEModule) and layer.can_expand:
                x = layer.distributor_forward(x, y, total_neurons)
            else:
                x = layer(x)
        return x

    def loss_fn(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        scores = self.forward(x)
        return self.criterion(scores, y)

    def expand_loss_fn(self, x: torch.Tensor, y: torch.Tensor, neurons_per_layer: Dict[int, int], train_only: int = 0) -> torch.Tensor:
        scores = self.expand_forward(x, neurons_per_layer, train_only=train_only)
        return self.criterion(scores, y)

    def distributor_loss_fn(self, x: torch.Tensor, y: torch.Tensor, total_neurons: int = 0) -> torch.Tensor:
        scores = self.distributor_forward(x, y, total_neurons)
        return self.criterion(scores, y)

    def update(self, x: torch.Tensor, y: torch.Tensor) -> float:
        loss = self.loss_fn(x, y)
        if torch.isnan(loss):
            return float("nan")
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach().cpu().item())

    def generate_random_hyperplanes(self, number_new_neurons: int, input_dim: int, device: torch.device):
        hyperplanes = []

        for _ in range(number_new_neurons):
            w = torch.empty(input_dim, device=device)
            init.kaiming_uniform_(w.unsqueeze(0), a=math.sqrt(5))
            bound = 1 / math.sqrt(input_dim)
            b = torch.empty(1, device=device).uniform_(-bound, bound)
            hyperplanes.append(torch.cat([w, b], dim=0))

        return hyperplanes

    def swe(self, loader, n_batches, number_new_neurons):
        for idx, layer in enumerate(self.net):
            if isinstance(layer, swe_ops.SWEModule):
                layer.swe_reset(number_new_neurons)

        clone = self.clone_self(0)
        clone.to(self.device)
        clone.train()

        for idx in reversed(range(len(clone.net))):
            layer = clone.net[idx]
            if isinstance(layer, swe_ops.SWEModule):
                enlarge_in = idx > 2
                enlarge_out = idx < len(clone.net) - 1
                layer.swe_reset(1, simple=True, total_neurons=2 * number_new_neurons)
                layer.distributor_add_new(enlarge_in=enlarge_in, enlarge_out=enlarge_out, total_neurons=2 * number_new_neurons)

        opt_w = RMSprop(clone.net.parameters(), lr=1e-3, momentum=0.1, alpha=0.9)

        for inputs, targets in loader:
            inputs = inputs.to(clone.device)
            targets = targets.to(clone.device)
            loss = clone.loss_fn(inputs, targets)
            opt_w.zero_grad()
            loss.backward()
            opt_w.step()

        for inputs, targets in loader:
            inputs = inputs.to(clone.device)
            targets = targets.to(clone.device)
            loss = clone.distributor_loss_fn(inputs, targets, total_neurons=2 * number_new_neurons)
            opt_w.zero_grad()
            loss.backward()
            for layer_idx in clone.layers_to_expand:
                clone.net[layer_idx].distributor_update_w(len(loader))

        all_w_values = []
        for layer_idx in clone.layers_to_expand:
            layer_w = clone.net[layer_idx].w
            if isinstance(layer_w, torch.Tensor):
                all_w_values.extend((value.item(), layer_idx) for value in layer_w.flatten())

        sorted_w_values = sorted(all_w_values, key=lambda item: item[0])
        layer_scores: Dict[int, int] = {}
        for _, layer_idx in sorted_w_values[:number_new_neurons]:
            layer_scores[layer_idx] = layer_scores.get(layer_idx, 0) + 1

        total_score = max(sum(layer_scores.values()), 1)
        neurons_per_layer = {idx: 0 for idx in range(len(clone.net))}
        for layer_idx, score in layer_scores.items():
            neurons_per_layer[layer_idx] = int(round(score / total_score * number_new_neurons))

        assigned = sum(neurons_per_layer.values())
        diff = number_new_neurons - assigned
        if diff != 0 and layer_scores:
            ordered_layers = sorted(layer_scores.items(), key=lambda item: -item[1])
            for i in range(abs(diff)):
                target_layer = ordered_layers[i % len(ordered_layers)][0]
                neurons_per_layer[target_layer] += 1 if diff > 0 else -1

        for idx, layer in enumerate(self.net):
            if isinstance(layer, swe_ops.SWEModule):
                layer.swe_reset(neurons_per_layer, simple=False, layer_num=idx)

        for idx, layer in enumerate(self.net):
            if isinstance(layer, swe_ops.SWEModule) and layer.can_expand:
                layer_neurons = neurons_per_layer[idx]
                input_dim = layer.module.in_features
                hyperplanes = self.generate_random_hyperplanes(layer_neurons, input_dim, self.device)
                if hyperplanes:
                    tensor_planes = torch.stack(hyperplanes)
                    layer.new_weights = [tensor_planes[i] for i in range(len(hyperplanes))]

        for layer_idx in reversed(range(len(self.net))):
            if isinstance(self.net[layer_idx], swe_ops.SWEModule) and self.net[layer_idx].can_expand:
                layer_neurons = neurons_per_layer[layer_idx]
                if layer_neurons == 0:
                    if hasattr(self.net[layer_idx], "noise"):
                        delattr(self.net[layer_idx], "noise")
                    if hasattr(self.net[layer_idx], "noise_bias"):
                        delattr(self.net[layer_idx], "noise_bias")
                    continue

                self.net[layer_idx].swe_active_expand(layer_neurons)
                self.net[layer_idx + 1].swe_passive_expand(layer_neurons)

                trainable_params = []
                for param in self.net[layer_idx].parameters():
                    if param.requires_grad:
                        trainable_params.append(param)
                for param in self.net[layer_idx + 1].parameters():
                    if param.requires_grad:
                        trainable_params.append(param)

                opt_local = Adam(trainable_params, lr=1e-3, betas=(self.config.beta1, self.config.beta2), weight_decay=1e-4)

                for inputs, targets in loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    loss = self.expand_loss_fn(inputs, targets, neurons_per_layer, train_only=layer_idx)
                    opt_local.zero_grad()
                    loss.backward()

                    with torch.no_grad():
                        weights = self.net[layer_idx].module.weight
                        bias = self.net[layer_idx].module.bias
                        if weights.grad is not None:
                            weights.grad.zero_()
                        if bias is not None and bias.grad is not None:
                            bias.grad.zero_()
                    opt_local.step()

                self.net[layer_idx].apply_noise(neurons_per_layer[layer_idx])
