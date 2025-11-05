import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import SWEModule


class SWEFullyConnected(SWEModule):

    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        can_expand: bool = True,
        use_bias: bool = True,
        activation: str = "relu",
        use_batch_norm: bool = False,
    ) -> None:
        super().__init__(can_expand=can_expand, activation=activation, use_batch_norm=use_batch_norm, use_bias=use_bias)

        if use_batch_norm:
            self.bn = nn.BatchNorm1d(out_features)
            self.use_bias = False

        self.module = nn.Linear(in_features=in_features, out_features=out_features, bias=self.use_bias)
        if self.use_bias:
            with torch.no_grad():
                self.module.bias.data.abs_()

        self.expanded_out = 0
        self.expanded_in = 0
        self.y = None
        self.w = None
        self.new_weights = []
        self.noise = None
        self.noise_bias = None

    def distributor_add_new(self, *, enlarge_out: bool = True, enlarge_in: bool = True, total_neurons: int = 0) -> None:
        self.expanded_out = total_neurons if enlarge_out else 0
        self.expanded_in = total_neurons if enlarge_in else 0

        n_out, n_in = self.module.weight.shape
        device = self.device

        new_layer = nn.Linear(
            in_features=n_in + self.expanded_in,
            out_features=n_out + self.expanded_out,
            bias=self.use_bias,
        ).to(device)

        new_layer.weight.data[:n_out, :n_in] = self.module.weight.data.clone()
        if self.use_bias:
            new_layer.bias.data[:n_out] = self.module.bias.data.clone()

        if self.expanded_in > 0:
            new_layer.weight.data[:, n_in:] = torch.empty_like(new_layer.weight.data[:, n_in:]).uniform_(-1e-2, 1e-2)
        if self.expanded_out > 0:
            new_layer.weight.data[n_out:, :] = torch.empty_like(new_layer.weight.data[n_out:, :]).uniform_(-1e-2, 1e-2)
            if self.use_bias:
                new_layer.bias.data[n_out:] = torch.empty_like(new_layer.bias.data[n_out:]).uniform_(-1e-2, 1e-2)

        self.module = new_layer

    def distributor_forward(self, x: torch.Tensor, _targets: torch.Tensor, total_neurons: int = 0) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        out = self.module(x)
        if total_neurons > 0 and self.y is not None:
            mod = out[:, -total_neurons:] * (1 + self.y)
            out = torch.cat([out[:, :-total_neurons], mod], dim=1)
        out = self._activate(out)
        self.output = out
        return out

    def distributor_update_w(self, batches: int) -> None:
        if self.y is None or self.y.grad is None:
            return
        if self.w is None:
            self.w = torch.zeros(self.y.shape[-1], device=self.device)
        divisor = max(float(batches), 1.0)
        self.w += (self.y.grad.data / divisor).view(-1)
        self.y.grad = None

    def swe_reset(self, neurons_info, *, simple: bool = True, layer_num: int = 0, total_neurons: int = 0) -> None:
        device = self.device
        self.new_weights = []

        if simple:
            count = total_neurons if total_neurons > 0 else self.module.weight.size(0)
            self.y = nn.Parameter(torch.zeros(1, count, device=device))
            self.y.retain_grad()
            self.w = torch.zeros(count, device=device)
            return

        if isinstance(neurons_info, dict):
            num_new = neurons_info.get(layer_num, 0)
        else:
            num_new = int(neurons_info)

        if num_new <= 0:
            self.noise = None
            self.noise_bias = None
            return

        n_out, n_in = self.module.weight.shape
        self.noise = nn.Parameter(torch.randn(num_new, n_out, n_in, device=device) * 1e-4)
        if self.use_bias:
            self.noise_bias = nn.Parameter(torch.randn(num_new, n_out, device=device) * 1e-4)

    def swe_active_expand(self, number_new_neurons: int = 1):
        if number_new_neurons < 1 or not self.new_weights:
            return None

        n_out, n_in = self.module.weight.shape
        device = self.device
        new_layer = nn.Linear(in_features=n_in, out_features=n_out + number_new_neurons, bias=self.use_bias).to(device)

        new_layer.weight.data[:n_out] = self.module.weight.data.clone()
        if self.use_bias:
            new_layer.bias.data[:n_out] = self.module.bias.data.clone()

        for idx, new_w in enumerate(self.new_weights[:number_new_neurons]):
            new_layer.weight.data[n_out + idx] = new_w[:-1].reshape(n_in)
            if self.use_bias:
                new_layer.bias.data[n_out + idx] = new_w[-1]

        self.module = new_layer
        return None

    def swe_passive_expand(self, number_new_neurons: int = 1):
        if number_new_neurons < 1:
            return []

        n_out, n_in = self.module.weight.shape
        device = self.device
        new_layer = nn.Linear(
            in_features=n_in + number_new_neurons,
            out_features=n_out,
            bias=self.use_bias,
        ).to(device)

        new_layer.weight.data[:, :n_in] = self.module.weight.data.clone()
        new_layer.weight.data[:, n_in:] = 0.0
        if self.use_bias and self.module.bias is not None:
            new_layer.bias.data = self.module.bias.data.clone()

        self.module = new_layer
        return [self.module.weight]

    def apply_noise(self, number_new_neurons: int):
        if number_new_neurons < 1 or self.noise is None:
            self.noise = None
            self.noise_bias = None
            return []

        n_out, n_in = self.module.weight.shape
        device = self.device
        n_old = n_out - number_new_neurons

        new_layer = nn.Linear(in_features=n_in, out_features=n_out, bias=self.use_bias).to(device)
        base_w = self.module.weight.data.clone()
        base_b = self.module.bias.data.clone() if self.use_bias and self.module.bias is not None else None

        for idx in range(number_new_neurons):
            noise_j = self.noise[idx, :n_old, :]
            base_w[:n_old] -= noise_j
            base_w[n_old + idx] += noise_j.sum(dim=0)

            if self.use_bias and base_b is not None and self.noise_bias is not None:
                bias_noise_j = self.noise_bias[idx, :n_old]
                base_b[:n_old] -= bias_noise_j
                base_b[n_old + idx] += bias_noise_j.sum()

        new_layer.weight.data.copy_(base_w)
        if base_b is not None:
            new_layer.bias.data.copy_(base_b)

        self.module = new_layer
        self.noise = None
        self.noise_bias = None
        return []

    def swe_forward(self, x: torch.Tensor, number_new_neurons: int):
        x = x.view(x.size(0), -1)
        weight = self.module.weight
        bias = self.module.bias
        n_out = weight.shape[0]
        n_old = n_out - number_new_neurons

        weight_mod = weight.clone()
        bias_mod = bias.clone() if bias is not None else None

        if number_new_neurons > 0 and self.noise is not None:
            for idx in range(number_new_neurons):
                noise_j = self.noise[idx]
                weight_mod[:n_old] -= noise_j
                weight_mod[n_old + idx] += noise_j.sum(dim=0)

                if bias_mod is not None and self.noise_bias is not None:
                    bias_noise_j = self.noise_bias[idx]
                    bias_mod[:n_old] -= bias_noise_j
                    bias_mod[n_old + idx] += bias_noise_j.sum()

        out = F.linear(x, weight_mod, bias_mod)
        return self._activate(out)
