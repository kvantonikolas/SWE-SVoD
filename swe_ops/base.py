import torch
import torch.nn as nn
import torch.nn.functional as F


class SWEModule(nn.Module):

    def __init__(self, *, can_expand: bool = True, activation: str = "relu", use_batch_norm: bool = False, use_bias: bool = True) -> None:
        super().__init__()
        self.can_expand = can_expand
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.use_bias = use_bias

        self.module: nn.Module | None = None
        self.bn: nn.Module | None = None

    @property
    def device(self) -> torch.device:
        if self.module is None or not hasattr(self.module, "weight"):
            raise RuntimeError("SWE module not initialised")
        return self.module.weight.device  # type: ignore[attr-defined]

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "leaky_relu":
            return F.leaky_relu(x, negative_slope=0.2)
        if self.activation == "swish":
            return x * torch.sigmoid(x)
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation == "softplus":
            return F.softplus(x)
        if self.activation == "none":
            return x
        raise NotImplementedError(f"Unsupported activation {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.module is None:
            raise RuntimeError("SWE module not initialised")
        x = self.module(x)
        if self.bn is not None:
            x = self.bn(x)
        return self._activate(x)

    def clear(self) -> None:
        pass
