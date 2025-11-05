import time
from typing import Dict

import torch.nn as nn

from .base import SWEModule


class SWENetwork(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.ModuleList()
        self.next_layers: Dict[int, list[int]] = {}
        self.layers_to_expand: list[int] = []
        self.verbose = True
        self.grow_ratio = 0.0

    def clear(self) -> None:
        for layer in self.net:
            if isinstance(layer, SWEModule) and hasattr(layer, "clear"):
                layer.clear()

    def grow(self, method: str, loader, new_neurons: int, draw_details=None, n_batches: int = -1):
        if method != "swe":
            raise NotImplementedError("Only swe growth is retained in this code path.")

        if self.verbose:
            print(f"[INFO] start swe growth (adding {new_neurons} neurons)")

        start_time = time.time()
        self.net.eval()

        self.swe(loader, n_batches, number_new_neurons=new_neurons)

        self.net.train()
        self.create_optimizer()

        if self.verbose:
            duration = time.time() - start_time
            print(f"[INFO] swe growth finished in {duration:0.2f}s")

        return {}

    def split(self, split_method: str, loader, ner: int, draw_details=None, n_batches: int = -1):
        return self.grow(split_method, loader, ner, draw_details, n_batches)
