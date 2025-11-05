import argparse


class Config:
    def __init__(self) -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--device", default="cuda", help="Compute device")
        parser.add_argument("--optimizer", default="Adam", choices=["Adam", "SGD"])
        parser.add_argument("--lr", default=1e-3, type=float)
        parser.add_argument("--weight-decay", default=1e-4, type=float)
        parser.add_argument("--beta1", default=0.9, type=float)
        parser.add_argument("--beta2", default=0.999, type=float)
        parser.add_argument("--momentum", default=0.9, type=float)
        parser.add_argument("--grow-ratio", default=0.35, type=float)
        parser.add_argument("--batch-size", default=128, type=int)
        parser.add_argument("--seed", default=0, type=int)
        parser.add_argument("--verbose", action="store_true")

        args = parser.parse_args()

        self.device = args.device
        self.optimizer = args.optimizer
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.momentum = args.momentum
        self.grow_ratio = args.grow_ratio
        self.batch_size = args.batch_size
        self.seed = args.seed
        self.verbose = args.verbose

    def add_argument(self, *args, **kwargs):
        raise RuntimeError("Config arguments must be declared at construction time.")
