import torch
from .base import Strategy

class DifferentialStrategy(Strategy):
    def __init__(self, lr_backbone=1e-4, lr_head=1e-3):
        self.lr_backbone = lr_backbone
        self.lr_head = lr_head

    def build_optimizer(self, model):
        return torch.optim.Adam([
            {"params": model.backbone.parameters(), "lr": self.lr_backbone},
            {"params": model.classifier.parameters(), "lr": self.lr_head},
            {"params": model.regressor.parameters(), "lr": self.lr_head},
        ])