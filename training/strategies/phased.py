import torch
from .base import Strategy

class PhasedStrategy(Strategy):
    def __init__(self, freeze_epochs=8, lr_frozen=1e-3, lr_finetune=1e-4, weight_decay=1e-4):
        self.freeze_epochs = freeze_epochs
        self.lr_frozen = lr_frozen
        self.lr_finetune = lr_finetune
        self.weight_decay = weight_decay

    def build_optimizer(self, model):
        for p in model.backbone.parameters():
            p.requires_grad = False
        heads = list(model.classifier.parameters()) + list(model.regressor.parameters())
        return torch.optim.Adam(heads, lr=self.lr_frozen, weight_decay=self.weight_decay)

    def on_epoch_start(self, model, epoch):
        if epoch == self.freeze_epochs + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            return torch.optim.Adam(model.parameters(), lr=self.lr_finetune, weight_decay=self.weight_decay)
        return None