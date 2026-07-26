import torch
from .base import Strategy

class PhasedStrategy(Strategy):
    def __init__(self, freeze_epochs=8, lr_frozen=1e-3, lr_finetune=1e-5):
        self.freeze_epochs = freeze_epochs
        self.lr_frozen = lr_frozen
        self.lr_finetune = lr_finetune

    def build_optimizer(self, model):
        for p in model.backbone.parameters():   # arranca congelado
            p.requires_grad = False
        heads = list(model.classifier.parameters()) + list(model.regressor.parameters())
        return torch.optim.Adam(heads, lr=self.lr_frozen)

    def on_epoch_start(self, model, epoch):
        if epoch == self.freeze_epochs + 1:      # descongela y cambia optimizador
            for p in model.backbone.parameters():
                p.requires_grad = True
            return torch.optim.Adam(model.parameters(), lr=self.lr_finetune)
        return None