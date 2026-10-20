from abc import ABC, abstractmethod
import torch

class Strategy(ABC):
    # Crea el optimizador inicial
    @abstractmethod
    def build_optimizer(self, model) -> torch.optim.Optimizer:
        pass

    def on_epoch_start(self, model, epoch: int):
        return None   # devuelve un optimizador nuevo si cambia, o None si se mantiene el actual