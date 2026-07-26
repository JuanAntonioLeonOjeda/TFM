from abc import ABC, abstractmethod
import torch

class Strategy(ABC):
    @abstractmethod
    def build_optimizer(self, model) -> torch.optim.Optimizer:
        """Crea el optimizador inicial."""

    def on_epoch_start(self, model, epoch: int):
        """Se llama al inicio de cada época. Por defecto no hace nada.
        Las estrategias por fases lo usan para descongelar."""
        return None   # devuelve un optimizador nuevo si cambia, o None