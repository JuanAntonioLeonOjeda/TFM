import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LumosNet(nn.Module):
    """ResNet50 + capa compartida + dos cabezas: clasificación (3 clases) y regresión de BMD.

    num_covariates > 0 añade una rama que embebe las covariables clínicas y las
    concatena con las features de imagen antes de la capa compartida.
    """

    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3,
                 num_covariates: int = 0, covariate_embed_dim: int = 32, shared_dim: int = 512):
        super().__init__()

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)
        in_features = backbone.fc.in_features  # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.num_covariates = num_covariates
        if num_covariates > 0:
            self.covariate_embed = nn.Sequential(
                nn.Linear(num_covariates, covariate_embed_dim),
                nn.ReLU(inplace=True),
            )
            combined_dim = in_features + covariate_embed_dim
        else:
            self.covariate_embed = None
            combined_dim = in_features

        self.shared = nn.Sequential(
            nn.Linear(combined_dim, shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.classifier = self._head(shared_dim, num_classes, dropout)
        self.regressor = self._head(shared_dim, 1, dropout)

    @staticmethod
    def _head(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, out_dim),
        )

    def forward(self, x, covariates: torch.Tensor | None = None):
        features = self.backbone(x)

        if self.covariate_embed is not None:
            if covariates is None:
                raise ValueError("Modelo creado con num_covariates > 0: forward() necesita covariates.")
            features = torch.cat([features, self.covariate_embed(covariates)], dim=1)

        shared = self.shared(features)
        return self.classifier(shared), self.regressor(shared).squeeze(1)

    def head_parameters(self):
        """Todo lo que no es backbone. Las estrategias deben usar esto para no dejar capas fuera del optimizador."""
        params = list(self.classifier.parameters()) + \
                 list(self.regressor.parameters()) + \
                 list(self.shared.parameters())
        if self.covariate_embed is not None:
            params += list(self.covariate_embed.parameters())
        return params


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")
    dummy = torch.randn(4, 3, 224, 224, device=device)

    for n_cov in (0, 3):
        model = LumosNet(pretrained=True, num_covariates=n_cov).to(device)
        cov = torch.randn(4, n_cov, device=device) if n_cov else None
        logits, bmd = model(dummy, cov)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"num_covariates={n_cov}: logits {tuple(logits.shape)}, bmd {tuple(bmd.shape)}, "
              f"{n_params:,} params")
