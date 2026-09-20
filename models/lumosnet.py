import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


def get_device():
    """Returns the best available device (MPS on Apple Silicon, else CUDA, else CPU)."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class LumosNet(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3,
                 num_covariates: int = 0, covariate_embed_dim: int = 32, shared_dim: int = 512):
        """
        num_covariates: número de covariables clínicas a concatenar (p.ej. 3
            para age/gender/BMI). 0 (default) desactiva esta rama por
            completo -- el modelo se comporta como la versión original,
            salvo por la capa compartida (ver más abajo), que SIEMPRE está
            presente, con o sin covariables.
        covariate_embed_dim: tamaño del embedding aprendido de las
            covariables, antes de concatenarlas con las features de imagen.
        shared_dim: tamaño de la capa compartida, colocada ANTES de
            bifurcar en las dos cabezas (clasificación y regresión).
        """
        super().__init__()

        # ---- shared backbone ----
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # number of features the backbone outputs before its classifier (2048 for ResNet50)
        in_features = backbone.fc.in_features

        # remove the original 1000-class classifier: keep everything up to the pooling
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # ---- covariables clínicas (opcional) ----
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

        # ---- capa compartida, ANTES de bifurcar en las dos cabezas ----
        self.shared = nn.Sequential(
            nn.Linear(combined_dim, shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ---- head 1: classification (3 classes) ----
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shared_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # ---- head 2: regression (BMD) ----
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(shared_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x, covariates: torch.Tensor | None = None):
        features = self.backbone(x)          # (B, 2048)

        if self.covariate_embed is not None:
            if covariates is None:
                raise ValueError(
                    "Este modelo se creó con num_covariates > 0, pero forward() "
                    "no recibió covariates. Revisa que el DataLoader las esté "
                    "devolviendo y que train.py/evaluate.py las estén pasando."
                )
            cov_emb = self.covariate_embed(covariates)          # (B, covariate_embed_dim)
            features = torch.cat([features, cov_emb], dim=1)    # (B, 2048 + embed_dim)

        shared = self.shared(features)        # (B, shared_dim)

        logits = self.classifier(shared)      # (B, num_classes)
        bmd = self.regressor(shared)          # (B, 1)

        return logits, bmd.squeeze(1)         # bmd -> (B,)

    def head_parameters(self):
        """
        Todos los parámetros que NO son del backbone: classifier, regressor,
        la capa compartida y (si existe) el embedding de covariables.

        Las estrategias de entrenamiento (training/strategies/) deben usar
        esto en vez de enumerar `model.classifier.parameters() + ...` a
        mano, para que un futuro cambio de arquitectura (nuevas capas entre
        el backbone y las cabezas) no las deje fuera del optimizador de
        forma silenciosa -- exactamente el bug que motivó este método.
        """
        params = list(self.classifier.parameters()) + \
                 list(self.regressor.parameters()) + \
                 list(self.shared.parameters())
        if self.covariate_embed is not None:
            params += list(self.covariate_embed.parameters())
        return params


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")

    print("\n--- Sin covariables (comportamiento equivalente al original + capa compartida) ---")
    model = LumosNet(pretrained=True).to(device)
    dummy = torch.randn(4, 3, 224, 224, device=device)
    logits, bmd = model(dummy)
    print(f"Input : {tuple(dummy.shape)}")
    print(f"Logits: {tuple(logits.shape)}   (expected (4, 3))")
    print(f"BMD   : {tuple(bmd.shape)}      (expected (4,))")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")

    print("\n--- Con 3 covariables (age, gender, BMI) ---")
    model_cov = LumosNet(pretrained=True, num_covariates=3).to(device)
    dummy_cov = torch.randn(4, 3, device=device)
    logits, bmd = model_cov(dummy, dummy_cov)
    print(f"Logits: {tuple(logits.shape)}   (expected (4, 3))")
    print(f"BMD   : {tuple(bmd.shape)}      (expected (4,))")
    n_params_cov = sum(p.numel() for p in model_cov.parameters())
    print(f"Total parameters: {n_params_cov:,}  (+{n_params_cov - n_params:,} vs sin covariables)")