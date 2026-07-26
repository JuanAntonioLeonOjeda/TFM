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
    def __init__(self, num_classes: int = 3, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        # ---- shared backbone ----
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        # number of features the backbone outputs before its classifier (2048 for ResNet50)
        in_features = backbone.fc.in_features

        # remove the original 1000-class classifier: keep everything up to the pooling
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # ---- head 1: classification (3 classes) ----
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        # ---- head 2: regression (BMD) ----
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        features = self.backbone(x)          # (B, 2048)

        logits = self.classifier(features)   # (B, num_classes)
        bmd = self.regressor(features)       # (B, 1)

        return logits, bmd.squeeze(1)        # bmd -> (B,)


if __name__ == "__main__":
    device = get_device()
    print(f"Device: {device}")

    model = LumosNet(pretrained=True).to(device)

    # dummy forward pass to check shapes
    dummy = torch.randn(4, 3, 224, 224, device=device)
    logits, bmd = model(dummy)

    print(f"Input : {tuple(dummy.shape)}")
    print(f"Logits: {tuple(logits.shape)}   (expected (4, 3))")
    print(f"BMD   : {tuple(bmd.shape)}      (expected (4,))")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")