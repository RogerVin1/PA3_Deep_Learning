"""Modelos do PA3: CNN pequena (do zero) e ResNet-18 (ImageNet), mais checkpoints."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from . import MAX_PARAMS


class SmallCNN(nn.Module):
    """CNN de 5 blocos (2 convs + BN + ReLU + maxpool), ~2,4 M de parâmetros.

    Serve de piso "do zero" (Parte 0a). O classificador se chama `fc`, como na ResNet,
    para o treino tratar cabeça e corpo igual nos dois modelos.
    """

    def __init__(self, num_classes: int, width: int = 32, dropout: float = 0.3):
        super().__init__()
        chans = [width, width * 2, width * 4, width * 8, width * 8]
        layers, c_in = [], 3
        for c in chans:
            layers += [
                nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(chans[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.features(x)).flatten(1)
        return self.fc(self.drop(x))


ARCHS = ("small_cnn", "resnet18")


def build_model(arch: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    if arch == "small_cnn":
        return SmallCNN(num_classes)
    if arch == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"arch desconhecida: {arch}. Opções: {ARCHS}")


def count_params(model: nn.Module) -> int:
    """Contagem exatamente como no enunciado: sum(p.numel() for p in model.parameters())."""
    return sum(p.numel() for p in model.parameters())


def check_param_budget(model: nn.Module) -> int:
    n = count_params(model)
    if n > MAX_PARAMS:
        raise ValueError(f"Modelo com {n:,} parâmetros passa do limite de {MAX_PARAMS:,}.")
    return n


# --------------------------------------------------------------------------- #
# Checkpoints: tudo que predict.py precisa para rodar sozinho, sem internet.
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str | Path, model: nn.Module, arch: str, class_names: list[str],
                    img_size: int, id_has_ext: bool, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": arch,
        "class_names": list(class_names),
        "img_size": int(img_size),
        "id_has_ext": bool(id_has_ext),
        "extra": extra or {},
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
    }, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = build_model(ckpt["arch"], len(ckpt["class_names"]), pretrained=False)  # sem baixar nada
    model.load_state_dict(ckpt["state_dict"])
    meta = {k: v for k, v in ckpt.items() if k != "state_dict"}
    return model.eval(), meta
