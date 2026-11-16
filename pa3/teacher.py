from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

TEACHERS = {
    "bioclip2": "hf-hub:imageomics/bioclip-2",   # ViT-L/14, treinado em espécies (TreeOfLife-200M)
    "bioclip": "hf-hub:imageomics/bioclip",      # ViT-B/16, treinado em espécies (TreeOfLife-10M)
}

# Nome científico de cada classe do ENA24: BioCLIP foi treinado com nomes taxonômicos.
SCIENTIFIC = {
    "american_black_bear": "Ursus americanus",
    "american_crow": "Corvus brachyrhynchos",
    "bird": "Aves",
    "bobcat": "Lynx rufus",
    "chicken": "Gallus gallus domesticus",
    "coyote": "Canis latrans",
    "dog": "Canis lupus familiaris",
    "domestic_cat": "Felis catus",
    "eastern_chipmunk": "Tamias striatus",
    "eastern_cottontail": "Sylvilagus floridanus",
    "eastern_fox_squirrel": "Sciurus niger",
    "eastern_gray_squirrel": "Sciurus carolinensis",
    "grey_fox": "Urocyon cinereoargenteus",
    "horse": "Equus caballus",
    "northern_raccoon": "Procyon lotor",
    "red_fox": "Vulpes vulpes",
    "striped_skunk": "Mephitis mephitis",
    "virginia_opossum": "Didelphis virginiana",
    "white_tailed_deer": "Odocoileus virginianus",
    "wild_turkey": "Meleagris gallopavo",
    "woodchuck": "Marmota monax",
}


def class_prompts(class_names: list[str]) -> list[list[str]]:
    """Três prompts por classe (nome comum, científico, os dois); os embeddings são promediados."""
    out = []
    for name in class_names:
        common = name.replace("_", " ")
        sci = SCIENTIFIC.get(name)
        prompts = [f"a photo of {common}."]
        if sci:
            prompts += [f"a photo of {sci}.", f"a photo of {sci}, {common}."]
        out.append(prompts)
    return out


def load_teacher(name: str, device):
    """Devolve (modelo, pré-processamento próprio do modelo, tokenizador). `name`: chave de TEACHERS
    ou qualquer identificador do open_clip (ex.: 'hf-hub:imageomics/bioclip')."""
    import open_clip

    ident = TEACHERS.get(name, name)
    model, _, preprocess = open_clip.create_model_and_transforms(ident)
    tokenizer = open_clip.get_tokenizer(ident)
    return model.to(device).eval(), preprocess, tokenizer


def squash_transform(model, preprocess) -> T.Compose:
    """Redimensiona a imagem inteira para o tamanho do modelo, SEM recorte central.

    O pré-processamento padrão do CLIP corta o centro e perderia animais pequenos nas bordas
    do quadro; esticar mantém todo o campo de visão (igual ao que o aluno vai ver).
    """
    norm = next(t for t in preprocess.transforms if isinstance(t, T.Normalize))
    size = model.visual.image_size
    size = size[0] if isinstance(size, (tuple, list)) else size
    return T.Compose([T.Resize((size, size), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
                      T.ToTensor(), norm])


class _Images(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.transform(Image.open(self.items[i].path).convert("RGB"))


@torch.no_grad()
def encode_images(model, items, transform, device, batch_size: int = 128, num_workers: int = 2) -> torch.Tensor:
    """Embeddings normalizados (N, D) das imagens, lidas nas 512 px originais."""
    loader = DataLoader(_Images(items, transform), batch_size=batch_size, shuffle=False, num_workers=num_workers)
    feats = []
    for x in loader:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            f = model.encode_image(x.to(device))
        feats.append(F.normalize(f.float(), dim=-1).cpu())
    return torch.cat(feats)


@torch.no_grad()
def encode_classes(model, tokenizer, prompts: list[list[str]], device) -> torch.Tensor:
    """Um embedding de texto (K, D) por classe: média dos prompts da classe, renormalizada."""
    out = []
    for plist in prompts:
        f = F.normalize(model.encode_text(tokenizer(plist).to(device)).float(), dim=-1).mean(0)
        out.append(F.normalize(f, dim=-1).cpu())
    return torch.stack(out)


def zero_shot_logits(img_feats: torch.Tensor, txt_feats: torch.Tensor, scale: float) -> torch.Tensor:
    """Logits do professor: similaridade de cosseno × escala aprendida (logit_scale do CLIP)."""
    return scale * img_feats @ txt_feats.T


def softmax_confidence(logits: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """(rótulo previsto, confiança = maior probabilidade do softmax) de cada imagem."""
    p = logits.softmax(1)
    conf, pred = p.max(1)
    return pred.numpy(), conf.numpy()
