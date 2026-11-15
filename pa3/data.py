"""Leitura do pacote de dados do PA3 (ENA24-detection, 21 classes).

O enunciado descreve o conteúdo do pacote (train, val, test_public,
rotulos_1/5/20.csv, classes.csv), mas não o layout exato de pastas nem os nomes
das colunas dos CSVs. Por isso o carregador *descobre* o layout em vez de
assumi-lo: procura o classes.csv, indexa todas as imagens por split e detecta
as colunas de id e de rótulo. Se algo não bater, o erro diz o que foi
encontrado; rode `python scripts/inspect_data.py` e ajuste as constantes abaixo.
"""
from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = ("train", "val", "test_public", "test_private")

# Nomes de coluna aceitos (minúsculas). Ajuste aqui se o pacote usar outros.
ID_COLS = ("image_id", "id", "image", "img", "filename", "file_name", "file", "path", "arquivo")
LABEL_COLS = ("label", "labels", "class", "classe", "class_name", "species", "especie", "target", "category")
_CLASS_HEADER_WORDS = {"id", "class_id", "idx", "index", "class", "classe", "class_name",
                       "name", "nome", "label", "species", "especie"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# Descoberta do layout
# --------------------------------------------------------------------------- #
def find_root(path: str | Path) -> Path:
    """Devolve a pasta que contém classes.csv (procura dentro de `path`)."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"Pasta de dados não encontrada: {p}\n"
            "Baixe e descompacte pa3_dados.zip e passe --data CAMINHO "
            "(ou defina a variável de ambiente PA3_DATA)."
        )
    hits = sorted(p.rglob("classes.csv"), key=lambda q: len(q.parts))
    if not hits:
        raise FileNotFoundError(
            f"Não achei classes.csv dentro de {p}. Conteúdo: "
            f"{sorted(x.name for x in p.iterdir())[:20]}"
        )
    return hits[0].parent


def read_classes(root: Path) -> list[str]:
    """Nomes das classes, na ordem dos índices 0..K-1."""
    path = Path(root) / "classes.csv"
    raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
    first = [str(c).strip().lower() for c in raw.iloc[0]]
    df = pd.read_csv(path, dtype=str, keep_default_na=False) if any(
        c in _CLASS_HEADER_WORDS for c in first) else raw
    df = df.apply(lambda col: col.str.strip())
    if df.shape[1] == 1:
        return df.iloc[:, 0].tolist()
    id_col = next((c for c in df.columns if df[c].str.fullmatch(r"\d+").all()), None)
    name_col = next(c for c in df.columns if c != id_col)
    if id_col is not None:
        df = df.assign(_i=df[id_col].astype(int)).sort_values("_i")
    return df[name_col].tolist()


class ImageIndex:
    """Índice stem -> caminho de todas as imagens abaixo de `root`, por split."""

    def __init__(self, root: Path):
        self.by_split: dict[str, dict[str, Path]] = {s: {} for s in SPLIT_DIRS}
        self.any: dict[str, Path] = {}
        for p in sorted(Path(root).rglob("*")):
            if p.suffix.lower() not in IMG_EXT:
                continue
            parent_parts = p.relative_to(root).parts[:-1]
            for s in SPLIT_DIRS:
                if s in parent_parts:
                    self.by_split[s].setdefault(p.stem, p)
                    break
            self.any.setdefault(p.stem, p)

    def find(self, split: str, image_id: str) -> Path | None:
        stem = Path(image_id).stem if Path(image_id).suffix.lower() in IMG_EXT else image_id
        return self.by_split.get(split, {}).get(stem) or self.any.get(stem)


@lru_cache(maxsize=4)
def get_index(root: str) -> ImageIndex:
    return ImageIndex(Path(root))


def find_csv(root: Path, name: str) -> Path | None:
    """Acha o CSV de um split ('rotulos_5', 'val', 'test_public', ...)."""
    root = Path(root)
    names = [f"{name}.csv"]
    if not name.startswith("rotulos"):
        names += [f"{name}_labels.csv", f"labels_{name}.csv", f"{name}_rotulos.csv",
                  f"rotulos_{name}.csv"]
    for n in names:
        hits = sorted(root.rglob(n), key=lambda q: len(q.parts))
        if hits:
            return hits[0]
    for n in ("labels.csv", "rotulos.csv"):  # CSV dentro da pasta do split
        if (root / name / n).exists():
            return root / name / n
    return None


# --------------------------------------------------------------------------- #
# Itens (id, caminho, rótulo)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Item:
    image_id: str  # exatamente como aparece no CSV (ou o stem do arquivo, no train)
    path: Path
    label: int     # -1 = sem rótulo


def _pick(columns, preferred) -> str | None:
    low = {str(c).strip().lower(): c for c in columns}
    return next((low[p] for p in preferred if p in low), None)


def _read_labeled_csv(csv: Path, classes: list[str], index: ImageIndex, split_dir: str) -> list[Item]:
    df = pd.read_csv(csv, dtype=str, keep_default_na=False)
    id_col = _pick(df.columns, ID_COLS) or df.columns[0]
    lab_col = _pick(df.columns, LABEL_COLS) or next(c for c in df.columns if c != id_col)
    name2idx = {n.strip().lower(): i for i, n in enumerate(classes)}
    items, missing, unknown = [], [], set()
    for image_id, lab in zip(df[id_col].str.strip(), df[lab_col].str.strip()):
        if lab.isdigit() and int(lab) < len(classes):
            y = int(lab)
        elif lab.lower() in name2idx:
            y = name2idx[lab.lower()]
        else:
            unknown.add(lab)
            continue
        path = index.find(split_dir, image_id)
        if path is None:
            missing.append(image_id)
            continue
        items.append(Item(image_id, path, y))
    if unknown:
        raise ValueError(f"{csv.name}: rótulos que não estão em classes.csv: {sorted(unknown)[:10]}")
    if missing:
        raise FileNotFoundError(
            f"{csv.name}: {len(missing)} imagens do CSV não foram achadas em disco "
            f"(ex.: {missing[:3]}). Colunas usadas: id='{id_col}', rótulo='{lab_col}'."
        )
    return items


def load_split(root: str | Path, name: str, classes: list[str] | None = None) -> list[Item]:
    """Carrega um split. `name` ∈ {train, val, test_public, rotulos_1, rotulos_5, rotulos_20}.

    `train` vem sem rótulo (label = -1): são todas as imagens da pasta train.
    """
    root = Path(root)
    classes = classes or read_classes(root)
    index = get_index(str(root))
    if name == "train":
        if not index.by_split["train"]:
            raise FileNotFoundError(f"Nenhuma imagem em uma pasta 'train' dentro de {root}.")
        return [Item(p.stem, p, -1) for p in sorted(index.by_split["train"].values())]
    csv = find_csv(root, name)
    if csv is None:
        raise FileNotFoundError(
            f"Não achei o CSV de '{name}' em {root}. Rode scripts/inspect_data.py "
            "para ver o que o pacote contém e ajuste find_csv() se preciso."
        )
    split_dir = "train" if name.startswith("rotulos") else name
    return _read_labeled_csv(csv, classes, index, split_dir)


def labeled_budget(root, budget: int, classes=None) -> list[Item]:
    return load_split(root, f"rotulos_{budget}", classes)


def ids_have_extension(items: list[Item]) -> bool:
    """Os image_id dos CSVs trazem extensão (.jpg)? predict.py segue o mesmo formato."""
    return any(Path(i.image_id).suffix.lower() in IMG_EXT for i in items)


def class_counts(items: list[Item], num_classes: int) -> np.ndarray:
    return np.bincount([i.label for i in items if i.label >= 0], minlength=num_classes)


def check_splits(train: list[Item], labeled: list[Item], val: list[Item], test: list[Item]) -> None:
    """Garante: rótulos ⊂ train, e train/val/test_public sem imagens em comum."""
    s = lambda items: {Path(i.image_id).stem for i in items}  # noqa: E731
    tr, lab, va, te = s(train), s(labeled), s(val), s(test)
    problems = []
    if not lab <= tr:
        problems.append(f"{len(lab - tr)} imagens rotuladas não estão no train (ex.: {sorted(lab - tr)[:3]})")
    for (na, a), (nb, b) in [(("train", tr), ("val", va)), (("train", tr), ("test_public", te)),
                              (("val", va), ("test_public", te))]:
        if a & b:
            problems.append(f"{len(a & b)} imagens em '{na}' e '{nb}' ao mesmo tempo (ex.: {sorted(a & b)[:3]})")
    if problems:
        raise AssertionError("Problema de split:\n  " + "\n  ".join(problems))


# --------------------------------------------------------------------------- #
# Dataset e transformações
# --------------------------------------------------------------------------- #
def load_image(path: Path, size: int) -> Image.Image:
    """Abre, converte para RGB e redimensiona para size×size (sem recorte).

    O recorte central cortaria animais pequenos nas bordas; esticar mantém todo o
    campo de visão. O custo é uma leve distorção de proporção, igual no treino e
    no teste.
    """
    try:
        img = Image.open(path).convert("RGB")
        return img.resize((size, size), Image.Resampling.BICUBIC)
    except (OSError, ValueError) as e:
        warnings.warn(f"Falha ao ler {path}: {e}. Usando imagem cinza no lugar.")
        return Image.new("RGB", (size, size), (124, 116, 104))


def train_transform(img_size: int = 224, scale=(0.5, 1.0)) -> T.Compose:
    """Aumentação do treino supervisionado. `scale` baixo demais corta o animal fora."""
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=scale, ratio=(3 / 4, 4 / 3),
                            interpolation=T.InterpolationMode.BICUBIC, antialias=True),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.2),
        T.RandomGrayscale(0.1),  # fotos noturnas (infravermelho) são sem cor
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def eval_transform() -> T.Compose:
    return T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


class PA3Dataset(Dataset):
    """Imagens pré-carregadas na RAM em size×size (rápido para treinar várias vezes)."""

    def __init__(self, items: list[Item], size: int, transform=None, preload: bool = True,
                 workers: int = 8):
        self.items, self.size, self.transform = list(items), size, transform
        self.images = None
        if preload:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                self.images = list(ex.map(lambda it: load_image(it.path, size), self.items))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        img = self.images[i] if self.images is not None else load_image(self.items[i].path, self.size)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.items[i].label

    @property
    def labels(self) -> torch.Tensor:
        return torch.tensor([it.label for it in self.items], dtype=torch.long)


def make_train_ds(items, img_size: int = 224, load_size: int = 256, scale=(0.5, 1.0)) -> PA3Dataset:
    """Carrega em load_size (>= img_size) para o RandomResizedCrop ter folga de resolução."""
    return PA3Dataset(items, load_size, train_transform(img_size, scale))


def make_eval_ds(items, img_size: int = 224, preload: bool = True) -> PA3Dataset:
    return PA3Dataset(items, img_size, eval_transform(), preload=preload)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, seed: int = 0, num_workers: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, generator=g, num_workers=num_workers,
        drop_last=shuffle and len(ds) > batch_size,  # BatchNorm não aceita batch de 1
        pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
    )
