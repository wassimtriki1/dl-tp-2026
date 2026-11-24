"""TP6 : vérification de l'environnement (à lancer une fois, via ex0.sh)."""
import os
import platform
import shutil

import torch
import torchvision

from common import RAW_ROOT, SMALL_ROOT, WNIDS, build_model


def cpu_flags():
    try:
        with open("/proc/cpuinfo") as f:
            flags = next(line for line in f if line.startswith("flags")).split()
    except (OSError, StopIteration):
        return "inconnus"
    wanted = ["avx512f", "avx512_vnni", "avx512_bf16", "amx_bf16", "amx_int8"]
    return ", ".join(fl for fl in wanted if fl in flags) or "aucun des drapeaux recherchés"


def count_images(root):
    try:
        return sum(len(list((root / "train" / w).iterdir())) for w in WNIDS), \
               sum(len(list((root / "val" / w).iterdir())) for w in WNIDS)
    except OSError as e:
        return f"inaccessible ({e.__class__.__name__})", ""


print(f"Machine        : {platform.node()}")
print(f"Python         : {platform.python_version()}")
print(f"torch          : {torch.__version__} (CUDA {torch.version.cuda})")
print(f"torchvision    : {torchvision.__version__}")
try:
    import torchao
    print(f"torchao        : {torchao.__version__}")
except ImportError:
    print("torchao        : NON INSTALLÉ (pip install torchao==0.18.0)")
print(f"CPU alloués    : {len(os.sched_getaffinity(0))}  ({platform.processor() or 'x86_64'})")
print(f"Drapeaux CPU   : {cpu_flags()}")
print(f"gcc            : {shutil.which('gcc') or 'absent'}")
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"GPU            : {torch.cuda.get_device_name()}")
    print(f"Mémoire GPU    : {total / 2**20:.0f} Mio au total, {free / 2**20:.0f} Mio libres")
    print(f"bf16 supporté  : {torch.cuda.is_bf16_supported()}")
else:
    print("GPU            : aucun")
print(f"Données raw    : {RAW_ROOT} -> {count_images(RAW_ROOT)} images (train, val)")
print(f"Données small  : {SMALL_ROOT} -> {count_images(SMALL_ROOT)} images (train, val)")

# Télécharge une fois les poids pré-entraînés (cache ~/.cache/torch) pour train.py
build_model("resnet50", pretrained=True)
print("Poids ResNet-50 pré-entraînés : OK")
