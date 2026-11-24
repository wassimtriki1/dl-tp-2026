"""TP6 : débit d'entraînement d'un ResNet sur un GPU, pour une configuration donnée."""
import argparse
import gc
import time

import torch
import torch.nn as nn

from common import (append_result, benchmark, build_model, device_name, make_dataset,
                    make_loader, make_train_step, median, seed_everything, set_fp32_mode,
                    synchronize)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="")
    p.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--precision", default="fp32", choices=["fp32", "tf32", "bf16"])
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--data", default="synthetic", choices=["synthetic", "raw", "small"])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--no-pin", action="store_true")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--async-demo", action="store_true", help="chronométrer un pas avec et sans synchronisation")
    p.add_argument("--epoch", action="store_true", help="chronométrer aussi une époque complète")
    p.add_argument("--find-max-batch", action="store_true")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def setup(args, device):
    """Construit le modèle et la fonction d'un pas d'entraînement selon la configuration."""
    seed_everything(args.seed)
    set_fp32_mode(tf32=(args.precision == "tf32"))
    model = build_model(args.arch).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else None
    fwd = torch.compile(model) if args.compile else model
    return make_train_step(fwd, opt, nn.CrossEntropyLoss(), amp_dtype)


def batch_source(args, batch_size, device):
    """Renvoie (next_batch, loader) : next_batch() fournit le prochain batch sur le device."""
    mf = torch.channels_last if args.channels_last else torch.contiguous_format
    if args.data == "synthetic":
        x = torch.randn(batch_size, 3, args.image_size, args.image_size, device=device)
        x = x.contiguous(memory_format=mf)
        y = torch.randint(0, 10, (batch_size,), device=device)
        return (lambda: (x, y)), None

    dataset = make_dataset(args.data, "train")
    loader = make_loader(dataset, batch_size, args.workers,
                         pin_memory=not args.no_pin, seed=args.seed)

    def forever():  # on reboucle sur le DataLoader sans stocker les batchs
        while True:
            yield from loader

    it = forever()

    def next_batch():
        x, y = next(it)
        x = x.to(device, non_blocking=True, memory_format=mf)
        y = y.to(device, non_blocking=True)
        return x, y

    return next_batch, loader


def run(args, device):
    step = setup(args, device)
    next_batch, loader = batch_source(args, args.batch_size, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    compile_s = float("nan")
    if args.compile:  # la 1re itération déclenche la compilation : on la chronomètre à part
        t0 = time.perf_counter()
        step(*next_batch())
        synchronize(device)
        compile_s = time.perf_counter() - t0

    rates = []
    for r in range(args.repeat):
        times = benchmark(lambda: step(*next_batch()), device,
                          n_warmup=args.warmup if r == 0 else 2,
                          n_iter=args.steps)
        rates.append(args.batch_size / median(times))

    epoch_s = float("nan")
    if args.epoch and loader is not None:
        mf = torch.channels_last if args.channels_last else torch.contiguous_format
        synchronize(device)
        t0 = time.perf_counter()
        for x, y in loader:
            step(x.to(device, non_blocking=True, memory_format=mf),
                 y.to(device, non_blocking=True))
        synchronize(device)
        epoch_s = time.perf_counter() - t0

    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
    return {
        "tag": args.tag, "materiel": device_name(device), "arch": args.arch,
        "precision": args.precision, "channels_last": args.channels_last,
        "compile": args.compile, "donnees": args.data,
        "workers": args.workers if args.data != "synthetic" else "",
        "pin_memory": (not args.no_pin) if args.data != "synthetic" else "",
        "batch": args.batch_size,
        "img_s": round(median(rates), 1), "img_s_min": round(min(rates), 1),
        "img_s_max": round(max(rates), 1), "pic_memoire_go": round(peak, 2),
        "compilation_s": round(compile_s, 1), "epoque_s": round(epoch_s, 1),
    }


def async_demo(args, device):
    """Un même pas chronométré deux fois : au retour de l'appel côté CPU, puis après synchronisation."""
    step = setup(args, device)
    next_batch, _ = batch_source(args, args.batch_size, device)
    for _ in range(3):  # chauffe
        step(*next_batch())
    x, y = next_batch()
    synchronize(device)  # file d'attente du GPU vide au départ
    t0 = time.perf_counter()
    step(x, y)
    t_cpu = time.perf_counter() - t0  # le CPU a fini de LANCER les noyaux
    synchronize(device)
    t_gpu = time.perf_counter() - t0  # le GPU a fini de les EXÉCUTER
    return t_cpu, t_gpu


def find_max_batch(args, device):
    """Double la taille du batch jusqu'à l'OOM, puis affine par dichotomie (pas de 8)."""
    args.data, args.compile = "synthetic", False

    def fits(bs):
        step = batch = None
        try:
            step = setup(args, device)
            batch, _ = batch_source(args, bs, device)
            for _ in range(3):
                step(*batch())
            synchronize(device)
            return True
        except torch.OutOfMemoryError:
            return False
        finally:
            del step, batch
            gc.collect()
            torch.cuda.empty_cache()

    lo, hi = 0, 16
    while fits(hi):
        lo, hi = hi, 2 * hi
    while hi - lo > 8:
        mid = (lo + hi) // 2 // 8 * 8
        lo, hi = (mid, hi) if fits(mid) else (lo, mid)
    return lo


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.async_demo:
        t_cpu, t_gpu = async_demo(args, device)
        print(f"[{args.tag}] retour de l'appel : {1e3 * t_cpu:.1f} ms ; "
              f"fin réelle du pas : {1e3 * t_gpu:.1f} ms")
        return
    if args.find_max_batch:
        if device.type != "cuda":
            raise SystemExit("--find-max-batch n'a de sens que sur GPU")
        bs = find_max_batch(args, device)
        print(f"[{args.tag}] batch maximal ({args.precision}) : {bs}")
        append_result({"tag": args.tag, "materiel": device_name(device),
                       "precision": args.precision, "channels_last": args.channels_last,
                       "batch_max": bs}, "results_maxbatch.csv")
        return
    row = run(args, device)
    print(" | ".join(f"{k}={v}" for k, v in row.items()), flush=True)
    append_result(row, "results_bench.csv")


if __name__ == "__main__":
    main()
