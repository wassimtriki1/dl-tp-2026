"""TP6 : débit d'entraînement avec DistributedDataParallel (un processus par GPU)."""
import argparse
import contextlib
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from common import (append_result, benchmark, build_model, device_name, make_dataset,
                    make_loader, median, seed_everything, set_fp32_mode)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="")
    p.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--batch-size", type=int, default=64, help="taille du batch PAR processus")
    p.add_argument("--precision", default="bf16", choices=["fp32", "bf16"])
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--data", default="synthetic", choices=["synthetic", "small"])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--bf16-hook", action="store_true", help="compresser les gradients en bf16")
    p.add_argument("--accum", type=int, default=1, help="micro-batchs par pas d'optimiseur")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def setup_distributed(device_type):
    """Initialise le groupe de processus à partir des variables fixées par torchrun."""
    local_rank = int(os.environ["LOCAL_RANK"])
    if device_type == "cuda":
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    backend = "nccl" if device_type == "cuda" else "gloo"
    # lit RANK, WORLD_SIZE, MASTER_ADDR et MASTER_PORT ; device_id lie le processus à son GPU
    dist.init_process_group(backend=backend, device_id=device if device_type == "cuda" else None)
    return device


def make_ddp_step(model, opt, criterion, amp_dtype, accum):
    """Un pas d'optimiseur sur `accum` micro-batchs ; seul le dernier backward communique."""
    def step(next_batch):
        for i in range(accum):
            last = i == accum - 1
            ctx = contextlib.nullcontext() if last else model.no_sync()
            with ctx:  # sous no_sync, DDP accumule les gradients localement, sans all-reduce
                x, y = next_batch()
                with torch.autocast(device_type=x.device.type, dtype=amp_dtype,
                                    enabled=amp_dtype is not None):
                    loss = criterion(model(x), y) / accum
                loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return step


def main():
    args = parse_args()
    device = setup_distributed(args.device)
    rank, world = dist.get_rank(), dist.get_world_size()
    seed_everything(args.seed)  # même initialisation partout (DDP diffuse de toute façon le rang 0)
    set_fp32_mode(tf32=True)
    mf = torch.channels_last if args.channels_last else torch.contiguous_format

    model = build_model(args.arch).to(device, memory_format=mf)
    model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    if args.bf16_hook:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        model.register_comm_hook(state=None, hook=default_hooks.bf16_compress_hook)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    amp = torch.bfloat16 if args.precision == "bf16" else None
    step = make_ddp_step(model, opt, nn.CrossEntropyLoss(), amp, args.accum)

    if args.data == "synthetic":  # données différentes sur chaque rang, déjà sur le device
        g = torch.Generator(device=device).manual_seed(args.seed + rank)
        x = torch.randn(args.batch_size, 3, args.image_size, args.image_size,
                        device=device, generator=g).contiguous(memory_format=mf)
        y = torch.randint(0, 10, (args.batch_size,), device=device, generator=g)

        def next_batch():
            return x, y
    else:
        train_set = make_dataset("small", "train")
        sampler = DistributedSampler(train_set, shuffle=True, seed=args.seed)
        loader = make_loader(train_set, args.batch_size, args.workers, sampler=sampler)

        def forever():
            epoch = 0
            while True:
                sampler.set_epoch(epoch)  # sinon : même ordre à chaque époque
                yield from loader
                epoch += 1

        it = forever()

        def next_batch():
            xb, yb = next(it)
            return (xb.to(device, non_blocking=True, memory_format=mf),
                    yb.to(device, non_blocking=True))

    dist.barrier()
    medians = []
    for r in range(args.repeat):
        times = benchmark(lambda: step(next_batch), device,
                          n_warmup=args.warmup if r == 0 else 2, n_iter=args.steps)
        medians.append(median(times))
    t = torch.tensor([median(medians)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)  # le processus le plus lent impose le rythme
    step_s = t.item()
    img_s = world * args.batch_size * args.accum / step_s  # débit global, tous processus confondus

    if rank == 0:
        row = {"tag": args.tag, "materiel": device_name(device), "processus": world,
               "noeuds": world // int(os.environ.get("LOCAL_WORLD_SIZE", 1)), "backend": dist.get_backend(),
               "precision": args.precision, "bf16_hook": args.bf16_hook, "donnees": args.data,
               "accumulation": args.accum, "batch_par_processus": args.batch_size,
               "batch_effectif": world * args.batch_size * args.accum,
               "pas_ms": round(1e3 * step_s, 1), "img_s": round(img_s, 1)}
        print(" | ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        append_result(row, "results_ddp.csv")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
