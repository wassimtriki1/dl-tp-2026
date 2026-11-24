"""TP6 : profilage de quelques itérations d'entraînement avec de vraies données."""
import argparse
import os

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, schedule

from common import (build_model, make_dataset, make_loader, make_train_step,
                    seed_everything, set_fp32_mode)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--data", default="raw", choices=["raw", "small"])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--precision", default="fp32", choices=["fp32", "bf16"])
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(0)
    set_fp32_mode(tf32=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.arch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    amp = torch.bfloat16 if args.precision == "bf16" else None
    step = make_train_step(model, opt, nn.CrossEntropyLoss(), amp)
    loader = make_loader(make_dataset(args.data, "train"), args.batch_size, args.workers)

    os.makedirs("traces", exist_ok=True)
    sort_key = "cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"

    def on_ready(p):
        print(p.key_averages().table(sort_by=sort_key, row_limit=12), flush=True)
        p.export_chrome_trace(f"traces/{args.tag}.json")  # à ouvrir dans ui.perfetto.dev

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities,
                 schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
                 on_trace_ready=on_ready, record_shapes=True) as prof:
        for i, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            step(x, y)
            prof.step()  # signale au profiler la fin d'une itération
            if i + 1 == 8:
                break


if __name__ == "__main__":
    main()
