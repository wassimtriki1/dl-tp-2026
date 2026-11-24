"""TP6 : fine-tuning court d'un ResNet-50 pré-entraîné, avec la configuration de votre choix."""
import argparse
import time

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from common import (append_result, build_model, device_name, make_dataset, make_loader,
                    make_train_step, seed_everything, set_fp32_mode, synchronize)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="best")
    p.add_argument("--arch", default="resnet50", choices=["resnet50", "resnet18"])
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--precision", default="bf16", choices=["fp32", "tf32", "bf16"])
    p.add_argument("--channels-last", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--data", default="small", choices=["raw", "small"])
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--out", default="best.pth")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, mf):
    model.eval()  # BatchNorm : statistiques courantes, pas celles du batch
    correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True, memory_format=mf)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None):
            pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_fp32_mode(tf32=(args.precision != "fp32"))
    mf = torch.channels_last if args.channels_last else torch.contiguous_format
    amp = torch.bfloat16 if args.precision == "bf16" else None

    model = build_model(args.arch, pretrained=not args.no_pretrained).to(device, memory_format=mf)
    fwd = torch.compile(model) if args.compile else model
    train_loader = make_loader(make_dataset(args.data, "train"), args.batch_size,
                               args.workers, seed=args.seed)
    val_loader = make_loader(make_dataset(args.data, "val"), 2 * args.batch_size,
                             args.workers, train=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_loader))
    step = make_train_step(fwd, opt, nn.CrossEntropyLoss(), amp)
    writer = SummaryWriter(f"runs/train-{args.tag}")

    best_acc, it = 0.0, 0
    for epoch in range(args.epochs):
        model.train()
        synchronize(device)
        t0 = time.perf_counter()
        for x, y in train_loader:
            loss = step(x.to(device, non_blocking=True, memory_format=mf),
                        y.to(device, non_blocking=True))
            sched.step()
            if it % 20 == 0:  # .item() force une synchronisation : on l'évite à chaque pas
                writer.add_scalar("Loss/Train", loss.item(), it)
            it += 1
        synchronize(device)
        epoch_s = time.perf_counter() - t0
        img_s = len(train_loader) * args.batch_size / epoch_s
        acc = evaluate(model, val_loader, device, amp, mf)
        writer.add_scalar("Accuracy/Validation", acc, epoch)
        writer.add_scalar("Speed/images_per_s", img_s, epoch)
        print(f"époque {epoch}: {epoch_s:.1f} s ({img_s:.0f} img/s), top-1 val = {acc:.3f}",
              flush=True)
        append_result({"tag": args.tag, "materiel": device_name(device), "epoque": epoch,
                       "precision": args.precision, "channels_last": args.channels_last,
                       "compile": args.compile, "donnees": args.data, "workers": args.workers,
                       "epoque_s": round(epoch_s, 1), "img_s": round(img_s, 1),
                       "top1_val": round(acc, 4)}, "results_train.csv")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.out)  # surtout pas fwd : clés « _orig_mod. »
    writer.close()
    print(f"Meilleur top-1 de validation : {best_acc:.3f} -> {args.out}")


if __name__ == "__main__":
    main()
