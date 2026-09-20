import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from aeon.datasets import load_classification

from mlp import MLP
from sech_kan import SechKAN
from efficient_kan import EfficientKAN
from cnn_1d import CNN1D_Crop
from resnet_1d import ResNet1D_Crop
from ds_cnn_1d import DSCNN1D_Crop
from schedulers import get_scheduler
from utils import set_seed, count_params

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(args):
    Xtr, ytr = load_classification("Crop", split="train")
    Xte, yte = load_classification("Crop", split="test")

    Xtr, Xte = Xtr[:, 0, :].astype(np.float32), Xte[:, 0, :].astype(np.float32)

    labels = sorted(np.unique(ytr.astype(str)))
    mapping = {v: i for i, v in enumerate(labels)}
    ytr = np.array([mapping[v] for v in ytr.astype(str)], dtype=np.int64)
    yte = np.array([mapping[v] for v in yte.astype(str)], dtype=np.int64)

    rng = np.random.default_rng(42) # seed = 42
    tr_idx, va_idx = [], []

    for c in np.unique(ytr):
        idx = np.flatnonzero(ytr == c)
        rng.shuffle(idx)
        n = max(1, int(np.ceil(len(idx) * 0.2)))
        va_idx.extend(idx[:n])
        tr_idx.extend(idx[n:])

    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)

    Xva, yva = Xtr[va_idx], ytr[va_idx]
    Xtr, ytr = Xtr[tr_idx], ytr[tr_idx]

    def loader(X, y, shuffle=False):
        return DataLoader(
            TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        )

    print(
        f"Crop | Train: {len(Xtr):,} | Val: {len(Xva):,} | "
        f"Test: {len(Xte):,} | Input: {Xtr.shape[1]:,} | Classes: {len(labels)}"
    )

    return (
        loader(Xtr, ytr, True),
        loader(Xva, yva),
        loader(Xte, yte),
        Xtr.shape[1],
        len(labels),
    )


def build_model(args, input_dim, num_classes):
    hidden = [int(x.strip()) for x in args.hidden_layers.split(",") if x.strip()]
    layers = [input_dim] + hidden + [num_classes]

    if args.model == "mlp":
        return MLP(layers, base_activation=args.activation, norm_type=args.norm_type)

    if args.model == "sech_kan":
        return SechKAN(
            layers, num_grids=args.num_grids, use_base_update=False,
            base_activation="silu", norm1_type=args.norm1_type,
            norm2_type=args.norm2_type, norm_mode=args.norm_mode,
            use_width=False, net_type="standard"
        )

    if args.model == "efficient_kan":
        return EfficientKAN(layers_hidden=layers, grid_size=5, spline_order=3)

    if args.model == "cnn_1d":
        return CNN1D_Crop(num_classes=num_classes)

    if args.model == "resnet_1d":
        return ResNet1D_Crop(num_classes=num_classes)

    if args.model == "ds_cnn_1d":
        return DSCNN1D_Crop(num_classes=num_classes)

    raise ValueError(args.model)


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, per_batch):
    model.train()
    loss_sum = correct = total = 0

    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x)
            loss = criterion(out, y)

        old_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if per_batch and scaler.get_scale() >= old_scale:
            scheduler.step()

        n = y.size(0)
        loss_sum += loss.item() * n
        correct += (out.argmax(1) == y).sum().item()
        total += n

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = correct = total = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x)
            loss = criterion(out, y)

        n = y.size(0)
        loss_sum += loss.item() * n
        correct += (out.argmax(1) == y).sum().item()
        total += n

    return loss_sum / total, correct / total


def train_model(model, train_loader, val_loader, test_loader, args):
    print(f"\nModel: {args.model}\nSeed: {args.seed}")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.scheduler == "StepLR":
        scheduler = get_scheduler(
            optimizer, name="StepLR", step_size=args.epochs // 3
        )
        per_batch = False

    elif args.scheduler == "CosineAnnealingLR":
        scheduler = get_scheduler(
            optimizer, name="CosineAnnealingLR", epochs=args.epochs
        )
        per_batch = False

    elif args.scheduler == "OneCycleLR":
        scheduler = get_scheduler(
            optimizer, name="OneCycleLR",
            step_size=len(train_loader) * args.epochs
        )
        per_batch = True

    elif args.scheduler == "ExponentialLR":
        scheduler = get_scheduler(optimizer, name="ExponentialLR")
        per_batch = False

    elif args.scheduler == "CyclicLR":
        scheduler = get_scheduler(
            optimizer, name="CyclicLR", step_size=len(train_loader) * 2
        )
        per_batch = True

    else:
        raise ValueError(args.scheduler)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    dataset = "crop"
    model_name = args.model.lower()
    save_dir = os.path.join("output", dataset, model_name)
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{model_name}__{args.note}.pt")
    history_path = os.path.join(save_dir, f"history__{args.note}.jsonl")
    summary_path = os.path.join(save_dir, f"summary__{args.note}.txt")

    open(history_path, "w").close()

    best_val, best_epoch = 0.0, 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()

    for epoch in range(args.epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, scheduler, per_batch
        )

        if not per_batch:
            scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch+1:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6g} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch + 1
            torch.save(model, model_path)

        with open(history_path, "a") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": optimizer.param_groups[0]["lr"],
                "time": epoch_time,
                "best_epoch": best_epoch,
            }) + "\n")

    params = count_params(model)

    model = torch.load(model_path, map_location=device, weights_only=False)
    model.to(device)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start

    print(
        f"\n{'-'*60}\n"
        f"Model      : {args.model}\n"
        f"Parameters : {params:,}\n"
        f"Scheduler  : {args.scheduler}\n"
        f"Best Val   : {best_val*100:.2f}% (epoch {best_epoch})\n"
        f"Test Acc   : {test_acc*100:.2f}%\n"
        f"Test Loss  : {test_loss:.4f}\n"
        f"Time       : {total_time:.1f}s\n"
        f"{'-'*60}"
    )

    with open(summary_path, "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Note: {args.note}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Scheduler: {args.scheduler}\n")
        f.write(f"LR: {args.lr}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Parameters: {params:,}\n")
        f.write(f"Train time: {total_time:.2f}s\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best Val Acc: {best_val:.6f}\n")
        f.write(f"Best Val Acc (%): {best_val*100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test Acc: {test_acc:.6f}\n")
        f.write(f"Test Acc (%): {test_acc*100:.2f}%\n")


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sech_kan",
                   choices=["mlp", "sech_kan", "efficient_kan",
                            "cnn_1d", "resnet_1d", "ds_cnn_1d"])
    p.add_argument("--note", default="")
    p.add_argument("--hidden_layers", default="256")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", default="OneCycleLR",
                   choices=["StepLR", "CosineAnnealingLR", "OneCycleLR",
                            "ExponentialLR", "CyclicLR"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--norm_type", default="layer")
    p.add_argument("--activation", default="silu")
    p.add_argument("--norm1_type", default="layer")
    p.add_argument("--norm2_type", default="")
    p.add_argument("--norm_mode", default="all",
                   choices=["none", "first", "except_first", "all"])
    p.add_argument("--num_grids", type=int, default=4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    global device
    args = get_args()
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    set_seed(args.seed, device)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader, val_loader, test_loader, input_dim, num_classes = \
        get_dataloaders(args)

    model = build_model(args, input_dim, num_classes)
    train_model(model, train_loader, val_loader, test_loader, args)


if __name__ == "__main__":
    main()

# 18804
# python run_crop.py --model "mlp" --hidden_layers "256" --batch_size 64 --epochs 20 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0"

# python run_crop.py --model "sech_kan" --num_grids 8 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0"

# python run_crop.py --model "cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0"

# python run_crop.py --model "resnet_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0"

# python run_crop.py --model "ds_cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0"

# python run_crop.py --model "efficient_kan" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0"