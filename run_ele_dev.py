import os, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from aeon.datasets import load_classification

from mlp import MLP
from sech_kan import SechKAN
from efficient_kan import EfficientKAN
from cnn_1d import CNN1D_Ele_Dev
from resnet_1d import ResNet1D_Ele_Dev
from ds_cnn_1d import DSCNN1D_Ele_Dev
from schedulers import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_dataloaders(args):
    Xtr, ytr = load_classification("ElectricDevices", split="train")
    Xte, yte = load_classification("ElectricDevices", split="test")
    Xtr, Xte = Xtr[:, 0, :].astype(np.float32), Xte[:, 0, :].astype(np.float32)

    labels = {v: i for i, v in enumerate(sorted(np.unique(ytr.astype(str))))}
    ytr = np.array([labels[v] for v in ytr.astype(str)], dtype=np.int64)
    yte = np.array([labels[v] for v in yte.astype(str)], dtype=np.int64)

    rng = np.random.default_rng(42) # seed = 42
    tr_idx, va_idx = [], []

    for c in np.unique(ytr):
        idx = np.flatnonzero(ytr == c)
        rng.shuffle(idx)
        n = max(1, round(len(idx) * .2))
        va_idx += idx[:n].tolist()
        tr_idx += idx[n:].tolist()

    rng.shuffle(tr_idx)
    rng.shuffle(va_idx)

    Xva, yva = Xtr[va_idx], ytr[va_idx]
    Xtr, ytr = Xtr[tr_idx], ytr[tr_idx]

    print(f"Electric Devices | Train: {len(Xtr):,} | Val: {len(Xva):,} | Test: {len(Xte):,} | Input: {Xtr.shape[1]:,} | Classes: {len(labels)}")
    print(f"Original Train: {len(Xtr) + len(Xva):,} | Validation split: {len(Xva)/(len(Xtr)+len(Xva))*100:.1f}%")

    def loader(X, y, shuffle=False):
        return DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)),
                          batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=args.num_workers, pin_memory=True)

    return loader(Xtr, ytr, True), loader(Xva, yva), loader(Xte, yte), Xtr.shape[1], len(labels)


def build_model(args, input_dim, num_classes):
    layers = [input_dim] + [int(x) for x in args.hidden_layers.split(",")] + [num_classes]

    if args.model == "mlp":
        return MLP(net_layers=layers, base_activation=args.activation, norm_type=args.norm_type)

    if args.model == "sech_kan":
        return SechKAN(net_layers=layers, num_grids=args.num_grids, use_base_update=False,
                       base_activation="silu", norm1_type=args.norm1_type,
                       norm2_type=args.norm2_type, norm_mode=args.norm_mode,
                       use_width=True, net_type="standard")

    if args.model == "efficient_kan":
        return EfficientKAN(layers_hidden=layers, grid_size=5, spline_order=3)

    if args.model == "cnn_1d":
        return CNN1D_Ele_Dev(num_classes)

    if args.model == "resnet_1d":
        return ResNet1D_Ele_Dev(num_classes)

    if args.model == "ds_cnn_1d":
        return DSCNN1D_Ele_Dev(num_classes)

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


def train(model, train_loader, val_loader, test_loader, criterion, args, input_dim, num_classes):
    print(f"\nModel: {args.model}")
    print(f"Seed: {args.seed}")

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "StepLR":
        scheduler = get_scheduler(optimizer, name="StepLR", step_size=max(1, args.epochs // 3))
        per_batch = False
    elif args.scheduler == "CosineAnnealingLR":
        scheduler = get_scheduler(optimizer, name="CosineAnnealingLR", epochs=args.epochs)
        per_batch = False
    elif args.scheduler == "OneCycleLR":
        scheduler = get_scheduler(optimizer, name="OneCycleLR", step_size=len(train_loader) * args.epochs)
        per_batch = True
    elif args.scheduler == "ExponentialLR":
        scheduler = get_scheduler(optimizer, name="ExponentialLR")
        per_batch = False
    elif args.scheduler == "CyclicLR":
        scheduler = get_scheduler(optimizer, name="CyclicLR", step_size=len(train_loader) * 2)
        per_batch = True
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler}")

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    # Save outputs
    dataset_name = "electric_devices"
    model_name = args.model.lower()
    save_dir = os.path.join("output", dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{model_name}__{args.note}.pt")
    history_path = os.path.join(save_dir, f"history__{args.note}.jsonl")
    summary_path = os.path.join(save_dir, f"summary__{args.note}.txt")
    open(history_path, "w").close()

    best_val_acc, best_epoch = 0.0, 0

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, per_batch)

        if not per_batch:
            scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6g} | Time: {epoch_time:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch + 1
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
                "best_epoch": best_epoch
            }) + "\n")

    params = count_params(model)

    # Test best model
    model = torch.load(model_path, map_location=device, weights_only=False).to(device)
    test_loss, test_acc = evaluate(model, test_loader, criterion)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    print(f"\n{'-'*60}\nModel      : {args.model}\nSeed       : {args.seed}\nParameters : {params:,}\nScheduler  : {args.scheduler}\nBest Val   : {best_val_acc*100:.2f}% (epoch {best_epoch})\nTest Acc   : {test_acc*100:.2f}%\nTest Loss  : {test_loss:.4f}\nTime       : {total_time:.1f}s\n{'-'*60}")

    with open(summary_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Input dim: {input_dim}\n")
        f.write(f"Classes: {num_classes}\n")
        f.write(f"Parameters: {params:,}\n")
        f.write(f"Train time: {total_time:.2f}s\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best Val Acc: {best_val_acc:.6f}\n")
        f.write(f"Best Val Acc (%): {best_val_acc*100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test Acc: {test_acc:.6f}\n")
        f.write(f"Test Acc (%): {test_acc*100:.2f}%\n")

    return model, best_val_acc, best_epoch, test_loss, test_acc


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="sech_kan",
                   choices=["mlp", "sech_kan", "cnn_1d", "resnet_1d", "ds_cnn_1d", "efficient_kan"])
    p.add_argument("--hidden_layers", default="256")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--scheduler", default="OneCycleLR")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ratio", type=int, default=2)
    p.add_argument("--pairwise_fn", default="atan")
    p.add_argument("--pairing", default="original")
    p.add_argument("--pair_type", default="weighted_sum_product")
    p.add_argument("--norm1_type", default="layer")
    p.add_argument("--norm2_type", default="")
    p.add_argument("--norm_mode", default="all")
    p.add_argument("--norm_type", default="layer")
    p.add_argument("--activation", default="silu")
    p.add_argument("--num_grids", type=int, default=4)
    p.add_argument("--note", default="run_0")
    return p.parse_args()


def main():
    args = get_args()
    set_seed(args.seed, device)

    train_loader, val_loader, test_loader, input_dim, num_classes = get_dataloaders(args)

    print(f"Dataset: Electric Devices | Input: ({input_dim},) | Classes: {num_classes} | Device: {device}")

    model = build_model(args, input_dim, num_classes)
    train(model, train_loader, val_loader, test_loader, nn.CrossEntropyLoss(), args, input_dim, num_classes)


if __name__ == "__main__":
    main()
    
#python run_ele_dev.py --model "mlp" --hidden_layers "256" --batch_size 64 --epochs 30 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0";

#python run_ele_dev.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 30 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0";

#python run_ele_dev.py --model "cnn_1d" --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0";

#python run_ele_dev.py --model "resnet_1d" --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0";

#python run_ele_dev.py --model "ds_cnn_1d" --batch_size 64 --epochs 30 --scheduler "OneCycleLR" --seed 0 --note "run_0";

#python run_ele_dev.py --model "efficient_kan" --batch_size 64 --hidden_layers "26" --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0";