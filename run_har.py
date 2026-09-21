import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import json
import time
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from mlp import MLP
from schedulers import *
from utils import *

from cnn_1d import CNN1D_HAR
from resnet_1d import ResNet1D_HAR
from ds_cnn_1d import DSCNN1D_HAR
from sech_kan import SechKAN

from efficient_kan import EfficientKAN


UCI_URL = ("https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip")

DATASET_DIR = "UCI HAR Dataset"


def download_and_extract(data_root="./data"):
    os.makedirs(data_root, exist_ok=True)

    dataset_dir = os.path.join(data_root, DATASET_DIR)
    if os.path.isdir(os.path.join(dataset_dir, "train")) and os.path.isdir(os.path.join(dataset_dir, "test")):
        return dataset_dir

    zip_path = os.path.join(data_root, "UCI HAR Dataset.zip")

    if not os.path.isfile(zip_path):
        print("Downloading UCI HAR dataset...")
        urllib.request.urlretrieve(UCI_URL, zip_path)

    print("Extracting UCI HAR dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_root)

    if os.path.isdir(dataset_dir):
        os.remove(zip_path)
        return dataset_dir

    # Find the extracted directory containing train/ and test/
    for name in os.listdir(data_root):
        path = os.path.join(data_root, name)
        if os.path.isdir(path) and os.path.isdir(os.path.join(path, "train")) and os.path.isdir(os.path.join(path, "test")):
            os.remove(zip_path)
            return path

    raise FileNotFoundError(f"Could not find extracted UCI HAR dataset in {data_root}")


def load_txt(path, dtype=np.float32):
    return np.loadtxt(path, dtype=dtype)


def load_har(data_root="./data", val_subject_fraction=0.2):
    """
    UCI HAR:
        X_train: 7352 x 561
        X_test : 2947 x 561
        6 activity classes.

    The official train/test split is preserved.
    Validation is created from training subjects only, so no subject
    appears in both training and validation.
    """
    root = download_and_extract(data_root)

    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")

    X_train = load_txt(os.path.join(train_dir, "X_train.txt"))
    y_train = load_txt(os.path.join(train_dir, "y_train.txt"),dtype=np.int64,) - 1
    subject_train = load_txt(os.path.join(train_dir, "subject_train.txt"), dtype=np.int64,)

    X_test = load_txt(os.path.join(test_dir, "X_test.txt"))
    y_test = load_txt(os.path.join(test_dir, "y_test.txt"), dtype=np.int64,) - 1

    if X_train.shape[1] != 561:
        raise ValueError(f"Expected 561 input features, got {X_train.shape[1]}")

    if len(np.unique(y_train)) != 6 or len(np.unique(y_test)) != 6:
        raise ValueError("UCI HAR should contain exactly 6 activity classes.")

    # Subject-disjoint validation split.
    rng = np.random.default_rng(42) # seed = 42
    subjects = np.unique(subject_train)
    rng.shuffle(subjects)

    n_val_subjects = max(1, int(np.ceil(len(subjects) * val_subject_fraction)))
    val_subjects = set(subjects[:n_val_subjects])

    val_mask = np.array([s in val_subjects for s in subject_train], dtype=bool,)
    train_mask = ~val_mask

    X_val = X_train[val_mask]
    y_val = y_train[val_mask]

    X_train = X_train[train_mask]
    y_train = y_train[train_mask]

    print(
        f"UCI HAR | "
        f"Train: {len(X_train):,} | "
        f"Val: {len(X_val):,} | "
        f"Test: {len(X_test):,} | "
        f"Input: {X_train.shape[1]:,} | "
        f"Classes: 6"
    )
    print(
        f"Validation subjects: {sorted(val_subjects)} | "
        f"Train subjects: {len(np.unique(subject_train[train_mask]))}"
    )

    # UCI HAR features are already preprocessed/normalized by the dataset.
    # Do not fit another normalization using validation/test data.
    train_x = torch.from_numpy(X_train)
    train_y = torch.from_numpy(y_train).long()

    val_x = torch.from_numpy(X_val)
    val_y = torch.from_numpy(y_val).long()

    test_x = torch.from_numpy(X_test)
    test_y = torch.from_numpy(y_test).long()

    return (train_x, train_y, val_x, val_y, test_x, test_y, X_train.shape[1], 6,)


def get_dataloaders(
    data_root="./data",
    batch_size=64,
    num_workers=4,
    val_subject_fraction=0.2,
    seed=42,
):
    (
        train_x,
        train_y,
        val_x,
        val_y,
        test_x,
        test_y,
        input_dim,
        num_classes,
    ) = load_har(
        data_root=data_root,
        val_subject_fraction=val_subject_fraction,
    )

    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    test_dataset = TensorDataset(test_x, test_y)

    def loader(dataset, shuffle=False):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return (loader(train_dataset, True), loader(val_dataset), loader(test_dataset), input_dim, num_classes,)


def build_model(args, input_dim, num_classes):
    hidden_layers = [
        int(x.strip())
        for x in args.hidden_layers.split(",")
        if x.strip()
    ]

    net_layers = [input_dim] + hidden_layers + [num_classes]


    if args.model == "mlp":
        return MLP(
            net_layers=net_layers,
            base_activation=args.activation,
            norm_type=args.norm_type,
        )

    
    if args.model == "sech_kan":
        return SechKAN(net_layers=net_layers, num_grids=args.num_grids, use_base_update = False, base_activation = "silu", norm1_type = args.norm1_type, norm2_type = args.norm2_type, norm_mode = args.norm_mode, use_width = False, net_type = 'standard')
        
    if args.model == "cnn_1d":
        return CNN1D_HAR(num_classes=num_classes)

    if args.model == "resnet_1d":
        return ResNet1D_HAR(num_classes=num_classes)

    if args.model == "ds_cnn_1d":
        return DSCNN1D_HAR(num_classes=num_classes)
    
    if (args.model == "efficient_kan"):
        return EfficientKAN(layers_hidden=net_layers, grid_size=5, spline_order=3)
   

    raise ValueError(f"Unknown model: {args.model}")


def accuracy_from_logits(outputs, targets):
    return (outputs.argmax(dim=1) == targets).sum().item()


def train_one_epoch(model, loader, criterion, optimizer, scaler, scheduler, scheduler_per_batch,):
    model.train()

    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, desc="Training"):
        
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            out = model(x)
            loss = criterion(out, y)

        old_scale = scaler.get_scale()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        new_scale = scaler.get_scale()

        if scheduler_per_batch and new_scale >= old_scale:
            scheduler.step()

        n = y.size(0)
        loss_sum += loss.item() * n
        correct += accuracy_from_logits(out, y)
        total += n

    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            out = model(x)
            loss = criterion(out, y)

        n = y.size(0)
        loss_sum += loss.item() * n
        correct += accuracy_from_logits(out, y)
        total += n

    return loss_sum / total, correct / total


def train_model(model, train_loader, val_loader, test_loader, args):
    print(f"\nModel: {args.model}")
    print(f"\nSeed: {args.seed}")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay,)

    # Define learning rate scheduler
    if(args.scheduler == 'StepLR'):
        scheduler = get_scheduler(optimizer, name="StepLR", step_size = args.epochs//3)
        scheduler_per_batch = False
    elif(args.scheduler == 'CosineAnnealingLR'):
        scheduler = get_scheduler(optimizer, name="CosineAnnealingLR", epochs = args.epochs)
        scheduler_per_batch = False
    elif(args.scheduler == 'OneCycleLR'):
        scheduler = get_scheduler(optimizer, name="OneCycleLR", step_size=len(train_loader)*args.epochs)
        scheduler_per_batch = True
    elif(args.scheduler == 'ExponentialLR'):
        scheduler = get_scheduler(optimizer, name="ExponentialLR")
        scheduler_per_batch = False
    elif(args.scheduler == 'CyclicLR'):
        scheduler = get_scheduler(optimizer, name="CyclicLR", step_size=len(train_loader)*2)
        scheduler_per_batch = True
    else:
        print('You should choose a scheduler (StepLR, CosineAnnealingLR, OneCycleLR, ExponentialLR).')
        return

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda",)

    dataset_name = "uci_har"
    model_name = args.model.lower()
    note = args.note

    save_dir = os.path.join("output", dataset_name, model_name)
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{model_name}__{note}.pt",)
    history_path = os.path.join(save_dir, f"history__{note}.jsonl",)
    summary_path = os.path.join(save_dir, f"summary__{note}.txt",)

    open(history_path, "w").close()

    best_val_acc = 0.0
    best_epoch = 0
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    for epoch in range(args.epochs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_start = time.perf_counter()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, scheduler, scheduler_per_batch)

        if not scheduler_per_batch:
            scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion,)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start

        print(
            f"Epoch {epoch + 1:02d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f}, "
            f"Train Acc: {train_acc * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc * 100:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6g} | "
            f"Time: {epoch_time:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model, model_path)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "time": epoch_time,
            "best_epoch": best_epoch,
        }

        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    params = count_params(model)

    model = torch.load(model_path, map_location=device, weights_only=False,)
    model.to(device)

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
    )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time

    print(
        f"\n{'-' * 60}\n"
        f"Model      : {args.model}\n"
        f"Parameters : {params:,}\n"
        f"Scheduler  : {args.scheduler}\n"
        f"Best Val   : {best_val_acc * 100:.2f}% "
        f"(epoch {best_epoch})\n"
        f"Test Acc   : {test_acc * 100:.2f}%\n"
        f"Test Loss  : {test_loss:.4f}\n"
        f"Time       : {total_time:.1f}s\n"
        f"{'-' * 60}"
    )

    with open(summary_path, "w") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Note: {note}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Scheduler: {args.scheduler}\n")
        f.write(f"LR: {args.lr}\n")
        f.write(f"Weight decay: {args.weight_decay}\n")
        f.write(f"Parameters: {params:,}\n")
        f.write(f"Train time: {total_time:.2f}s\n")
        f.write(f"Best epoch: {best_epoch}\n")
        f.write(f"Best Val Acc: {best_val_acc:.6f}\n")
        f.write(f"Best Val Acc (%): {best_val_acc * 100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test Acc: {test_acc:.6f}\n")
        f.write(f"Test Acc (%): {test_acc * 100:.2f}%\n")


def get_args():
    parser = argparse.ArgumentParser(description="Train models on UCI HAR")

    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn_1d", "resnet_1d", "ds_cnn_1d", "sech_kan", "efficient_kan"])
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument("--val_subject_fraction", type=float, default=0.2)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden_layers", type=str, default="256")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--scheduler", type=str, default="OneCycleLR", choices=["StepLR", "CosineAnnealingLR", "OneCycleLR", "ExponentialLR", "CyclicLR"])
    #parser.add_argument("--gamma", type=float, default=0.95)

    parser.add_argument("--ratio", type=int, default=2)
    parser.add_argument("--pairwise_fn", type=str, default="atan")
    parser.add_argument("--pairing", type=str, default="original", choices=["original", "shuffle", "reverse"])
    parser.add_argument("--pair_type", type=str, default="raw_weighted_sum_product")
    parser.add_argument("--norm1_type", type=str, default="")
    parser.add_argument("--norm2_type", type=str, default="layer")
    parser.add_argument("--norm_mode", type=str, default="all", choices=["none", "first", "except_first", "all"])

    parser.add_argument("--norm_type", type=str, default="layer")
    parser.add_argument("--activation", type=str, default="silu")

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    
    #SechKAN
    parser.add_argument("--num_grids", type=int, default=4)

    return parser.parse_args()


def main():
    global device
    args = get_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    set_seed(args.seed, device)

    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    train_loader, val_loader, test_loader, input_dim, num_classes = get_dataloaders(data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers, val_subject_fraction=args.val_subject_fraction, seed=args.seed)
    model = build_model(args, input_dim=input_dim, num_classes=num_classes)
    train_model(model, train_loader, val_loader, test_loader, args)

if __name__ == "__main__":
    main()


#147,048
# python run_har.py --model "mlp" --hidden_layers "256" --batch_size 64 --epochs 20 --norm_type "layer" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0111"
  
# python run_har.py --model "sech_kan" --num_grids 4 --hidden_layers "256" --batch_size 64 --epochs 20 --norm1_type "layer" --norm2_type "" --norm_mode "all" --activation "silu" --scheduler "OneCycleLR" --seed 0 --note "run_0"

# python run_har.py --model "cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0"

# python run_har.py --model "resnet_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0"

# python run_har.py --model "ds_cnn_1d" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0"

# python run_har.py --model "efficient_kan" --batch_size 64 --epochs 20 --scheduler "OneCycleLR" --seed 0 --note "run_0"
