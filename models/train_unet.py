"""
TinyUNet training script for PBF neural post-processing.

Expects a dataset directory with two sub-folders:
  input/   — RGBA PNG frames (4-channel) captured from the sim at runtime
  target/  — RGB  PNG frames (3-channel) used as ground truth

Filenames must match across the two folders (e.g. 0000.png / 0000.png).

Typical ground-truth strategies:
  - Temporal supersampling: freeze sim, accumulate N identical frames, average → clean target
  - Synthetic noise:        add Gaussian noise to clean frames; train to denoise
  - Offline render:         higher-quality offline render as target, real-time as input

Usage:
    pip install torch torchvision pillow
    python models/train_unet.py --data path/to/dataset
    python models/train_unet.py --data path/to/dataset --resume unet_trained.pth
    python models/train_unet.py --data path/to/dataset --epochs 200 --batch 4 --lr 3e-4
"""

import argparse
import os
import glob
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode

from export_unet import TinyUNet, export


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FramePairDataset(Dataset):
    """
    Loads matched (input RGBA, target RGB) PNG pairs from:
      root/input/*.png   — 4-channel RGBA, values [0, 255]
      root/target/*.png  — 3-channel RGB,  values [0, 255]
    """
    def __init__(self, root: str):
        input_dir  = os.path.join(root, "input")
        target_dir = os.path.join(root, "target")

        self.inputs  = sorted(glob.glob(os.path.join(input_dir,  "*.png")))
        self.targets = sorted(glob.glob(os.path.join(target_dir, "*.png")))

        if len(self.inputs) == 0:
            raise FileNotFoundError(f"No PNG files found in {input_dir}")
        if len(self.inputs) != len(self.targets):
            raise ValueError(
                f"Input/target count mismatch: {len(self.inputs)} vs {len(self.targets)}"
            )

        # Verify filenames match
        for i, t in zip(self.inputs, self.targets):
            if os.path.basename(i) != os.path.basename(t):
                raise ValueError(
                    f"Filename mismatch: {os.path.basename(i)} vs {os.path.basename(t)}"
                )

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # read_image returns uint8 [C, H, W]; divide to get float32 in [0, 1]
        inp = read_image(self.inputs[idx],  ImageReadMode.RGBA).float() / 255.0
        tgt = read_image(self.targets[idx], ImageReadMode.RGB ).float() / 255.0
        return inp, tgt


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset split: 90% train, 10% validation
    full_dataset = FramePairDataset(args.data)
    n_val   = max(1, int(0.1 * len(full_dataset)))
    n_train = len(full_dataset) - n_val
    train_set, val_set = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Dataset: {len(full_dataset)} pairs  —  {n_train} train / {n_val} val")

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=1,          shuffle=False,
                              num_workers=1, pin_memory=True)

    # Model
    model = TinyUNet().to(device)
    if args.resume and os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"TinyUNet  —  {param_count:,} parameters")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Halve LR if val loss stalls for 10 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, verbose=True
    )

    loss_fn = nn.L1Loss()   # L1 produces sharper results than MSE

    best_val_loss = float("inf")
    checkpoint    = os.path.join(args.out_dir, "unet_best.pth")
    os.makedirs(args.out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_loss = 0.0
        t0 = time.time()
        for inp, tgt in train_loader:
            inp, tgt = inp.to(device), tgt.to(device)
            pred = model(inp)
            loss = loss_fn(pred, tgt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inp.size(0)
        train_loss /= n_train

        # ---- Validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp, tgt in val_loader:
                inp, tgt = inp.to(device), tgt.to(device)
                pred = model(inp)
                val_loss += loss_fn(pred, tgt).item() * inp.size(0)
        val_loss /= n_val

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:4d}/{args.epochs}  "
              f"train {train_loss:.4f}  val {val_loss:.4f}  "
              f"lr {optimizer.param_groups[0]['lr']:.2e}  {elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint)
            print(f"  → saved best checkpoint ({best_val_loss:.4f})")

    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: {checkpoint}")

    # Export the best weights to ONNX so the app can use them immediately
    if args.export_onnx:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        # Infer resolution from the first sample
        sample_inp, _ = full_dataset[0]
        _, H, W = sample_inp.shape
        onnx_path = os.path.join(args.out_dir, "unet_postprocess.onnx")
        export(onnx_path, height=H, width=W)
        print(f"ONNX model exported to {onnx_path}")
        print(f"Copy to Bin\\models\\ to use in the app.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    default_out = os.path.join(os.path.dirname(__file__), "..", "Bin", "models")

    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True,
                        help="Dataset root with input/ and target/ sub-folders")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch",       type=int,   default=2)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--resume",      default="",
                        help="Path to a .pth checkpoint to continue training from")
    parser.add_argument("--out-dir",     default=os.path.normpath(default_out),
                        help="Directory for checkpoints and exported ONNX model")
    parser.add_argument("--export-onnx", action="store_true",
                        help="Export the best checkpoint to ONNX after training")
    args = parser.parse_args()

    train(args)
