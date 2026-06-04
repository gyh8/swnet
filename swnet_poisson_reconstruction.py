"""Shared-Kernel Wavelet Network

Default data layout:
    ./data/BSDS500/train
    ./data/BSDS500/test

Typical usage:
    python swnet_poisson_reconstruction.py --mode train --gpu 0
    python swnet_poisson_reconstruction.py --mode test --checkpoint_path ./BSDS_results/5x5kernel.pth
"""

# ============================================================
# Imports
# ============================================================

import argparse
import glob
import os
import time
from typing import Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure


# ============================================================
# Model: PoissonNet
# ============================================================


class PoissonNet(nn.Module):
    """Shared-Kernel Wavelet Network

    H: analysis kernel.
    G: lateral correction kernel.
    K: synthesis kernel.

    The RGB channels are processed independently using groups=3.
    """

    def __init__(self, H_init: torch.Tensor, G_init: torch.Tensor, K_init: torch.Tensor):
        super().__init__()
        self.H = nn.Parameter(H_init.float())
        self.G = nn.Parameter(G_init.float())
        self.K = nn.Parameter(K_init.float())
        self._depth_cache = {}

    def _get_max_level(self, h: int, w: int) -> int:
        key = (int(h), int(w))
        if key not in self._depth_cache:
            self._depth_cache[key] = int(np.ceil(np.log2(max(h, w))))
        return self._depth_cache[key]

    def _k_synthesis_transpose(self, rd: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:

        fs = self.K.shape[2]
        pad = fs // 2
        target_h, target_w = target_hw
        in_h, in_w = rd.shape[-2:]

        base_h = (in_h - 1) * 2 - 2 * pad + fs
        base_w = (in_w - 1) * 2 - 2 * pad + fs
        out_pad_h = int(target_h - base_h)
        out_pad_w = int(target_w - base_w)

        if out_pad_h not in (0, 1) or out_pad_w not in (0, 1):
            # Exact fallback for unexpected shapes.
            up = torch.zeros((rd.shape[0], rd.shape[1], target_h, target_w), device=rd.device, dtype=rd.dtype)
            up[:, :, ::2, ::2] = rd
            return F.conv2d(up, self.K, padding=pad, groups=3)

        K_flip = torch.flip(self.K, dims=[2, 3])
        return F.conv_transpose2d(
            rd,
            K_flip,
            bias=None,
            stride=2,
            padding=pad,
            output_padding=(out_pad_h, out_pad_w),
            groups=3,
        )

    def forward(self, org_image: torch.Tensor, divG: torch.Tensor) -> torch.Tensor:
        _, _, h, w = divG.shape
        max_level = self._get_max_level(h, w)
        fs = self.H.shape[2]
        H_pad = self.H.shape[2] // 2
        G_pad = self.G.shape[2] // 2

        # Analysis pyramid.
        pyr = [None] * max_level
        pyr[0] = F.pad(-divG, (fs, fs, fs, fs), mode="constant", value=0)
        for i in range(1, max_level):
            down = F.conv2d(pyr[i - 1], self.H, padding=H_pad, groups=3)
            down = down[:, :, ::2, ::2]
            pyr[i] = F.pad(down, (fs, fs, fs, fs), mode="constant", value=0)

        # Coarse-to-fine synthesis pyramid.
        u = F.conv2d(pyr[max_level - 1], self.G, padding=G_pad, groups=3)
        for i in range(max_level - 2, -1, -1):
            rd = u[:, :, fs:-fs, fs:-fs]
            k_branch = self._k_synthesis_transpose(rd, target_hw=pyr[i].shape[-2:])
            g_branch = F.conv2d(pyr[i], self.G, padding=G_pad, groups=3)
            u = k_branch + g_branch

        ahat = u[:, :, fs:-fs, fs:-fs]

        # Mean correction.
        padded_image = F.pad(org_image, (1, 1, 1, 1), mode="constant", value=0)
        res = ahat - torch.mean(ahat, dim=(2, 3), keepdim=True) + torch.mean(padded_image, dim=(2, 3), keepdim=True)
        res = res[:, :, 1:-1, 1:-1]
        return res


# ============================================================
# Dataset and Laplacian utilities
# ============================================================


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageDataset(Dataset):
    """
    On-demand image folder dataset.
    """

    def __init__(self, folder_path: str):
        self.image_paths = sorted(
            p for p in glob.glob(os.path.join(folder_path, "*"))
            if os.path.splitext(p)[1].lower() in IMAGE_EXTENSIONS
        )
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {folder_path}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = self.image_paths[idx]
        with Image.open(img_path) as img:
            image = np.array(img.convert("RGB"), dtype=np.float32)
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous()
        return image


def build_laplacian_kernel(device: torch.device) -> torch.Tensor:
    lap_kernel = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        device=device,
        dtype=torch.float32,
    )
    lap_kernel = lap_kernel.unsqueeze(0).expand(3, -1, -1).unsqueeze(1).contiguous()
    return lap_kernel


def compute_laplacian_batch(image: torch.Tensor, lap_kernel: torch.Tensor) -> torch.Tensor:
    padded = F.pad(image, (1, 1, 1, 1), mode="constant", value=0)
    return F.conv2d(padded, lap_kernel, padding="same", groups=3)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


# ============================================================
# Kernel initialization and export
# ============================================================


def build_initial_kernels(device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build initial H/G/K kernels.
    """
    H_org = torch.tensor(
        [
            [0.0225, 0.0750, 0.1050, 0.0750, 0.0225],
            [0.0750, 0.2500, 0.3500, 0.2500, 0.0750],
            [0.1050, 0.3500, 0.4900, 0.3500, 0.1050],
            [0.0750, 0.2500, 0.3500, 0.2500, 0.0750],
            [0.0225, 0.0750, 0.1050, 0.0750, 0.0225],
        ],
        device=device,
        dtype=torch.float32,
    )
    H_org = H_org.unsqueeze(0).expand(3, -1, -1).unsqueeze(1).clone()

    G_org = torch.tensor(
        [
            [0.0306, 0.0957, 0.0306],
            [0.0957, 0.2992, 0.0957],
            [0.0306, 0.0957, 0.0306],
        ],
        device=device,
        dtype=torch.float32,
    )
    G_org = G_org.unsqueeze(0).expand(3, -1, -1).unsqueeze(1).clone()

    K_org = torch.tensor(
        [
            [-0.02, 0.09, 0.19, 0.08, -0.02],
            [0.09, 0.26, 0.33, 0.25, 0.09],
            [0.19, 0.33, 0.34, 0.33, 0.19],
            [0.08, 0.25, 0.33, 0.25, 0.09],
            [-0.02, 0.09, 0.19, 0.09, -0.02],
        ],
        device=device,
        dtype=torch.float32,
    )
    K_org = K_org.unsqueeze(0).expand(3, -1, -1).unsqueeze(1).clone()
    return H_org, G_org, K_org


def export_learned_kernels(model: PoissonNet, out_dir: str = "./BSDS_results/kernels") -> None:
    """Export learned H/G/K kernels to txt and npy files."""
    os.makedirs(out_dir, exist_ok=True)
    kernel_dict = {
        "H": model.H.detach().cpu().numpy(),
        "G": model.G.detach().cpu().numpy(),
        "K": model.K.detach().cpu().numpy(),
    }
    channel_names = ["R", "G", "B"]

    for kernel_name, kernel_value in kernel_dict.items():
        txt_path = os.path.join(out_dir, f"{kernel_name}.txt")
        npy_path = os.path.join(out_dir, f"{kernel_name}.npy")

        with open(txt_path, "w") as f:
            f.write(f"{kernel_name} kernel tensor shape: {kernel_value.shape}\n")
            for c in range(kernel_value.shape[0]):
                ch_name = channel_names[c] if c < len(channel_names) else f"C{c}"
                f.write(f"\n{kernel_name}_{ch_name}:\n")
                np.savetxt(f, kernel_value[c, 0], fmt="%.6f")

        np.save(npy_path, kernel_value)

    print(f"H/G/K kernels have been exported to: {out_dir}")


# ============================================================
# Metrics and timing utilities
# ============================================================


def sync_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.time()


def compute_psnr(output: torch.Tensor, target: torch.Tensor, max_val: float = 255.0) -> torch.Tensor:
    mse_per_image = torch.mean((output - target) ** 2, dim=[1, 2, 3])
    psnr_per_image = 10 * torch.log10(max_val ** 2 / (mse_per_image + 1e-10))
    return psnr_per_image.mean()


def get_device(gpu_id: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


# ============================================================
# Argument parser
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shared-Kernel Wavelet Network.")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test"], help="run mode")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id")
    parser.add_argument("--train_dir", type=str, default="./data/BSDS500/train", help="training image folder")
    parser.add_argument("--test_dir", type=str, default="./data/BSDS500/test", help="testing/evaluation image folder")
    parser.add_argument("--checkpoint_path", type=str, default="./output_results/5x5kernel.pth", help="checkpoint path")
    parser.add_argument("--epoch_loss_path", type=str, default="./output_results/5x5kernel_ep_loss.txt", help="epoch loss txt path")
    parser.add_argument("--kernel_out_dir", type=str, default="./output_results/kernels", help="output folder for learned kernels")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--epochs", type=int, default=2000, help="number of training epochs")
    parser.add_argument("--patience", type=int, default=100, help="early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--pin_memory", action="store_true", default=True, help="enable DataLoader pin_memory")
    parser.add_argument("--no_pin_memory", dest="pin_memory", action="store_false", help="disable DataLoader pin_memory")
    return parser.parse_args()


# ============================================================
# Train and test functions
# ============================================================


def train(args: argparse.Namespace, model: PoissonNet, device: torch.device) -> None:
    loss_fn = nn.MSELoss()
    lap_kernel = build_laplacian_kernel(device)

    test_dataset = ImageDataset(args.test_dir)
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    train_dataset = ImageDataset(args.train_dir)
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    train_epoch_losses = []
    test_epoch_losses = []
    best_test_loss = float("inf")
    early_stop_counter = 0

    for epoch in range(args.epochs):
        epoch_start_time = sync_time(device)
        model.train()
        total_samples = 0
        epoch_loss = 0.0

        for original_image in train_loader:
            batch_size = original_image.size(0)
            original_image = original_image.to(device, non_blocking=True)
            laplacian_image = compute_laplacian_batch(original_image, lap_kernel)

            optimizer.zero_grad()
            output = model(original_image, laplacian_image)
            loss = loss_fn(output, original_image)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_size
            total_samples += batch_size

        average_epoch_loss = epoch_loss / total_samples
        train_epoch_losses.append(average_epoch_loss)

        model.eval()
        total_test_samples = 0
        total_test_loss = 0.0
        with torch.no_grad():
            for original_image in test_loader:
                original_image = original_image.to(device, non_blocking=True)
                laplacian_image = compute_laplacian_batch(original_image, lap_kernel)
                output = model(original_image, laplacian_image)
                loss = loss_fn(output, original_image)

                total_test_loss += loss.item()
                total_test_samples += 1

        average_test_loss = total_test_loss / total_test_samples
        test_epoch_losses.append(average_test_loss)

        epoch_time = sync_time(device) - epoch_start_time
        print(
            f"Epoch [{epoch + 1}/{args.epochs}], "
            f"Train Loss: {average_epoch_loss:.5f}, "
            f"Test Loss: {average_test_loss:.5f}, "
            f"Time: {epoch_time:.2f}s"
        )

        if average_test_loss < best_test_loss:
            best_test_loss = average_test_loss
            early_stop_counter = 0
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                args.checkpoint_path,
            )
            print(f"Model saved to {args.checkpoint_path}")
        else:
            early_stop_counter += 1
            print(f"No improvement for {early_stop_counter} epochs. Best loss: {best_test_loss}")

        if early_stop_counter >= args.patience:
            print(f"Early stopping triggered. Best loss: {best_test_loss}.")
            break

    os.makedirs(os.path.dirname(args.epoch_loss_path), exist_ok=True)
    with open(args.epoch_loss_path, "w") as f:
        f.write(f"Best testing loss: {best_test_loss}\n")
        f.write(f"Training Data - Learning Rate: {args.lr}, Batch Size: {args.batch_size}\n")
        for loss in train_epoch_losses:
            f.write(f"{loss}\n")
        f.write("END_OF_TRAINING_DATA\n")
        f.write(f"Testing Data - Learning Rate: {args.lr}, Batch Size: {args.batch_size}\n")
        for loss in test_epoch_losses:
            f.write(f"{loss}\n")


def test(args: argparse.Namespace, model: PoissonNet, device: torch.device) -> None:
    loss_fn = nn.MSELoss()
    lap_kernel = build_laplacian_kernel(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=255.0, reduction="elementwise_mean").to(device)

    test_dataset = ImageDataset(args.test_dir)
    test_loader = make_loader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    export_learned_kernels(model, out_dir=args.kernel_out_dir)
    model.eval()

    total_test_mse = 0.0
    total_test_psnr = 0.0
    total_test_ssim = 0.0
    total_samples = 0

    with torch.no_grad():
        for original_image in test_loader:
            original_image = original_image.to(device, non_blocking=True)
            laplacian_image = compute_laplacian_batch(original_image, lap_kernel)

            output = model(original_image, laplacian_image)
            mse = loss_fn(output, original_image)
            psnr = compute_psnr(output, original_image).item()
            ssim = ssim_metric(output, original_image).item()
            batch_size = original_image.size(0)

            print(f"Test Loss: {mse:.5f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

            total_test_mse += mse.item() * batch_size
            total_test_psnr += psnr * batch_size
            total_test_ssim += ssim * batch_size
            total_samples += batch_size

    average_mse = total_test_mse / total_samples
    average_psnr = total_test_psnr / total_samples
    average_ssim = total_test_ssim / total_samples

    print(
        f"total_samples: {total_samples}, "
        f"Average Test Loss: {average_mse:.5f}, "
        f"Average PSNR: {average_psnr:.2f} dB, "
        f"Average SSIM: {average_ssim:.4f}"
    )


# ============================================================
# Main entry
# ============================================================


def main() -> None:
    args = parse_args()
    device = get_device(args.gpu)
    H, G, K = build_initial_kernels(device)
    model = PoissonNet(H, G, K).to(device)

    if args.mode == "train":
        train(args, model, device)
    elif args.mode == "test":
        test(args, model, device)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
