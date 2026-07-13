"""
Lightweight SimCLR-style self-supervised pretraining for chart images.

This module is intentionally compact to keep the end-to-end pipeline runnable
on CPU while still providing a valid contrastive pretraining stage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class SimCLRConfig:
    image_size: int = 128
    batch_size: int = 32
    epochs: int = 10
    lr: float = 1e-4
    max_samples: int = 2000
    max_steps_per_epoch: int = 80
    temperature: float = 0.2


class SimCLRPretrainer:
    """Run small-scale contrastive pretraining and save backbone weights."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, image_paths: List[str], cfg: SimCLRConfig | None = None) -> Dict:
        cfg = cfg or SimCLRConfig()

        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            import torch.optim as optim
            from PIL import Image
            from torch.utils.data import DataLoader, Dataset
            from torchvision import transforms
        except Exception as exc:
            return {"trained": False, "reason": f"torch/torchvision unavailable: {exc}"}

        if not image_paths:
            return {"trained": False, "reason": "no images available for pretraining"}

        image_paths = image_paths[: cfg.max_samples]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class PairDataset(Dataset):
            def __init__(self, paths: List[str], transform):
                self.paths = paths
                self.transform = transform

            def __len__(self):
                return len(self.paths)

            def __getitem__(self, idx):
                img = Image.open(self.paths[idx]).convert("RGB")
                return self.transform(img), self.transform(img)

        aug = transforms.Compose(
            [
                transforms.Resize((cfg.image_size, cfg.image_size)),
                transforms.RandomResizedCrop(cfg.image_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
                transforms.ToTensor(),
            ]
        )

        loader = DataLoader(
            PairDataset(image_paths, aug),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        if len(loader) == 0:
            return {"trained": False, "reason": "insufficient images for configured batch size"}

        class TinyEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 32, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1)),
                )
                self.proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(inplace=True), nn.Linear(128, 64))

            def forward(self, x):
                z = self.backbone(x).flatten(1)
                p = self.proj(z)
                return F.normalize(p, dim=1)

        model = TinyEncoder().to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

        def nt_xent(z1, z2, temperature: float):
            # SimCLR NT-Xent loss over positive pairs (z1_i, z2_i).
            z = torch.cat([z1, z2], dim=0)
            sim = torch.matmul(z, z.t()) / temperature
            n = z.size(0)
            mask = torch.eye(n, device=sim.device, dtype=torch.bool)
            sim = sim.masked_fill(mask, -9e15)

            positives = torch.cat(
                [
                    torch.arange(z1.size(0), n, device=sim.device),
                    torch.arange(0, z1.size(0), device=sim.device),
                ]
            )
            return F.cross_entropy(sim, positives)

        loss_history = []
        model.train()
        for epoch in range(cfg.epochs):
            epoch_loss = 0.0
            step_count = 0
            for x1, x2 in loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                optimizer.zero_grad()
                z1 = model(x1)
                z2 = model(x2)
                loss = nt_xent(z1, z2, cfg.temperature)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                step_count += 1
                if step_count >= cfg.max_steps_per_epoch:
                    break

            avg_loss = epoch_loss / max(1, step_count)
            loss_history.append(avg_loss)
            logger.info("SimCLR epoch %s/%s loss=%.4f", epoch + 1, cfg.epochs, avg_loss)

        ckpt_path = self.output_dir / "simclr_backbone.pt"
        torch.save({"state_dict": model.state_dict(), "config": cfg.__dict__}, ckpt_path)

        result = {
            "trained": True,
            "device": str(device),
            "num_images": len(image_paths),
            "epochs": cfg.epochs,
            "max_steps_per_epoch": cfg.max_steps_per_epoch,
            "loss_history": [round(float(v), 6) for v in loss_history],
            "checkpoint": str(ckpt_path),
        }
        (self.output_dir / "simclr_results.json").write_text(json.dumps(result, indent=2))
        return result
