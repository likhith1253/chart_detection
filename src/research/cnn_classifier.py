from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from src.utils.runtime import detect_runtime, get_git_commit, memory_snapshot
from src.utils.dataset_paths import resolve_split_dataframe, validate_images

logger = logging.getLogger(__name__)

CHART_LABELS = ["bar_chart", "histogram", "line_chart", "pie_chart", "scatter_plot"]


@dataclass
class TrainConfig:
    image_size: int = 160
    batch_size: int = 32
    epochs: int = 6
    freeze_epochs: int = 2
    finetune_last_blocks: int = 2
    lr_head: float = 1e-3
    lr_finetune: float = 1e-4
    weight_decay: float = 1e-4
    patience: int = 3
    min_delta: float = 1e-4
    num_workers: int = 0
    max_train_samples: Optional[int] = None
    seed: int = 42


class _ChartDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label_idx"]), row["image_path"], row["label"]


class CNNChartClassifier:
    def __init__(self, output_dir: str | Path = "results/cnn_baseline") -> None:
        runtime = detect_runtime()
        self.runtime = runtime
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = self.output_dir / "checkpoints"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.fig_dir = self.output_dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(os.environ.get("LOG_DIR", runtime.project_root / "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(runtime.device)
        self.use_amp = self.device.type == "cuda"
        self.config = TrainConfig()
        self.model: Optional[nn.Module] = None
        self.label_to_idx: Dict[str, int] = {}
        self.idx_to_label: Dict[int, str] = {}
        self.history: List[Dict] = []
        self.latest_checkpoint = self.models_dir / "latest_checkpoint.pt"
        self.best_checkpoint = self.models_dir / "best_checkpoint.pt"
        self.optimizer_path = self.models_dir / "optimizer.pt"
        self.scheduler_path = self.models_dir / "scheduler.pt"
        self.training_state_path = self.models_dir / "training_state.json"
        self.meta_path = self.models_dir / "experiment.json"
        self.results_path = self.output_dir / "cnn_metrics.json"
        self.training_log_path = self.log_dir / "training.log"
        self._setup_logging()
        self.writer = self._make_writer()
        self._runtime_summary_written = False

    def _make_writer(self):
        try:
            from torch.utils.tensorboard import SummaryWriter

            return SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        except Exception:
            return None

    def _setup_logging(self) -> None:
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "") == str(self.training_log_path) for h in root.handlers):
            file_handler = logging.FileHandler(self.training_log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(formatter)
            root.addHandler(stream_handler)

    def _gpu_snapshot(self) -> Dict[str, object]:
        if self.device.type != "cuda":
            return {"gpu_name": "none", "gpu_utilization": None, "gpu_memory_used_mb": None, "gpu_memory_total_mb": None}
        try:
            import subprocess

            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
            name, util, used, total = [part.strip() for part in out.split(",")]
            return {
                "gpu_name": name,
                "gpu_utilization": float(util),
                "gpu_memory_used_mb": float(used),
                "gpu_memory_total_mb": float(total),
            }
        except Exception:
            return {"gpu_name": "cuda", "gpu_utilization": None, "gpu_memory_used_mb": None, "gpu_memory_total_mb": None}

    def _log_system_metrics(self, rows: List[Dict]) -> None:
        if not rows:
            return
        pd.DataFrame(rows).to_csv(self.output_dir / "system_metrics.csv", index=False)

    def _save_training_state(self, state: Dict) -> None:
        self.training_state_path.write_text(json.dumps(state, indent=2))

    def _load_training_state(self) -> Dict:
        if self.training_state_path.exists():
            try:
                return json.loads(self.training_state_path.read_text())
            except Exception:
                return {}
        return {}

    def _validate_inputs(self, df: pd.DataFrame) -> None:
        assert not df.empty, "Dataset split is empty"
        assert {"train", "validation", "test"}.issubset(set(df["split"].unique())), "Missing required splits"
        assert df["label"].isin(CHART_LABELS).all(), "Unexpected labels found"
        assert self.output_dir.exists() and self.models_dir.exists() and self.fig_dir.exists(), "Output folders missing"

    def _seed_all(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _load_split(
        self,
        split_csv: str | Path,
        metadata_root: str | Path | None = None,
        image_root: str | Path | None = None,
        split_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        df = split_df.copy() if split_df is not None else pd.read_csv(split_csv)
        df = df[df["label"].isin(CHART_LABELS)].copy()
        df = resolve_split_dataframe(
            df,
            metadata_root=metadata_root or Path(split_csv).parent,
            image_root=image_root or "data/raw_images",
        )
        validate_images(df["image_path"].tolist())
        return df

    def _maybe_subsample(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if cfg.max_train_samples and len(df) > cfg.max_train_samples:
            per_class = max(1, cfg.max_train_samples // max(1, df["label"].nunique()))
            df = (
                df.groupby("label", group_keys=False)
                .apply(lambda g: g.sample(min(len(g), per_class), random_state=cfg.seed))
                .reset_index(drop=True)
            )
        return df

    def _build_model(self, num_classes: int) -> nn.Module:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    def _freeze_backbone(self) -> None:
        assert self.model is not None
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True

    def _unfreeze_last_blocks(self, blocks: int) -> None:
        assert self.model is not None
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        if hasattr(self.model, "features"):
            for block in list(self.model.features.children())[-blocks:]:
                for param in block.parameters():
                    param.requires_grad = True

    def _transforms(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        train_tf = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.12, contrast=0.12),
                transforms.ToTensor(),
                normalize,
            ]
        )
        eval_tf = transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        return train_tf, eval_tf

    def _load_checkpoint(self) -> Dict:
        if not self.latest_checkpoint.exists():
            return {}
        return torch.load(self.latest_checkpoint, map_location=self.device)

    def _save_checkpoint(self, epoch: int, best_score: float, optimizer, scheduler, scaler, stage: str, save_best: bool = False) -> None:
        assert self.model is not None
        payload = {
            "epoch": epoch,
            "stage": stage,
            "best_score": best_score,
            "state_dict": self.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": asdict(self.config),
            "label_to_idx": self.label_to_idx,
        }
        torch.save(payload, self.latest_checkpoint)
        if save_best:
            torch.save(payload, self.best_checkpoint)
        torch.save(optimizer.state_dict(), self.optimizer_path)
        if scheduler is not None:
            torch.save(scheduler.state_dict(), self.scheduler_path)
        self._save_training_state({
            "epoch": epoch,
            "best_score": best_score,
            "stage": stage,
            "latest_checkpoint": str(self.latest_checkpoint),
            "best_checkpoint": str(self.best_checkpoint),
            "optimizer_path": str(self.optimizer_path),
            "scheduler_path": str(self.scheduler_path),
        })

    def _evaluate(self, loader: DataLoader, criterion: Optional[nn.Module] = None) -> Dict[str, float]:
        assert self.model is not None
        self.model.eval()
        y_true, y_pred = [], []
        confidences = []
        probabilities = []
        total_loss = 0.0
        total_seen = 0
        with torch.no_grad():
            for images, labels, _, _ in loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                logits = self.model(images)
                if criterion is not None:
                    loss = criterion(logits, labels)
                    total_loss += float(loss.item()) * int(labels.size(0))
                    total_seen += int(labels.size(0))
                probs = torch.softmax(logits, dim=1)
                conf, preds_t = torch.max(probs, dim=1)
                preds = preds_t.cpu().numpy().tolist()
                y_pred.extend(preds)
                y_true.extend(labels.cpu().numpy().tolist())
                confidences.extend(conf.cpu().numpy().astype(float).tolist())
                probabilities.extend(probs.cpu().numpy().astype(float).tolist())
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(self.idx_to_label)))).tolist(),
            "y_true": y_true,
            "y_pred": y_pred,
            "confidence": confidences,
            "probabilities": probabilities,
            "loss": float(total_loss / max(1, total_seen)) if criterion is not None else None,
        }

    def fit_from_split(
        self,
        split_csv: str | Path,
        config_in: Optional[TrainConfig] = None,
        metadata_root: str | Path | None = None,
        image_root: str | Path | None = None,
        split_df: pd.DataFrame | None = None,
    ) -> Dict:
        self.config = config_in or self.config
        self._seed_all(self.config.seed)

        df = self._maybe_subsample(
            self._load_split(
                split_csv,
                metadata_root=metadata_root,
                image_root=image_root,
                split_df=split_df,
            )
        )
        self._validate_inputs(df)
        train_df = df[df["split"] == "train"].copy()
        val_df = df[df["split"] == "validation"].copy()
        test_df = df[df["split"] == "test"].copy()
        classes = sorted(df["label"].unique().tolist())
        self.label_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}
        for frame in (train_df, val_df, test_df):
            frame["label_idx"] = frame["label"].map(self.label_to_idx)

        train_tf, eval_tf = self._transforms()
        loader_kwargs = {
            "num_workers": self.config.num_workers,
            "pin_memory": self.device.type == "cuda",
        }
        if self.config.num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = 2
        train_loader = DataLoader(
            _ChartDataset(train_df, train_tf),
            batch_size=self.config.batch_size,
            shuffle=True,
            **loader_kwargs,
        )
        val_loader = DataLoader(
            _ChartDataset(val_df, eval_tf),
            batch_size=self.config.batch_size,
            shuffle=False,
            **loader_kwargs,
        )
        test_loader = DataLoader(
            _ChartDataset(test_df, eval_tf),
            batch_size=self.config.batch_size,
            shuffle=False,
            **loader_kwargs,
        )

        self.model = self._build_model(len(classes)).to(self.device)
        state = self._load_training_state()
        ckpt = self._load_checkpoint()
        start_epoch = int(state.get("epoch", 0))
        best_score = float(state.get("best_score", -1.0))
        stage = state.get("stage", "freeze")
        if ckpt:
            try:
                self.model.load_state_dict(ckpt["state_dict"])
                self.label_to_idx = ckpt.get("label_to_idx", self.label_to_idx)
                self.idx_to_label = {i: c for c, i in self.label_to_idx.items()}
                self.config = TrainConfig(**ckpt.get("config", asdict(self.config)))
                logger.info("Resuming CNN from epoch %s stage %s", start_epoch, stage)
            except Exception:
                logger.warning("Checkpoint load failed; starting fresh")
                start_epoch = 0
                best_score = -1.0
                stage = "freeze"

        self._freeze_backbone()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.lr_head, weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
        scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        criterion = nn.CrossEntropyLoss()

        history: List[Dict] = []
        epoch_rows: List[Dict] = []
        system_rows: List[Dict] = []
        best_epoch = start_epoch
        no_improve = 0
        total_epochs = self.config.epochs
        train_start = perf_counter()
        for epoch in range(start_epoch, total_epochs):
            if epoch >= self.config.freeze_epochs:
                if stage != "finetune":
                    self._unfreeze_last_blocks(self.config.finetune_last_blocks)
                    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.config.lr_finetune, weight_decay=self.config.weight_decay)
                    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=1)
                    stage = "finetune"
            self.model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            for images, labels, _, _ in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    logits = self.model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += float(loss.item()) * labels.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))
            train_loss = epoch_loss / max(1, total)
            train_acc = correct / max(1, total)
            val_loss = float("nan")
            val_metrics = self._evaluate(val_loader, criterion)
            scheduler.step(val_metrics["macro_f1"])
            elapsed = perf_counter() - train_start
            eta = max(0.0, (epoch + 1 < total_epochs) * ((elapsed / max(1, epoch + 1)) * (total_epochs - (epoch + 1))))
            gpu = self._gpu_snapshot()
            sys_snap = memory_snapshot()
            checkpoint_saved = False
            history.append({
                "epoch": epoch + 1,
                "stage": stage,
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"] if val_metrics["loss"] is not None else float("nan"),
                "train_accuracy": train_acc,
                "val_accuracy": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "lr": optimizer.param_groups[0]["lr"],
                "epoch_time_sec": elapsed / max(1, epoch + 1),
                "elapsed_time_sec": elapsed,
                "eta_sec": eta,
                "best_val_accuracy": max((h["val_accuracy"] for h in history), default=val_metrics["accuracy"]),
                "checkpoint_saved": False,
                **gpu,
                **sys_snap,
            })
            epoch_rows.append(history[-1])
            system_rows.append({
                "epoch": epoch + 1,
                "elapsed_time_sec": elapsed,
                "cpu_usage": sys_snap["cpu_percent"],
                "ram_usage": sys_snap["ram_percent"],
                "gpu_name": gpu["gpu_name"],
                "gpu_utilization": gpu["gpu_utilization"],
                "gpu_memory_used_mb": gpu["gpu_memory_used_mb"],
            })
            if self.writer is not None:
                self.writer.add_scalar("Loss/train", train_loss, epoch + 1)
                self.writer.add_scalar("Accuracy/train", train_acc, epoch + 1)
                self.writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch + 1)
                self.writer.add_scalar("F1/val_macro", val_metrics["macro_f1"], epoch + 1)
                self.writer.add_scalar("Loss/lr", optimizer.param_groups[0]["lr"], epoch + 1)
                self.writer.add_histogram("model/classifier_weight", self.model.classifier[1].weight.detach().cpu(), epoch + 1)
                self.writer.add_image("validation/confusion_matrix", self._confusion_figure_array(val_metrics["confusion_matrix"]), epoch + 1, dataformats="HWC")
                val_cm = np.array(val_metrics["confusion_matrix"])
                for idx, label in enumerate(classes):
                    self.writer.add_scalar(f"PerClassAccuracy/{label}", float(val_cm[idx, idx]) / max(1, int(val_cm[idx].sum())), epoch + 1)

            logger.info(
                "Epoch=%s/%s stage=%s train_loss=%.4f val_loss=%.4f train_acc=%.4f val_acc=%.4f lr=%.6f epoch_time=%.2fs elapsed=%.2fs eta=%.2fs gpu_mem=%s gpu_util=%s cpu=%.1f ram=%.1f best_val_acc=%.4f checkpoint_saved=%s",
                epoch + 1,
                total_epochs,
                stage,
                train_loss,
                val_metrics["loss"] if val_metrics["loss"] is not None else float("nan"),
                train_acc,
                val_metrics["accuracy"],
                optimizer.param_groups[0]["lr"],
                elapsed / max(1, epoch + 1),
                elapsed,
                eta,
                gpu.get("gpu_memory_used_mb"),
                gpu.get("gpu_utilization"),
                sys_snap["cpu_percent"],
                sys_snap["ram_percent"],
                max((h["val_accuracy"] for h in history), default=val_metrics["accuracy"]),
                checkpoint_saved,
            )
            improved = val_metrics["macro_f1"] > best_score + self.config.min_delta
            if improved:
                best_score = val_metrics["macro_f1"]
                best_epoch = epoch + 1
                no_improve = 0
                checkpoint_saved = True
            else:
                no_improve += 1
                if no_improve >= self.config.patience:
                    logger.info("CNN early stopping at epoch %s", epoch + 1)
                    self._save_checkpoint(epoch + 1, best_score, optimizer, scheduler, scaler, stage, save_best=False)
                    history[-1]["checkpoint_saved"] = True
                    epoch_rows[-1]["checkpoint_saved"] = True
                    self._save_training_state({
                        "epoch": epoch + 1,
                        "best_score": best_score,
                        "stage": stage,
                        "best_epoch": best_epoch,
                    })
                    break
            self._save_checkpoint(epoch + 1, best_score, optimizer, scheduler, scaler, stage, save_best=improved)
            history[-1]["checkpoint_saved"] = True
            epoch_rows[-1]["checkpoint_saved"] = True
            self._save_training_state({
                "epoch": epoch + 1,
                "best_score": best_score,
                "stage": stage,
                "best_epoch": best_epoch,
            })

        if self.best_checkpoint.exists():
            payload = torch.load(self.best_checkpoint, map_location=self.device)
            self.model.load_state_dict(payload["state_dict"])
        test_start = perf_counter()
        test_metrics = self._evaluate(test_loader, criterion)
        inference_time = (perf_counter() - test_start) / max(1, len(test_df))
        total_time = perf_counter() - train_start
        cm = np.array(test_metrics["confusion_matrix"])
        per_class = {classes[i]: float(cm[i, i]) / max(1, int(cm[i].sum())) for i in range(len(classes))}
        test_metrics.update({
            "per_class_accuracy": per_class,
            "inference_time_per_image_sec": inference_time,
            "training_time_sec": total_time,
            "best_val_macro_f1": best_score,
            "best_epoch": best_epoch,
        })
        self.history = history
        self._write_outputs(df, train_df, val_df, test_df, classes, history, test_metrics, inference_time)
        self._write_training_csvs(history, epoch_rows, system_rows)
        self._log_system_metrics(system_rows)
        result = {
            "backbone": "efficientnet_b0",
            "device": str(self.device),
            "classes": classes,
            "history": history,
            "test": test_metrics,
            "best_val_macro_f1": best_score,
            "best_epoch": best_epoch,
            "training_time_sec": total_time,
            "model_path": str(self.best_checkpoint),
        }
        self.results_path.write_text(json.dumps(result, indent=2))
        self.meta_path.write_text(json.dumps({
            "git_commit": get_git_commit(),
            "timestamp": pd.Timestamp.utcnow().isoformat(),
            "hyperparameters": asdict(self.config),
            "dataset_fingerprint": self._dataset_fingerprint(df),
            "model_version": "efficientnet_b0",
            "random_seed": self.config.seed,
            "pytorch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu_name": self._gpu_snapshot()["gpu_name"],
            "training_time_sec": total_time,
        }, indent=2))
        (self.output_dir / "experiment_summary.json").write_text(json.dumps({
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_score,
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "training_time_sec": total_time,
            "inference_time_per_image_sec": inference_time,
            "device": str(self.device),
            "checkpoint_paths": {
                "latest": str(self.latest_checkpoint),
                "best": str(self.best_checkpoint),
            },
        }, indent=2))
        if self.writer is not None:
            self.writer.close()
        return result

    def _write_outputs(self, full_df, train_df, val_df, test_df, classes, history, metrics, inference_time):
        hist = pd.DataFrame(history)
        hist.to_csv(self.output_dir / "training_history.csv", index=False)
        self._save_curves(hist)
        pd.DataFrame(metrics["confusion_matrix"], index=classes, columns=classes).to_csv(self.output_dir / "cnn_confusion_matrix.csv")
        self._save_confusion_png(metrics["confusion_matrix"], classes)
        pd.DataFrame([{
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "training_time_sec": metrics["training_time_sec"],
            "inference_time_per_image_sec": inference_time,
        }]).to_csv(self.output_dir / "cnn_metrics_summary.csv", index=False)
        self._save_error_analysis(test_df, metrics)
        self._comparison_table(metrics)
        self._save_featureless_figures(full_df, train_df, val_df, test_df, metrics)

    def _write_training_csvs(self, history: List[Dict], epoch_rows: List[Dict], system_rows: List[Dict]) -> None:
        history_df = pd.DataFrame(history)
        epoch_df = pd.DataFrame(epoch_rows)
        system_df = pd.DataFrame(system_rows)
        history_df.to_csv(self.output_dir / "training_history.csv", index=False)
        epoch_df.to_csv(self.output_dir / "epoch_metrics.csv", index=False)
        epoch_df.to_csv(self.output_dir / "training_metrics.csv", index=False)
        system_df.to_csv(self.output_dir / "system_metrics.csv", index=False)

    def _save_curves(self, hist: pd.DataFrame):
        if hist.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hist["epoch"], hist["train_loss"], label="train_loss")
        ax.plot(hist["epoch"], hist["val_macro_f1"], label="val_macro_f1")
        ax.set_xlabel("Epoch")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.fig_dir / "training_curves.png", dpi=300)
        fig.savefig(self.fig_dir / "training_curves.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hist["epoch"], hist["train_accuracy"], label="train_acc")
        ax.plot(hist["epoch"], hist["val_accuracy"], label="val_acc")
        ax.set_xlabel("Epoch")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.fig_dir / "accuracy_curves.png", dpi=300)
        fig.savefig(self.fig_dir / "accuracy_curves.pdf")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(hist["epoch"], hist["train_loss"], label="loss")
        ax.set_xlabel("Epoch")
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.fig_dir / "loss_curves.png", dpi=300)
        fig.savefig(self.fig_dir / "loss_curves.pdf")
        plt.close(fig)

    def _save_confusion_png(self, cm, classes):
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(self.fig_dir / "cnn_confusion_matrix.png", dpi=300)
        fig.savefig(self.fig_dir / "cnn_confusion_matrix.pdf")
        plt.close(fig)

    def _save_error_analysis(self, test_df, metrics):
        pred = np.array(metrics["y_pred"])
        truth = np.array(metrics["y_true"])
        confs = np.array(metrics.get("confidence", np.ones(len(pred))))
        wrong = truth != pred
        error_df = test_df.copy().reset_index(drop=True)
        error_df["true_label"] = truth
        error_df["predicted_label"] = pred
        error_df["is_error"] = wrong
        error_df = error_df[error_df["is_error"]].copy()
        error_df["prediction_confidence"] = confs[wrong]
        error_df.to_csv(self.output_dir / "cnn_error_analysis.csv", index=False)
        error_df.sort_values("prediction_confidence", ascending=False).head(50).to_csv(self.output_dir / "cnn_top50_mistakes.csv", index=False)

        try:
            xgb = pd.read_csv("results/classification/error_analysis.csv")
            xgb_paths = set(xgb["image_path"].astype(str).tolist())
            cnn_paths = set(error_df["image_path"].astype(str).tolist())
            both = sorted(xgb_paths & cnn_paths)
            only_cnn = sorted(cnn_paths - xgb_paths)
            only_xgb = sorted(xgb_paths - cnn_paths)
            pd.DataFrame({"image_path": both}).to_csv(self.output_dir / "both_models_fail.csv", index=False)
            pd.DataFrame({"image_path": only_cnn}).to_csv(self.output_dir / "cnn_only_fail.csv", index=False)
            pd.DataFrame({"image_path": only_xgb}).to_csv(self.output_dir / "xgb_only_fail.csv", index=False)
        except Exception:
            pass

    def _comparison_table(self, cnn_metrics):
        try:
            xgb_metrics = json.loads(Path("results/classification/validation_metrics.json").read_text())
            xgb_model_size = Path("results/classification/models/xgboost.joblib").stat().st_size
            xgb_inf = xgb_metrics.get("inference_time_per_image_sec", np.nan)
            xgb_acc = xgb_metrics.get("accuracy", np.nan)
        except Exception:
            xgb_metrics, xgb_model_size, xgb_inf, xgb_acc = {}, np.nan, np.nan, np.nan
        classical = []
        try:
            train_metrics = json.loads(Path("results/classification/metrics.json").read_text())
            for name in ["logistic_regression", "svm_rbf", "random_forest", "xgboost"]:
                if name in train_metrics.get("models", {}):
                    m = train_metrics["models"][name]
                    classical.append({
                        "model": name,
                        "accuracy": m.get("accuracy", np.nan),
                        "runtime": m.get("train_time_sec", np.nan),
                        "model_size": (Path("results/classification/models") / f"{name}.joblib").stat().st_size if (Path("results/classification/models") / f"{name}.joblib").exists() else np.nan,
                        "inference_speed": np.nan,
                    })
        except Exception:
            pass
        classical.append({
            "model": "efficientnet_b0",
            "accuracy": cnn_metrics["accuracy"],
            "runtime": cnn_metrics["training_time_sec"],
            "model_size": self.best_checkpoint.stat().st_size if self.best_checkpoint.exists() else np.nan,
            "inference_speed": cnn_metrics["inference_time_per_image_sec"],
        })
        pd.DataFrame(classical).to_csv(self.output_dir / "comparison_table.csv", index=False)

    def _save_featureless_figures(self, full_df, train_df, val_df, test_df, metrics):
        fig, ax = plt.subplots(figsize=(7, 5))
        counts = full_df["label"].value_counts().reindex(CHART_LABELS).fillna(0)
        ax.bar(counts.index, counts.values)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        fig.savefig(self.fig_dir / "class_distribution.png", dpi=300)
        fig.savefig(self.fig_dir / "class_distribution.pdf")
        plt.close(fig)

        probs = np.array(metrics.get("probabilities", []))
        if probs.size and len(metrics.get("y_true", [])):
            y_true = np.array(metrics["y_true"])
            y_bin = np.eye(len(CHART_LABELS))[y_true]
            fig, ax = plt.subplots(figsize=(7, 6))
            for idx, label in enumerate(CHART_LABELS):
                fpr, tpr, _ = roc_curve(y_bin[:, idx], probs[:, idx])
                ax.plot(fpr, tpr, label=label)
            ax.plot([0, 1], [0, 1], "k--")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(self.fig_dir / "roc_curve.png", dpi=300)
            fig.savefig(self.fig_dir / "roc_curve.pdf")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7, 6))
            for idx, label in enumerate(CHART_LABELS):
                precision, recall, _ = precision_recall_curve(y_bin[:, idx], probs[:, idx])
                ax.plot(recall, precision, label=label)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(self.fig_dir / "precision_recall_curve.png", dpi=300)
            fig.savefig(self.fig_dir / "precision_recall_curve.pdf")
            plt.close(fig)

    def _confusion_figure_array(self, cm: List[List[int]]) -> np.ndarray:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(cm, cmap="Blues")
        ax.axis("off")
        fig.canvas.draw()
        arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        plt.close(fig)
        return arr

    def _dataset_fingerprint(self, df: pd.DataFrame) -> str:
        import hashlib

        payload = {
            "rows": int(len(df)),
            "labels": sorted(df["label"].unique().tolist()),
            "splits": df["split"].value_counts().to_dict(),
            "image_size": self.config.image_size,
            "seed": self.config.seed,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def predict(self, image_paths: Sequence[str]) -> Tuple[pd.Series, pd.Series]:
        if self.model is None and not self.load():
            raise RuntimeError("CNN model is not available")
        _, eval_tf = self._transforms()
        ds = _ChartDataset(pd.DataFrame({"image_path": list(image_paths), "label_idx": [0] * len(image_paths), "label": ["bar_chart"] * len(image_paths)}), eval_tf)
        loader = DataLoader(ds, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        preds: List[str] = []
        confs: List[float] = []
        self.model.eval()
        with torch.no_grad():
            for images, _, _, _ in loader:
                images = images.to(self.device)
                probs = torch.softmax(self.model(images), dim=1)
                conf, idx = torch.max(probs, dim=1)
                preds.extend([self.idx_to_label[int(i)] for i in idx.cpu().numpy().tolist()])
                confs.extend(conf.cpu().numpy().astype(float).tolist())
        return pd.Series(preds), pd.Series(confs)

    def load(self) -> bool:
        if not self.best_checkpoint.exists():
            return False
        payload = torch.load(self.best_checkpoint, map_location=self.device)
        cfg = TrainConfig(**payload.get("config", asdict(self.config)))
        self.config = cfg
        self.label_to_idx = {str(k): int(v) for k, v in payload["label_to_idx"].items()}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        self.model = self._build_model(len(self.label_to_idx)).to(self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.model.eval()
        return True
