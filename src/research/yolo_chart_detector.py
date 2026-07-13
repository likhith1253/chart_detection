"""
YOLOv8 chart element detection with automatic pseudo-label generation.
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

import config
from src.ocr.ocr_engine import OCREngine
from src.segmentation.chart_element_segmenter import ChartElementSegmenter

logger = logging.getLogger(__name__)


class YOLOChartElementDetector:
    """Train/infer YOLOv8 for chart component detection."""

    CLASS_NAMES = [
        "bars",
        "line_segments",
        "scatter_points",
        "pie_slices",
        "axes",
        "legend_boxes",
        "text_regions",
    ]

    def __init__(self, work_dir: Path | None = None, imgsz: int = 640, device: str = "cpu"):
        self.work_dir = work_dir or (config.RESULT_DIR / "yolo")
        self.dataset_dir = self.work_dir / "dataset"
        self.run_dir = self.work_dir / "runs"
        self.model_dir = self.work_dir / "models"
        self.imgsz = imgsz
        self.device = device
        self.segmenter = ChartElementSegmenter()
        self.ocr_engine = OCREngine()
        self.model = None
        self.best_weights: Path | None = None

        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clip_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(0, min(w - 1, x2))
        y2 = max(0, min(h - 1, y2))
        if x2 <= x1:
            x2 = min(w - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h - 1, y1 + 1)
        return x1, y1, x2, y2

    @staticmethod
    def _to_yolo_line(class_id: int, box: Tuple[int, int, int, int], w: int, h: int) -> str:
        x1, y1, x2, y2 = box
        cx = ((x1 + x2) / 2.0) / w
        cy = ((y1 + y2) / 2.0) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"

    @staticmethod
    def _line_to_box(line: List[int], w: int, h: int, thickness: int = 6) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = map(int, line)
        return YOLOChartElementDetector._clip_box(
            min(x1, x2) - thickness,
            min(y1, y2) - thickness,
            max(x1, x2) + thickness,
            max(y1, y2) + thickness,
            w,
            h,
        )

    def _detect_legend_boxes(self, image: np.ndarray, plot_area: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Detect small colored boxes outside plot area as legend proxies."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        mask = cv2.inRange(sat, 45, 255)
        mask = cv2.bitwise_and(mask, cv2.inRange(val, 40, 255))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[Tuple[int, int, int, int]] = []

        plot_box = plot_area[0] if plot_area else (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
        px, py, pw, ph = plot_box

        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < 40 or area > (w * h) * 0.02:
                continue

            in_plot = (px <= x <= px + pw) and (py <= y <= py + ph)
            if in_plot:
                continue

            ar = cw / float(ch + 1e-6)
            if 0.5 <= ar <= 2.5:
                boxes.append((x, y, x + cw, y + ch))
        return boxes

    def _pseudo_labels_for_image(self, image_path: Path) -> Tuple[List[str], Dict[str, int]]:
        img = cv2.imread(str(image_path))
        if img is None:
            return [], {}

        h, w = img.shape[:2]
        seg = self.segmenter.segment_elements(str(image_path))
        text_regions = self.ocr_engine.detect_text_regions(str(image_path))
        legend_boxes = self._detect_legend_boxes(img, seg.get("plot_area", []))

        class_id = {name: idx for idx, name in enumerate(self.CLASS_NAMES)}
        lines: List[str] = []
        counts = {name: 0 for name in self.CLASS_NAMES}

        for (x, y, bw, bh) in seg.get("bars", []):
            box = self._clip_box(x, y, x + bw, y + bh, w, h)
            lines.append(self._to_yolo_line(class_id["bars"], box, w, h))
            counts["bars"] += 1

        for p in seg.get("points", []):
            cx, cy = int(p[0]), int(p[1])
            box = self._clip_box(cx - 5, cy - 5, cx + 5, cy + 5, w, h)
            lines.append(self._to_yolo_line(class_id["scatter_points"], box, w, h))
            counts["scatter_points"] += 1

        for line in seg.get("lines", []):
            box = self._line_to_box(line, w, h)
            lines.append(self._to_yolo_line(class_id["line_segments"], box, w, h))
            counts["line_segments"] += 1

        for line in seg.get("axes", []):
            box = self._line_to_box(line, w, h, thickness=8)
            lines.append(self._to_yolo_line(class_id["axes"], box, w, h))
            counts["axes"] += 1

        for contour in seg.get("pie_slices", []):
            x, y, bw, bh = cv2.boundingRect(contour)
            box = self._clip_box(x, y, x + bw, y + bh, w, h)
            lines.append(self._to_yolo_line(class_id["pie_slices"], box, w, h))
            counts["pie_slices"] += 1

        for box in legend_boxes:
            lines.append(self._to_yolo_line(class_id["legend_boxes"], box, w, h))
            counts["legend_boxes"] += 1

        for (x, y, bw, bh) in text_regions:
            box = self._clip_box(x, y, x + bw, y + bh, w, h)
            lines.append(self._to_yolo_line(class_id["text_regions"], box, w, h))
            counts["text_regions"] += 1

        return lines, counts

    def build_dataset(
        self,
        image_paths: List[Path],
        max_images: int | None = None,
        random_seed: int = 42,
    ) -> Path:
        """Create YOLO dataset with pseudo-labels from segmentation + OCR."""
        random.seed(random_seed)

        if max_images is not None and len(image_paths) > max_images:
            image_paths = random.sample(image_paths, max_images)
        image_paths = sorted(image_paths)

        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
        for split in ("train", "val"):
            (self.dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        split_idx = int(len(image_paths) * 0.8)
        train_paths = image_paths[:split_idx]
        val_paths = image_paths[split_idx:]

        summary: Dict[str, int] = {name: 0 for name in self.CLASS_NAMES}
        kept = 0
        for split, paths in (("train", train_paths), ("val", val_paths)):
            for src_path in paths:
                try:
                    labels, counts = self._pseudo_labels_for_image(src_path)
                    if not labels:
                        continue
                    kept += 1
                    for k, v in counts.items():
                        summary[k] += int(v)

                    dst_img = self.dataset_dir / "images" / split / src_path.name
                    dst_lbl = self.dataset_dir / "labels" / split / f"{src_path.stem}.txt"
                    shutil.copy2(src_path, dst_img)
                    dst_lbl.write_text("\n".join(labels))
                except Exception as exc:
                    logger.warning("YOLO pseudo-label failed for %s: %s", src_path.name, exc)

        data_yaml = self.dataset_dir / "data.yaml"
        names_map = {idx: name for idx, name in enumerate(self.CLASS_NAMES)}
        yaml_text = (
            f"path: {self.dataset_dir.as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            + "\n".join([f"  {idx}: {name}" for idx, name in names_map.items()])
            + "\n"
        )
        data_yaml.write_text(yaml_text)

        summary_path = self.work_dir / "annotation_summary.json"
        summary_payload = {
            "images_requested": len(image_paths),
            "images_with_labels": kept,
            "element_counts": summary,
            "classes": self.CLASS_NAMES,
            "data_yaml": str(data_yaml),
        }
        summary_path.write_text(json.dumps(summary_payload, indent=2))
        logger.info("YOLO pseudo-label dataset created: %s", data_yaml)
        return data_yaml

    def train(self, data_yaml: Path, epochs: int = 40, batch: int = 16) -> Dict:
        """Train YOLOv8 detector with transfer learning."""
        try:
            from ultralytics import YOLO
        except Exception as exc:
            logger.warning("Ultralytics unavailable, skipping YOLO training: %s", exc)
            return {"trained": False, "reason": str(exc)}

        try:
            model = YOLO("yolov8n.pt")
            result = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=self.imgsz,
                batch=batch,
                workers=0,
                device=self.device,
                project=str(self.run_dir),
                name="chart_elements",
                exist_ok=True,
                verbose=False,
                pretrained=True,
            )
            save_dir = Path(result.save_dir)
            weights = save_dir / "weights" / "best.pt"
            if weights.exists():
                self.best_weights = self.model_dir / "yolo_chart_elements_best.pt"
                shutil.copy2(weights, self.best_weights)
                self.model = YOLO(str(self.best_weights))
                return {
                    "trained": True,
                    "weights": str(self.best_weights),
                    "results_dir": str(save_dir),
                }
            return {"trained": False, "reason": "best.pt not found"}
        except Exception as exc:
            logger.warning("YOLO training failed: %s", exc)
            return {"trained": False, "reason": str(exc)}

    def load_best(self) -> bool:
        """Load previously trained YOLO weights if available."""
        try:
            from ultralytics import YOLO
        except Exception:
            return False

        candidate = self.model_dir / "yolo_chart_elements_best.pt"
        if not candidate.exists():
            return False
        self.best_weights = candidate
        self.model = YOLO(str(candidate))
        return True

    @staticmethod
    def structure_classify(counts: Dict[str, int], histogram_bias: float = 0.5) -> Tuple[str, float]:
        """Classify chart type from structured counts only."""
        bars = counts.get("bars", 0)
        lines = counts.get("line_segments", 0)
        points = counts.get("scatter_points", 0)
        pie = counts.get("pie_slices", 0)
        axes = counts.get("axes", 0)

        scores = {
            "bar_chart": 0.0,
            "histogram": 0.0,
            "line_chart": 0.0,
            "scatter_plot": 0.0,
            "pie_chart": 0.0,
        }

        scores["bar_chart"] += min(3.0, bars * 0.5) + min(1.0, axes * 0.2)
        scores["histogram"] += min(4.0, bars * 0.6) + histogram_bias
        scores["line_chart"] += min(4.0, lines * 0.5) + (1.0 if bars < 3 else 0.0)
        scores["scatter_plot"] += min(4.0, points * 0.2) + (1.0 if lines < 6 else 0.0)
        scores["pie_chart"] += min(5.0, pie * 0.7) + (0.5 if bars < 2 else 0.0)

        pred = max(scores, key=scores.get)
        ordered = sorted(scores.values(), reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        conf = min(0.99, 0.5 + margin / 5.0)
        return pred, float(conf)

    def infer_counts(self, image_path: Path, conf: float = 0.25) -> Dict[str, int]:
        """Infer structured counts from YOLO model or fallback segmentation."""
        counts = {name: 0 for name in self.CLASS_NAMES}

        if self.model is not None:
            try:
                results = self.model.predict(
                    source=str(image_path),
                    conf=conf,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
                if results:
                    r0 = results[0]
                    if hasattr(r0, "boxes") and r0.boxes is not None and r0.boxes.cls is not None:
                        classes = r0.boxes.cls.detach().cpu().numpy().astype(int).tolist()
                        for cls_id in classes:
                            if 0 <= cls_id < len(self.CLASS_NAMES):
                                counts[self.CLASS_NAMES[cls_id]] += 1
                    return counts
            except Exception:
                pass

        # Fallback to segmentation + OCR text region detection.
        try:
            seg = self.segmenter.segment_elements(str(image_path))
            counts["bars"] = len(seg.get("bars", []))
            counts["line_segments"] = len(seg.get("lines", []))
            counts["scatter_points"] = len(seg.get("points", []))
            counts["pie_slices"] = len(seg.get("pie_slices", []))
            counts["axes"] = len(seg.get("axes", []))
            counts["text_regions"] = len(self.ocr_engine.detect_text_regions(str(image_path)))
        except Exception:
            pass
        return counts
