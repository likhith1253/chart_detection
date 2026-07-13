"""
Detection wrapper around YOLO chart element detector with thread safety.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Dict

from src.research.yolo_chart_detector import YOLOChartElementDetector
from src.segmentation.chart_element_segmenter import ChartElementSegmenter

logger = logging.getLogger(__name__)


class ChartDetectionService:
    """Safe adapter for YOLO detection in parallel pipeline runs."""

    def __init__(self, imgsz: int = 512, device: str = "cpu") -> None:
        self.detector = YOLOChartElementDetector(imgsz=imgsz, device=device)
        self._lock = threading.Lock()
        self.segmenter = ChartElementSegmenter()
        self.detector.load_best()

    def infer_counts(self, image_path: str | Path) -> Dict[str, int]:
        with self._lock:
            try:
                if self.detector.model is None:
                    seg = self.segmenter.segment_elements(str(image_path))
                    return {
                        "bars": len(seg.get("bars", [])),
                        "line_segments": len(seg.get("lines", [])),
                        "scatter_points": len(seg.get("points", [])),
                        "pie_slices": len(seg.get("pie_slices", [])),
                        "axes": len(seg.get("axes", [])),
                        "legend_boxes": 0,
                        "text_regions": 0,
                    }
                return self.detector.infer_counts(Path(image_path))
            except Exception as exc:
                logger.warning("YOLO inference failed for %s: %s", image_path, exc)
                return {name: 0 for name in self.detector.CLASS_NAMES}
