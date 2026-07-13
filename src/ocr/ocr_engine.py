"""
Robust OCR engine wrapper with fallback, retries, ensemble fusion, and caching.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from src.caching.ocr_cache import OCRCache
from src.ocr.ensemble import fuse_ocr_tokens
from src.pipeline.config import OCRConfig
from src.preprocessing.image_preprocessor import ImagePreprocessor

logger = logging.getLogger(__name__)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


class OCREngine:
    """Thread-safe OCR wrapper with Paddle retry logic and EasyOCR fallback."""

    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        cache: Optional[OCRCache] = None,
    ) -> None:
        self.config = config or OCRConfig()
        self.preprocessor = preprocessor
        self.cache = cache

        self.easyocr_reader = None
        self.paddleocr_reader = None
        self._easy_lock = threading.Lock()
        self._paddle_lock = threading.Lock()
        self._stat_lock = threading.Lock()

        self.stats = {
            "total_requests": 0,
            "paddle_attempts": 0,
            "paddle_successes": 0,
            "easy_attempts": 0,
            "easy_successes": 0,
            "fallback_events": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

        if "easyocr" in self.config.enabled_engines:
            self._init_easyocr()
        if "paddleocr" in self.config.enabled_engines:
            self._init_paddleocr(use_mkldnn=True)

    def _init_easyocr(self) -> None:
        try:
            import easyocr

            self.easyocr_reader = easyocr.Reader(["en"], gpu=False)
            logger.info("EasyOCR initialized")
        except Exception as exc:
            logger.warning("EasyOCR init failed: %s", exc)
            self.easyocr_reader = None

    def _init_paddleocr(self, use_mkldnn: bool) -> None:
        try:
            from paddleocr import PaddleOCR

            try:
                self.paddleocr_reader = PaddleOCR(
                    lang="en",
                    use_textline_orientation=False,
                    use_mkldnn=use_mkldnn,
                )
            except Exception as inner_exc:
                # Some PaddleOCR versions do not expose use_mkldnn in constructor.
                if "unknown argument" not in str(inner_exc).lower() and "unexpected keyword" not in str(inner_exc).lower():
                    raise
                self.paddleocr_reader = PaddleOCR(
                    lang="en",
                    use_textline_orientation=False,
                )
            logger.info("PaddleOCR initialized (use_mkldnn=%s)", use_mkldnn)
        except Exception as exc:
            logger.warning("PaddleOCR init failed (use_mkldnn=%s): %s", use_mkldnn, exc)
            self.paddleocr_reader = None

    @staticmethod
    def _to_bbox(points: List) -> List[int]:
        pts = np.asarray(points, dtype=float)
        x1, y1 = pts[:, 0].min(), pts[:, 1].min()
        x2, y2 = pts[:, 0].max(), pts[:, 1].max()
        return [int(x1), int(y1), int(x2), int(y2)]

    def _run_easyocr(self, image_path: str, image_input: Optional[np.ndarray]) -> Dict:
        with self._stat_lock:
            self.stats["easy_attempts"] += 1
        if self.easyocr_reader is None:
            return {"tokens": [], "texts": [], "confidence": 0.0, "success": False}

        try:
            source = image_path
            if image_input is not None and isinstance(image_input, np.ndarray):
                source = image_input
                if source.ndim == 2:
                    source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
            with self._easy_lock:
                try:
                    result = self.easyocr_reader.readtext(source, detail=1)
                except Exception:
                    result = self.easyocr_reader.readtext(image_path, detail=1)
            tokens = []
            for item in result:
                if len(item) < 3:
                    continue
                bbox = self._to_bbox(item[0])
                text = str(item[1]).strip()
                conf = float(item[2]) if item[2] is not None else 0.0
                tokens.append({"text": text, "confidence": conf, "bbox": bbox, "engine": "easyocr"})
            texts = [t["text"] for t in tokens if t["text"]]
            confs = [t["confidence"] for t in tokens]
            success = bool(tokens)
            with self._stat_lock:
                if success:
                    self.stats["easy_successes"] += 1
            return {
                "tokens": tokens,
                "texts": texts,
                "confidence": float(sum(confs) / len(confs)) if confs else 0.0,
                "success": success,
            }
        except Exception as exc:
            logger.warning(
                "EasyOCR failed for %s: %s",
                image_path,
                exc,
                extra={"event": "ocr_failure", "engine": "easyocr", "image_name": Path(image_path).name},
            )
            return {"tokens": [], "texts": [], "confidence": 0.0, "success": False}

    def _parse_paddle_result(self, result: object) -> List[Dict]:
        tokens: List[Dict] = []
        if not isinstance(result, list):
            return tokens

        for block in result:
            if block is None:
                continue
            if isinstance(block, list):
                for line in block:
                    if not isinstance(line, list) or len(line) < 2:
                        continue
                    raw_bbox = line[0]
                    text_info = line[1]
                    if not isinstance(text_info, (list, tuple)) or len(text_info) < 1:
                        continue
                    text = str(text_info[0]).strip()
                    conf = float(text_info[1]) if len(text_info) > 1 else 0.0
                    bbox = self._to_bbox(raw_bbox) if isinstance(raw_bbox, list) else [0, 0, 0, 0]
                    tokens.append({"text": text, "confidence": conf, "bbox": bbox, "engine": "paddleocr"})
            elif isinstance(block, dict):
                rec_texts = block.get("rec_texts", [])
                rec_scores = block.get("rec_scores", [])
                for idx, text in enumerate(rec_texts):
                    conf = float(rec_scores[idx]) if idx < len(rec_scores) else 0.0
                    tokens.append(
                        {"text": str(text).strip(), "confidence": conf, "bbox": [0, 0, 0, 0], "engine": "paddleocr"}
                    )
        return tokens

    def _run_paddle_once(self, image_path: str, image_input: Optional[np.ndarray]) -> Dict:
        if self.paddleocr_reader is None:
            return {"tokens": [], "texts": [], "confidence": 0.0, "success": False}
        with self._paddle_lock:
            try:
                source = image_path
                if image_input is not None and isinstance(image_input, np.ndarray):
                    source = image_input
                    if source.ndim == 2:
                        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
                try:
                    result = self.paddleocr_reader.ocr(source)
                except Exception:
                    result = self.paddleocr_reader.ocr(image_path)
            except Exception as exc:
                raise RuntimeError(str(exc)) from exc

        tokens = self._parse_paddle_result(result)
        texts = [t["text"] for t in tokens if t["text"]]
        confs = [t["confidence"] for t in tokens]
        return {
            "tokens": tokens,
            "texts": texts,
            "confidence": float(sum(confs) / len(confs)) if confs else 0.0,
            "success": bool(tokens),
        }

    def _run_paddle_with_retry(self, image_path: str, image_input: Optional[np.ndarray]) -> Dict:
        with self._stat_lock:
            self.stats["paddle_attempts"] += 1
        if self.paddleocr_reader is None:
            return {"tokens": [], "texts": [], "confidence": 0.0, "success": False}

        # Try normal call.
        for _attempt in range(self.config.paddle_retry_count + 1):
            try:
                out = self._run_paddle_once(image_path, image_input)
                if out["success"]:
                    with self._stat_lock:
                        self.stats["paddle_successes"] += 1
                    return out
            except Exception as exc:
                logger.warning(
                    "PaddleOCR attempt failed for %s: %s",
                    image_path,
                    exc,
                    extra={"event": "ocr_failure", "engine": "paddleocr", "image_name": Path(image_path).name},
                )

        # Reinitialize with mkldnn disabled and retry once.
        if self.config.disable_mkldnn_on_retry:
            logger.warning(
                "PaddleOCR failed; retrying with MKLDNN disabled for %s",
                image_path,
                extra={"event": "fallback_event", "engine": "paddleocr", "image_name": Path(image_path).name},
            )
            self._init_paddleocr(use_mkldnn=False)
            if self.paddleocr_reader is not None:
                try:
                    out = self._run_paddle_once(image_path, image_input)
                    if out["success"]:
                        with self._stat_lock:
                            self.stats["paddle_successes"] += 1
                        return out
                except Exception as exc:
                    logger.warning(
                        "PaddleOCR MKLDNN-disabled retry failed for %s: %s",
                        image_path,
                        exc,
                        extra={"event": "ocr_failure", "engine": "paddleocr", "image_name": Path(image_path).name},
                    )

        return {"tokens": [], "texts": [], "confidence": 0.0, "success": False}

    @staticmethod
    def clean_text(text_list: List[str]) -> List[str]:
        if not text_list:
            return []
        out = []
        seen = set()
        for text in text_list:
            text = re.sub(r"\s+", " ", str(text).strip())
            text = re.sub(r"[^\x20-\x7E]", "", text)
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    def run_ocr(self, image_path: str, invalidate_cache: bool = False) -> Dict:
        with self._stat_lock:
            self.stats["total_requests"] += 1

        if not self.config.enabled_engines and not (self.config.force_engine or "").strip():
            return {
                "text": "",
                "confidence": 0.0,
                "engine": "disabled",
                "tokens": [],
                "easyocr_text": [],
                "paddleocr_text": [],
                "easyocr_confidence": 0.0,
                "paddleocr_confidence": 0.0,
                "cleaned_text": [],
                "best_ocr_engine": "disabled",
                "best_ocr_text": [],
                "best_ocr_confidence": 0.0,
                "paddle_success_rate": 0.0,
                "easyocr_success_rate": 0.0,
                "fallback_rate": 0.0,
            }

        cache_key = None
        if self.cache is not None:
            prep_sig = ""
            if self.preprocessor is not None and hasattr(self.preprocessor, "config"):
                c = self.preprocessor.config
                prep_sig = (
                    f"prep={int(c.enabled)}{int(c.grayscale)}{int(c.clahe)}"
                    f"{int(c.adaptive_threshold)}{int(c.denoise)}{int(c.morphology_cleanup)}{int(c.deskew)}"
                )
            salt = (
                f"ensemble={self.config.ensemble_enabled}|force={self.config.force_engine}|"
                f"engines={','.join(self.config.enabled_engines)}|{prep_sig}"
            )
            cache_key = self.cache.image_hash(image_path, salt=salt)
            if invalidate_cache:
                pass
            else:
                cached = self.cache.get(cache_key)
                if cached is not None:
                    with self._stat_lock:
                        self.stats["cache_hits"] += 1
                    return cached
                with self._stat_lock:
                    self.stats["cache_misses"] += 1

        image_input = None
        if self.preprocessor is not None and self.config.enabled_engines:
            image_input = self.preprocessor.process_path(image_path)

        forced = (self.config.force_engine or "").lower().strip() or None
        easy = {"tokens": [], "texts": [], "confidence": 0.0, "success": False}
        paddle = {"tokens": [], "texts": [], "confidence": 0.0, "success": False}

        if forced == "easyocr":
            easy = self._run_easyocr(image_path, image_input)
        elif forced == "paddleocr":
            paddle = self._run_paddle_with_retry(image_path, image_input)
            if not paddle["success"]:
                easy = self._run_easyocr(image_path, image_input)
                with self._stat_lock:
                    self.stats["fallback_events"] += 1
        else:
            if "easyocr" in self.config.enabled_engines:
                easy = self._run_easyocr(image_path, image_input)
            if "paddleocr" in self.config.enabled_engines:
                paddle = self._run_paddle_with_retry(image_path, image_input)
            if "paddleocr" in self.config.enabled_engines and not paddle["success"]:
                with self._stat_lock:
                    self.stats["fallback_events"] += 1

        if self.config.ensemble_enabled and forced is None:
            ensemble = fuse_ocr_tokens(easy["tokens"], paddle["tokens"])
            final_engine = "ensemble"
            final_text = ensemble.get("text", "")
            final_conf = float(ensemble.get("confidence", 0.0))
            final_tokens = ensemble.get("tokens", [])
        else:
            candidates = [("easyocr", easy), ("paddleocr", paddle)]
            best_name, best_data = max(candidates, key=lambda x: float(x[1].get("confidence", 0.0)))
            final_engine = best_name
            final_tokens = best_data.get("tokens", [])
            final_text = " ".join(best_data.get("texts", []))
            final_conf = float(best_data.get("confidence", 0.0))
            ensemble = {
                "text": final_text.strip(),
                "confidence": final_conf,
                "engine": final_engine,
                "tokens": final_tokens,
            }

        cleaned = self.clean_text(easy.get("texts", []) + paddle.get("texts", []))
        rates = self.get_success_rates()
        payload = {
            "text": ensemble.get("text", ""),
            "confidence": float(ensemble.get("confidence", 0.0)),
            "engine": ensemble.get("engine", "ensemble"),
            "tokens": ensemble.get("tokens", []),
            "easyocr_text": easy.get("texts", []),
            "paddleocr_text": paddle.get("texts", []),
            "easyocr_confidence": float(easy.get("confidence", 0.0)),
            "paddleocr_confidence": float(paddle.get("confidence", 0.0)),
            "cleaned_text": cleaned,
            "best_ocr_engine": final_engine,
            "best_ocr_text": [t.get("text", "") for t in final_tokens] if final_tokens else cleaned,
            "best_ocr_confidence": final_conf,
            "paddle_success_rate": rates["paddle_success_rate"],
            "easyocr_success_rate": rates["easyocr_success_rate"],
            "fallback_rate": rates["fallback_rate"],
        }

        if self.cache is not None and cache_key is not None:
            self.cache.set(cache_key, payload)

        return payload

    def detect_text_regions(self, image_path: str) -> List[List[int]]:
        """Detect approximate text regions as x, y, w, h bounding boxes."""
        try:
            result = self.run_ocr(image_path)
            boxes = []
            for token in result.get("tokens", []):
                bbox = token.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                w, h = max(1, x2 - x1), max(1, y2 - y1)
                boxes.append([x1, y1, w, h])
            if boxes:
                return boxes
        except Exception:
            pass

        # Lightweight fallback using contour grouping.
        image = cv2.imread(image_path)
        if image is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        grouped = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(grouped, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < 80 or h < 6:
                continue
            if w / float(h + 1e-6) > 20:
                continue
            regions.append([int(x), int(y), int(w), int(h)])
        return regions

    def get_success_rates(self) -> Dict[str, float]:
        with self._stat_lock:
            paddle_attempts = max(1, int(self.stats["paddle_attempts"]))
            easy_attempts = max(1, int(self.stats["easy_attempts"]))
            total_requests = max(1, int(self.stats["total_requests"]))
            return {
                "paddle_success_rate": float(self.stats["paddle_successes"]) / paddle_attempts,
                "easyocr_success_rate": float(self.stats["easy_successes"]) / easy_attempts,
                "fallback_rate": float(self.stats["fallback_events"]) / total_requests,
                "cache_hits": int(self.stats["cache_hits"]),
                "cache_misses": int(self.stats["cache_misses"]),
            }
