"""
OCR ensemble fusion with token-level alignment and confidence arbitration.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def correct_numeric_token(token: str) -> str:
    cleaned = token.strip()
    if not cleaned:
        return cleaned

    # Apply replacements only when token is mostly numeric-like.
    numeric_like = bool(re.fullmatch(r"[\d\.,:%$OoIlSB\-+]+", cleaned))
    if not numeric_like:
        return cleaned

    table = str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "S": "5", "B": "8"})
    return cleaned.translate(table)


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _token_match(a: Dict, b: Dict, iou_threshold: float = 0.3) -> bool:
    text_a = normalize_whitespace(a.get("text", "")).lower()
    text_b = normalize_whitespace(b.get("text", "")).lower()
    if text_a == text_b and text_a:
        return True

    if text_a and text_b and (text_a in text_b or text_b in text_a):
        if _bbox_iou(tuple(a.get("bbox", (0, 0, 0, 0))), tuple(b.get("bbox", (0, 0, 0, 0)))) >= iou_threshold:
            return True

    return _bbox_iou(tuple(a.get("bbox", (0, 0, 0, 0))), tuple(b.get("bbox", (0, 0, 0, 0)))) >= iou_threshold


def fuse_ocr_tokens(
    easy_tokens: List[Dict],
    paddle_tokens: List[Dict],
) -> Dict:
    """Fuse OCR outputs via alignment, confidence selection, and dedupe."""
    merged: List[Dict] = []
    consumed = set()

    for idx_e, e in enumerate(easy_tokens):
        best_idx = None
        best_conf = -1.0
        for idx_p, p in enumerate(paddle_tokens):
            if idx_p in consumed:
                continue
            if _token_match(e, p):
                candidate_conf = float(p.get("confidence", 0.0))
                if candidate_conf > best_conf:
                    best_idx = idx_p
                    best_conf = candidate_conf

        if best_idx is None:
            token = dict(e)
        else:
            consumed.add(best_idx)
            p = paddle_tokens[best_idx]
            token = dict(e if float(e.get("confidence", 0.0)) >= float(p.get("confidence", 0.0)) else p)

        token["text"] = correct_numeric_token(normalize_whitespace(str(token.get("text", ""))))
        merged.append(token)

    for idx_p, p in enumerate(paddle_tokens):
        if idx_p in consumed:
            continue
        token = dict(p)
        token["text"] = correct_numeric_token(normalize_whitespace(str(token.get("text", ""))))
        merged.append(token)

    # Duplicate suppression with bbox+text matching.
    deduped: List[Dict] = []
    for token in sorted(merged, key=lambda t: (t.get("bbox", [0, 0, 0, 0])[1], t.get("bbox", [0, 0, 0, 0])[0])):
        text_norm = normalize_whitespace(str(token.get("text", ""))).lower()
        if not text_norm:
            continue
        duplicate = False
        for existing in deduped:
            if _token_match(token, existing) and normalize_whitespace(str(existing.get("text", ""))).lower() == text_norm:
                if float(token.get("confidence", 0.0)) > float(existing.get("confidence", 0.0)):
                    existing.update(token)
                duplicate = True
                break
        if not duplicate:
            deduped.append(token)

    confidences = [float(t.get("confidence", 0.0)) for t in deduped]
    text = " ".join([str(t.get("text", "")) for t in deduped]).strip()
    return {
        "text": normalize_whitespace(text),
        "confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
        "engine": "ensemble",
        "tokens": deduped,
    }
