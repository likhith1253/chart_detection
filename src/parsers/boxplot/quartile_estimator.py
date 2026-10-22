from __future__ import annotations

from .geometry import BoxGeometry, QuartileGeometry


class QuartileEstimator:
    def estimate(self, box: BoxGeometry, orientation: str = "vertical") -> QuartileGeometry:
        span = box.height if orientation == "vertical" else box.width
        lower = 0.25 * span
        median = 0.5 * span
        upper = 0.75 * span
        if orientation == "vertical":
            return QuartileGeometry(lower_quartile=box.y2 - upper, median=box.y2 - median, upper_quartile=box.y2 - lower)
        return QuartileGeometry(lower_quartile=box.x1 + lower, median=box.x1 + median, upper_quartile=box.x1 + upper)

