from __future__ import annotations

import re


class TextCleaner:
    def clean(self, text: str) -> str:
        text = str(text or "").strip()
        text = re.sub(r"\s+", " ", text)
        text = text.replace("O", "0") if text.isdigit() else text
        return text

    def normalize_numeric(self, text: str) -> str:
        cleaned = self.clean(text)
        cleaned = cleaned.replace(",", "")
        return cleaned

