"""
Utility module for evaluating OCR accuracy.
Compares results from different OCR engines against ground truth.
"""

from difflib import SequenceMatcher
from typing import Dict, List, Union


class OCREvaluator:
    """Evaluates the accuracy of extracted OCR text against ground truth."""

    def _calculate_similarity(self, extracted: List[str], ground_truth_labels: List[str]) -> float:
        """
        Calculates a similarity score between 0.0 and 1.0.
        Joins the text lists into single strings and compares them using SequenceMatcher.
        
        Args:
            extracted: List of strings extracted by an OCR engine.
            ground_truth_labels: List of expected ground truth strings.
            
        Returns:
            Float representing the similarity ratio.
        """
        if not extracted and not ground_truth_labels:
            return 1.0  # Both empty matches perfectly
        
        if not extracted or not ground_truth_labels:
            return 0.0  # One is empty while the other is not
            
        text1 = " ".join(extracted).lower()
        text2 = " ".join(ground_truth_labels).lower()
        
        return SequenceMatcher(None, text1, text2).ratio()

    def evaluate(self, easyocr_text: List[str], paddleocr_text: List[str], ground_truth: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Evaluates EasyOCR and PaddleOCR results against ground truth.
        
        Args:
            easyocr_text: Text extracted by EasyOCR.
            paddleocr_text: Text extracted by PaddleOCR.
            ground_truth: Ground truth dictionary containing 'labels' key.
            
        Returns:
            Dictionary with the similarity scores for both engines.
        """
        gt_labels = ground_truth.get("labels", [])
        
        easyocr_score = self._calculate_similarity(easyocr_text, gt_labels)
        paddleocr_score = self._calculate_similarity(paddleocr_text, gt_labels)
        
        return {
            "easyocr_score": round(easyocr_score, 4),
            "paddleocr_score": round(paddleocr_score, 4)
        }
