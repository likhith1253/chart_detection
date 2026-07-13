"""
Debug pipeline for testing each module independently.
Run this script to verify that all components work before running the main pipeline.
"""

import json
import sys
import logging
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config
from src.classifier.chart_classifier import ChartClassifier
from src.ocr.ocr_engine import OCREngine
from src.extraction.value_extractor import ValueExtractor
from src.utils.evaluator import OCREvaluator

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')


def get_test_image() -> str:
    """Finds the first available image in the raw_images directory."""
    image_dir = config.RAW_IMAGE_DIR
    if not image_dir.exists():
        return ""
    images = (
        list(image_dir.glob("*.png"))
        + list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
    )
    return str(images[0]) if images else ""


def separator(title: str) -> None:
    """Prints a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_classifier(image_path: str) -> dict:
    """Tests the ChartClassifier independently."""
    separator("TEST 1: Chart Classifier")
    try:
        classifier = ChartClassifier()
        result = classifier.classify_chart(image_path)
        print(f"  Image:      {Path(image_path).name}")
        print(f"  Chart Type: {result['chart_type']}")
        
        # Print explicitly extracted geographic metrics
        metrics = result.get('metrics', {})
        if metrics:
            print("  Explicit Metrics:")
            for m_key, m_val in metrics.items():
                if isinstance(m_val, float):
                    print(f"    - {m_key:26}: {m_val:.4f}")
                else:
                    print(f"    - {m_key:26}: {m_val}")
                    
        print(f"  Summary:    {result['summary']}")
        print("  [PASS]")
        return result
    except Exception as e:
        print(f"  [FAIL] Classifier error: {e}")
        return {}


def test_easyocr(ocr_engine: OCREngine, image_path: str) -> list:
    """Tests EasyOCR independently."""
    separator("TEST 2: EasyOCR")
    try:
        texts = ocr_engine.run_easyocr(image_path)
        print(f"  Detected {len(texts)} text regions:")
        for i, t in enumerate(texts):
            print(f"    [{i+1}] {t}")
        print("  [PASS]")
        return texts
    except Exception as e:
        print(f"  [FAIL] EasyOCR error: {e}")
        return []


def test_paddleocr(ocr_engine: OCREngine, image_path: str) -> list:
    """Tests PaddleOCR independently."""
    separator("TEST 3: PaddleOCR")
    try:
        texts = ocr_engine.run_paddleocr(image_path)
        print(f"  Detected {len(texts)} text regions:")
        for i, t in enumerate(texts):
            print(f"    [{i+1}] {t}")
        if texts:
            print("  [PASS]")
        else:
            print("  [WARN] No text detected (PaddleOCR may be unavailable)")
        return texts
    except Exception as e:
        print(f"  [FAIL] PaddleOCR error: {e}")
        return []


def test_text_cleaning(ocr_engine: OCREngine, raw_texts: list) -> list:
    """Tests the text cleaning utility."""
    separator("TEST 4: Text Cleaning")
    try:
        cleaned = OCREngine.clean_text(raw_texts)
        print(f"  Input:   {len(raw_texts)} items")
        print(f"  Cleaned: {len(cleaned)} items")
        for i, t in enumerate(cleaned[:5]):
            print(f"    [{i+1}] {t}")
        print("  [PASS]")
        return cleaned
    except Exception as e:
        print(f"  [FAIL] Text cleaning error: {e}")
        return []



def test_segmenter(image_path: str):
    """Tests ChartElementSegmenter independently and visually."""
    separator("TEST 4: Segmentation (Visual)")
    from src.segmentation.chart_element_segmenter import ChartElementSegmenter
    import cv2
    import config
    
    try:
        segmenter = ChartElementSegmenter()
        segmented = segmenter.segment_elements(image_path)
        
        print("  Found Elements:")
        print(f"    Plot Areas:   {len(segmented.get('plot_area', []))}")
        print(f"    Bars:         {len(segmented.get('bars', []))}")
        print(f"    Points:       {len(segmented.get('points', []))}")
        print(f"    Pie Slices:   {len(segmented.get('pie_slices', []))}")
        print(f"    Axes:         {len(segmented.get('axes', []))}")
        
        # Draw for visualization
        img = cv2.imread(image_path)
        if img is not None:
            # Draw plot area (blue)
            for x, y, w, h in segmented.get("plot_area", []):
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Draw axes (red)
            for [x1, y1, x2, y2] in segmented.get("axes", []):
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
            # Draw bars (green)
            for x, y, w, h in segmented.get("bars", []):
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            # Draw points (yellow)
            for cx, cy in segmented.get("points", []):
                cv2.circle(img, (cx, cy), 5, (0, 255, 255), -1)
                
            # Draw pie slices (purple)
            for c in segmented.get("pie_slices", []):
                cv2.drawContours(img, [c], -1, (255, 0, 255), 2)

            out_dir = config.RESULT_DIR / "plots"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / "debug_segmentation.png"
            cv2.imwrite(str(out_file), img)
            print(f"  [PASS] Saved visual debug to: {out_file}")
            
    except Exception as e:
        print(f"  [FAIL] Segmentation error: {e}")

def test_value_extraction(value_extractor: ValueExtractor, image_path: str, labels: list):
    """Tests raw value extraction independently."""
    separator("TEST 5: Value Extraction")
    try:
        values = value_extractor.extract_bar_values(image_path, labels=labels)
        print(f"  Extracted {len(values)} values")
        for label, val in values.items():
            print(f"    {label}: {val}")
        if values:
            print("  [PASS]")
        else:
            print("  [WARN] No values extracted (image may not be a bar chart)")
        return values
    except Exception as e:
        print(f"  [FAIL] Value extractor error: {e}")
        return {}


def test_evaluator(easyocr_text: list, paddleocr_text: list) -> dict:
    """Tests the OCREvaluator independently."""
    separator("TEST 6: Evaluator")
    try:
        evaluator = OCREvaluator()

        # Create mock ground truth from OCR results for testing
        mock_labels = easyocr_text[:3] + paddleocr_text[:3]
        if not mock_labels:
            mock_labels = ["Test Label 1", "Test Label 2", "Test Label 3"]

        ground_truth = {"labels": mock_labels}

        scores = evaluator.evaluate(
            easyocr_text=easyocr_text,
            paddleocr_text=paddleocr_text,
            ground_truth=ground_truth
        )
        print(f"  EasyOCR score:   {scores['easyocr_score']:.4f}")
        print(f"  PaddleOCR score: {scores['paddleocr_score']:.4f}")
        print("  [PASS]")
        return scores
    except Exception as e:
        print(f"  [FAIL] Evaluator error: {e}")
        return {}


def run_debug():
    """Runs all debugging tests sequentially."""
    print("\n" + "=" * 60)
    print("  CHART RESEARCH PROJECT — DEBUG PIPELINE")
    print("=" * 60)

    image_path = get_test_image()
    if not image_path:
        print("\n[ERROR] No test image found in data/raw_images/")
        print("  Run 'python download_test_images.py' first.")
        return

    print(f"\n  Test image: {image_path}")

    # 1. Classifier
    classification = test_classifier(image_path)

    # 2-3. OCR Engines
    ocr_engine = OCREngine()
    easyocr_text = test_easyocr(ocr_engine, image_path)
    paddleocr_text = test_paddleocr(ocr_engine, image_path)

    # 4. Text Cleaning
    all_text = easyocr_text + paddleocr_text
    cleaned = test_text_cleaning(ocr_engine, all_text)

    # 4b. Segmenter
    test_segmenter(image_path)

    # 5. Value Extractor
    value_extractor = ValueExtractor()
    test_value_extraction(value_extractor, image_path, labels=cleaned[:4])

    # 6. Evaluator
    test_evaluator(easyocr_text, paddleocr_text)

    # Summary
    separator("DEBUG SUMMARY")
    print("  All modules tested. Review output above for [PASS]/[FAIL]/[WARN].")
    print("  If all pass, run 'python main.py' for the full pipeline.\n")


if __name__ == "__main__":
    run_debug()
