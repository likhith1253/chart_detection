from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.detection.model import AbstractDetector, DetectionPrediction
from src.detection.runtime import DetectionRunner, DetectionVisualizer, apply_nms, build_detection_predictions


class StubDetector(AbstractDetector):
    def load(self) -> None:
        return None

    def predict(self, image):
        return [
            DetectionPrediction(bbox=[1, 1, 20, 20], confidence=0.95, class_name="chart"),
            DetectionPrediction(bbox=[2, 2, 19, 19], confidence=0.90, class_name="chart"),
        ]


def _write_image(path: Path, size: tuple[int, int] = (64, 64)) -> None:
    Image.new("RGB", size, color="white").save(path)


def test_nms_and_prediction_conversion() -> None:
    raw = [
        DetectionPrediction(bbox=[0, 0, 10, 10], confidence=0.9, class_name="chart"),
        DetectionPrediction(bbox=[1, 1, 9, 9], confidence=0.8, class_name="chart"),
    ]
    preds = apply_nms(raw, iou_threshold=0.5)
    assert len(preds) == 1

    converted = build_detection_predictions(raw, backend_name="stub", image_id="img1")
    assert converted[0].backend_name == "stub"
    assert converted[0].image_id == "img1"


def test_runner_on_single_image_and_visualizer(tmp_path) -> None:
    img_path = tmp_path / "page.png"
    _write_image(img_path)
    runner = DetectionRunner(detector=StubDetector(), backend_name="stub")

    preds, crops = runner.run_on_image(img_path, save_crops_dir=tmp_path / "crops")
    assert preds
    assert crops
    assert preds[0].crop is not None
    assert preds[0].backend_name == "stub"
    assert (tmp_path / "crops").exists()

    vis = DetectionVisualizer()
    annotated = vis.save_annotated_page(Image.open(img_path), preds, tmp_path / "annotated.png")
    assert annotated.exists()


def test_runner_on_folder_and_manifest(tmp_path) -> None:
    folder = tmp_path / "pages"
    folder.mkdir()
    _write_image(folder / "a.png")
    _write_image(folder / "b.png")
    runner = DetectionRunner(detector=StubDetector(), backend_name="stub")
    preds, crops = runner.run_on_folder(folder)
    assert len(preds) == 2
    assert len(crops) == 2

    # Use a tiny manifest so the test does not depend on (or scan) the real, git-ignored benchmark data.
    manifest = tmp_path / "benchmark_manifest.csv"
    manifest.write_text(
        "image_id,dataset,image_path,chart_type,split,license,width,height\n"
        f"a,chartqa,{folder / 'a.png'},bar,train,unknown,64,64\n"
        f"b,chartqa,{folder / 'b.png'},line,train,unknown,64,64\n",
        encoding="utf-8",
    )
    preds2, crops2 = runner.run_on_manifest(split="train", manifest_csv=manifest)
    assert preds2
    assert crops2
