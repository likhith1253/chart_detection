from .configuration import OCRBackendConfig, OCRConfiguration
from .language import OCRLanguageProfile
from .ocr_cache import OCRCache
from .ocr_factory import OCRFactory, get_ocr_backend
from .paddle_backend import PaddleOCRBackend
from .postprocessing import OCRPostprocessor
from .preprocessing import OCRPreprocessor

__all__ = [
    "OCRBackendConfig",
    "OCRCache",
    "OCRConfiguration",
    "OCRFactory",
    "OCRLanguageProfile",
    "OCRPostprocessor",
    "OCRPreprocessor",
    "PaddleOCRBackend",
    "get_ocr_backend",
]

