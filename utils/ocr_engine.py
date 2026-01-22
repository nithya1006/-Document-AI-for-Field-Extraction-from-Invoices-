"""
OCR Engine - Optimized & Production Ready
Fast, accurate, and batch-safe for large datasets
"""

import numpy as np
from typing import List, Tuple
import logging
from dataclasses import dataclass
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    bbox: Tuple[int, int, int, int]
    confidence: float


class OCREngine:
    def __init__(self, languages: List[str] = None, use_gpu: bool = False):
        self.languages = languages or ["en"]
        self.use_gpu = use_gpu
        self.ocr_instance = None
        self._initialize_engine()

    def _initialize_engine(self):
        try:
            import easyocr
            logger.info(f"Initializing EasyOCR with languages: {self.languages}")

            self.ocr_instance = easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False
            )

            logger.info("EasyOCR initialized successfully")
            if not self.use_gpu:
                logger.info("Running in CPU mode")

        except Exception as e:
            logger.error(f"OCR init failed: {e}")
            raise RuntimeError("EasyOCR initialization failed")

    # ✅ FIX 1: Correct resize method
    def _resize_for_ocr(self, image: np.ndarray, max_dim: int = 1000) -> np.ndarray:
        h, w = image.shape[:2]
        scale = max_dim / max(h, w)

        if scale < 1:
            image = cv2.resize(
                image,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )
        return image

    def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        if self.ocr_instance is None:
            raise RuntimeError("OCR engine not initialized")

        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # ✅ FIX 2: Resize image BEFORE OCR
        image = self._resize_for_ocr(image)

        h, w = image.shape[:2]

        # ✅ FIX 3: Region-based OCR (HUGE speed boost)
        regions = [
            image[0:int(0.3 * h), :],            # Header
            image[int(0.3 * h):int(0.7 * h), :], # Body
            image[int(0.7 * h):h, :]             # Footer
        ]

        results: List[OCRResult] = []

        try:
            for region in regions:
                detections = self.ocr_instance.readtext(
                    region,
                    detail=1,
                    paragraph=False,
                    decoder="greedy",
                    beamWidth=1,
                    text_threshold=0.6,
                    low_text=0.4,
                    link_threshold=0.4
                )

                for bbox_points, text, confidence in detections:
                    if not text.strip():
                        continue

                    x_coords = [p[0] for p in bbox_points]
                    y_coords = [p[1] for p in bbox_points]

                    bbox = (
                        int(min(x_coords)),
                        int(min(y_coords)),
                        int(max(x_coords)),
                        int(max(y_coords))
                    )

                    results.append(OCRResult(
                        text=text.strip(),
                        bbox=bbox,
                        confidence=float(confidence)
                    ))

            logger.info(f"Extracted {len(results)} text elements")

        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            raise

        return results

    def get_full_text(self, image: np.ndarray) -> str:
        results = self.extract_text(image)
        return " ".join(r.text for r in results)

    def extract_with_confidence(
        self,
        image: np.ndarray,
        min_confidence: float = 0.5
    ) -> List[OCRResult]:
        results = self.extract_text(image)
        return [r for r in results if r.confidence >= min_confidence]

def test_ocr():
    """Test OCR engine with a simple image"""
    
    # Create test image with text
    test_image = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "Test Invoice 12345", (50, 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print("Initializing OCR engine...")
    ocr = OCREngine(languages=["en"])
    
    print("Extracting text...")
    results = ocr.extract_text(test_image)
    
    print(f"\nFound {len(results)} text elements:")
    for r in results:
        print(f"  Text: '{r.text}'")
        print(f"  Confidence: {r.confidence:.2%}")
        print(f"  BBox: {r.bbox}\n")


if __name__ == "__main__":
    test_ocr()