# Signature/stamp detection
"""
Signature and Stamp Detector - Uses computer vision to detect signatures and stamps
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignatureStampDetector:
    """Detect signatures and stamps using traditional CV and optionally YOLO"""
    
    def __init__(self, use_yolo: bool = False, model_path: str = None):
        self.use_yolo = use_yolo
        self.model_path = model_path
        self.yolo_model = None
        
        if use_yolo and model_path:
            self._load_yolo_model()
    
    def _load_yolo_model(self):
        """Load YOLO model for detection"""
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(self.model_path)
            logger.info(f"Loaded YOLO model from {self.model_path}")
        except Exception as e:
            logger.warning(f"Could not load YOLO model: {e}")
            self.use_yolo = False
    
    def detect_signature(self, image: np.ndarray, 
                        confidence_threshold: float = 0.5) -> Dict:
        """
        Detect signature in image
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Dictionary with presence flag and bounding box
        """
        if self.use_yolo and self.yolo_model:
            return self._detect_with_yolo(image, "signature", confidence_threshold)
        else:
            return self._detect_signature_traditional(image, confidence_threshold)
    
    def detect_stamp(self, image: np.ndarray, 
                    confidence_threshold: float = 0.5) -> Dict:
        """
        Detect stamp in image
        
        Args:
            image: Input image
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Dictionary with presence flag and bounding box
        """
        if self.use_yolo and self.yolo_model:
            return self._detect_with_yolo(image, "stamp", confidence_threshold)
        else:
            return self._detect_stamp_traditional(image, confidence_threshold)
    
    def _detect_with_yolo(self, image: np.ndarray, 
                         object_type: str,
                         confidence_threshold: float) -> Dict:
        """Detect using YOLO model"""
        try:
            results = self.yolo_model(image, conf=confidence_threshold)
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Assuming class 0 is signature, class 1 is stamp
                    if (object_type == "signature" and cls == 0) or \
                       (object_type == "stamp" and cls == 1):
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        return {
                            "present": True,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                            "confidence": conf
                        }
            
            return {"present": False, "bbox": None, "confidence": 0.0}
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return {"present": False, "bbox": None, "confidence": 0.0}
    
    def _detect_signature_traditional(self, image: np.ndarray,
                                     confidence_threshold: float) -> Dict:
        """
        Detect signature using traditional CV methods
        
        Signatures are typically:
        - Handwritten, cursive text
        - Black ink on white background
        - Horizontal orientation
        - Medium size (not too small, not too large)
        - Often in lower part of document
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        h, w = image.shape[:2]
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filter based on signature characteristics
            area = cw * ch
            aspect_ratio = cw / ch if ch > 0 else 0
            
            # Signatures are typically:
            # - Width 100-500 pixels
            # - Height 30-150 pixels
            # - Aspect ratio 2:1 to 6:1
            # - In lower 60% of document
            
            if (100 < cw < 500 and 
                30 < ch < 150 and 
                2 < aspect_ratio < 6 and
                y > h * 0.4):  # Lower part of document
                
                # Calculate density (how much ink vs white space)
                roi = binary[y:y+ch, x:x+cw]
                density = np.sum(roi > 0) / (cw * ch)
                
                # Signatures typically have 10-40% density
                if 0.1 < density < 0.4:
                    confidence = self._calculate_signature_confidence(roi, aspect_ratio, density)
                    
                    if confidence >= confidence_threshold:
                        candidates.append({
                            "bbox": [x, y, x+cw, y+ch],
                            "confidence": confidence
                        })
        
        # Return best candidate
        if candidates:
            best = max(candidates, key=lambda x: x["confidence"])
            return {
                "present": True,
                "bbox": best["bbox"],
                "confidence": best["confidence"]
            }
        
        return {"present": False, "bbox": None, "confidence": 0.0}
    
    def _detect_stamp_traditional(self, image: np.ndarray,
                                 confidence_threshold: float) -> Dict:
        """
        Detect stamp using traditional CV methods
        
        Stamps are typically:
        - Circular or rectangular
        - High density (lots of ink)
        - Red or blue color (if color image)
        - Contains text in circular pattern
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        h, w = image.shape[:2]
        
        for contour in contours:
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # Filter based on stamp characteristics
            area = cv2.contourArea(contour)
            aspect_ratio = cw / ch if ch > 0 else 0
            
            # Stamps are typically:
            # - 50-300 pixels in both dimensions
            # - Roughly square (aspect ratio 0.7-1.3)
            # - High density
            
            if (50 < cw < 300 and 
                50 < ch < 300 and 
                0.7 < aspect_ratio < 1.3):
                
                # Calculate density
                roi = binary[y:y+ch, x:x+cw]
                density = np.sum(roi > 0) / (cw * ch)
                
                # Stamps typically have high density (40-80%)
                if 0.3 < density < 0.8:
                    # Check for circular shape
                    perimeter = cv2.arcLength(contour, True)
                    circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                    
                    confidence = self._calculate_stamp_confidence(
                        roi, aspect_ratio, density, circularity
                    )
                    
                    if confidence >= confidence_threshold:
                        candidates.append({
                            "bbox": [x, y, x+cw, y+ch],
                            "confidence": confidence
                        })
        
        # If color image, check for red/blue stamps
        if len(image.shape) == 3 and candidates:
            candidates = self._filter_by_color(image, candidates)
        
        # Return best candidate
        if candidates:
            best = max(candidates, key=lambda x: x["confidence"])
            return {
                "present": True,
                "bbox": best["bbox"],
                "confidence": best["confidence"]
            }
        
        return {"present": False, "bbox": None, "confidence": 0.0}
    
    def _calculate_signature_confidence(self, roi: np.ndarray, 
                                       aspect_ratio: float,
                                       density: float) -> float:
        """Calculate confidence score for signature"""
        confidence = 0.5  # Base confidence
        
        # Ideal aspect ratio 3-4
        if 3 <= aspect_ratio <= 4:
            confidence += 0.2
        elif 2 <= aspect_ratio <= 6:
            confidence += 0.1
        
        # Ideal density 15-30%
        if 0.15 <= density <= 0.30:
            confidence += 0.2
        elif 0.10 <= density <= 0.40:
            confidence += 0.1
        
        # Check for connected components (signature should have some)
        num_labels, _ = cv2.connectedComponents(roi)
        if 5 <= num_labels <= 20:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _calculate_stamp_confidence(self, roi: np.ndarray,
                                   aspect_ratio: float,
                                   density: float,
                                   circularity: float) -> float:
        """Calculate confidence score for stamp"""
        confidence = 0.5  # Base confidence
        
        # Ideal aspect ratio ~1 (square/circular)
        if 0.9 <= aspect_ratio <= 1.1:
            confidence += 0.2
        elif 0.7 <= aspect_ratio <= 1.3:
            confidence += 0.1
        
        # Ideal density 40-70%
        if 0.4 <= density <= 0.7:
            confidence += 0.2
        elif 0.3 <= density <= 0.8:
            confidence += 0.1
        
        # High circularity indicates stamp
        if circularity > 0.7:
            confidence += 0.2
        elif circularity > 0.5:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _filter_by_color(self, image: np.ndarray, 
                        candidates: List[Dict]) -> List[Dict]:
        """Filter candidates by color (stamps are often red/blue)"""
        filtered = []
        
        for candidate in candidates:
            x1, y1, x2, y2 = candidate["bbox"]
            roi = image[y1:y2, x1:x2]
            
            # Convert to HSV
            hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
            
            # Check for red or blue dominant color
            # Red: H in [0-10] or [160-180]
            # Blue: H in [100-130]
            
            red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255)) + \
                      cv2.inRange(hsv, (160, 50, 50), (180, 255, 255))
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            
            red_ratio = np.sum(red_mask > 0) / red_mask.size
            blue_ratio = np.sum(blue_mask > 0) / blue_mask.size
            
            # Boost confidence if red or blue
            if red_ratio > 0.2 or blue_ratio > 0.2:
                candidate["confidence"] = min(candidate["confidence"] + 0.2, 1.0)
                filtered.append(candidate)
            else:
                filtered.append(candidate)
        
        return filtered
    
    def calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """
        Calculate IoU between two bounding boxes
        
        Args:
            box1, box2: [x1, y1, x2, y2]
            
        Returns:
            IoU score
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def test_detector():
    """Test signature and stamp detection"""
    # Create a test image
    test_image = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Draw a fake signature (scribble)
    pts = np.array([[100, 600], [150, 580], [200, 590], [250, 570], [300, 580]], np.int32)
    cv2.polylines(test_image, [pts], False, (0, 0, 0), 2)
    
    # Draw a fake stamp (circle)
    cv2.circle(test_image, (500, 650), 50, (255, 0, 0), 3)
    cv2.putText(test_image, "DEALER", (470, 655), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    detector = SignatureStampDetector()
    
    sig_result = detector.detect_signature(test_image)
    stamp_result = detector.detect_stamp(test_image)
    
    print("Signature Detection:")
    print(f"  Present: {sig_result['present']}")
    print(f"  BBox: {sig_result['bbox']}")
    print(f"  Confidence: {sig_result['confidence']:.2f}")
    
    print("\nStamp Detection:")
    print(f"  Present: {stamp_result['present']}")
    print(f"  BBox: {stamp_result['bbox']}")
    print(f"  Confidence: {stamp_result['confidence']:.2f}")


if __name__ == "__main__":
    test_detector()