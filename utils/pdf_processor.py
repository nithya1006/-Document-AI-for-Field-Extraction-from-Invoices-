# PDF to image conversion
"""
PDF Processor - Converts PDF documents to images for processing
"""

import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF to image conversion and preprocessing"""
    
    def __init__(self, dpi: int = 300, max_size: int = 2048):
        self.dpi = dpi
        self.max_size = max_size
    
    def pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert PDF to list of images (one per page)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of numpy arrays (images)
        """
        try:
            doc = fitz.open(pdf_path)
            images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Render page to image
                zoom = self.dpi / 72  # Standard PDF DPI is 72
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to numpy array
                img = np.frombuffer(pix.samples, dtype=np.uint8)
                img = img.reshape(pix.height, pix.width, pix.n)
                
                # Convert RGBA to RGB if needed
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                
                images.append(img)
                logger.info(f"Processed page {page_num + 1}/{len(doc)}")
            
            doc.close()
            return images
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
    
    def preprocess_image(self, image: np.ndarray, 
                        operations: List[str] = None) -> np.ndarray:
        """
        Apply preprocessing operations to image
        
        Args:
            image: Input image
            operations: List of operations to apply
            
        Returns:
            Preprocessed image
        """
        if operations is None:
            operations = ["denoise", "deskew"]
        
        processed = image.copy()
        
        for op in operations:
            if op == "grayscale":
                if len(processed.shape) == 3:
                    processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            
            elif op == "denoise":
                if len(processed.shape) == 3:
                    processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)
                else:
                    processed = cv2.fastNlMeansDenoising(processed, None, 10, 7, 21)
            
            elif op == "deskew":
                processed = self._deskew(processed)
            
            elif op == "adaptive_threshold":
                if len(processed.shape) == 3:
                    gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
                else:
                    gray = processed
                processed = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                    cv2.THRESH_BINARY, 11, 2
                )
            
            elif op == "resize":
                processed = self._resize_image(processed)
        
        return processed
    
    def _deskew(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate skew angle
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) == 0:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        
        # Rotate image
        if abs(angle) > 0.5:  # Only deskew if angle is significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        
        return image
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """Resize image if larger than max_size"""
        h, w = image.shape[:2]
        
        if max(h, w) > self.max_size:
            scale = self.max_size / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized
        
        return image
    
    def extract_regions(self, image: np.ndarray, 
                       region_type: str = "text") -> List[Tuple[int, int, int, int]]:
        """
        Extract bounding boxes for different regions
        
        Args:
            image: Input image
            region_type: Type of region to extract ('text', 'signature', 'stamp')
            
        Returns:
            List of bounding boxes (x, y, w, h)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter based on region type
            if region_type == "text" and w > 20 and h > 10:
                regions.append((x, y, w, h))
            elif region_type in ["signature", "stamp"] and w > 50 and h > 30:
                regions.append((x, y, w, h))
        
        return regions


def test_pdf_processor():
    """Test the PDF processor"""
    processor = PDFProcessor(dpi=300)
    
    # Test with a sample PDF (you'll need to provide a path)
    pdf_path = "data/raw/sample_invoice.pdf"
    
    if os.path.exists(pdf_path):
        images = processor.pdf_to_images(pdf_path)
        print(f"Converted {len(images)} pages")
        
        # Preprocess first page
        processed = processor.preprocess_image(images[0])
        print(f"Preprocessed image shape: {processed.shape}")
    else:
        print(f"Sample PDF not found at {pdf_path}")


if __name__ == "__main__":
    test_pdf_processor()