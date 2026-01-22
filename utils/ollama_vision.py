"""
Ollama Vision Engine for Invoice Text Extraction
Uses local LLaVA model for document understanding
"""

import base64
import json
import requests
import numpy as np
import cv2
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result from Ollama vision extraction"""
    raw_text: str
    fields: Dict
    confidence: float


class OllamaVisionExtractor:
    """
    Extract text and fields from invoices using Ollama's vision models.
    Uses LLaVA or similar multimodal models for document understanding.
    """

    def __init__(
        self,
        model: str = "llama3.2-vision",
        base_url: str = "http://localhost:11434",
        timeout: int = 120
    ):
        """
        Initialize the Ollama vision extractor.

        Args:
            model: Ollama vision model name (llava, llava-llama3, bakllava, etc.)
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._verify_connection()

    def _verify_connection(self):
        """Verify Ollama is running and model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = response.json().get('models', [])
            model_names = [m['name'].split(':')[0] for m in models]

            if self.model not in model_names and f"{self.model}:latest" not in [m['name'] for m in models]:
                available = ', '.join(model_names) if model_names else 'none'
                raise RuntimeError(
                    f"Model '{self.model}' not found. Available models: {available}. "
                    f"Pull it with: ollama pull {self.model}"
                )

            logger.info(f"Ollama connected. Using model: {self.model}")

        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Cannot connect to Ollama. Make sure it's running with: ollama serve"
            )

    def _encode_image(self, image: np.ndarray) -> str:
        """Convert numpy image to base64 string"""
        # Ensure image is in RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Convert to BGR for cv2.imencode
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Resize if too large (max 1024px on longest side for efficiency)
        h, w = image_bgr.shape[:2]
        max_dim = 1024
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image_bgr = cv2.resize(
                image_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA
            )

        # Encode to PNG
        _, buffer = cv2.imencode('.png', image_bgr)
        return base64.b64encode(buffer).decode('utf-8')

    def _call_ollama(self, prompt: str, image_b64: str) -> str:
        """Make API call to Ollama with image"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
            "options": {
                "temperature": 0.1,  # Low temperature for consistent extraction
                "num_predict": 2048
            }
        }

        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout
        )
        response.raise_for_status()

        return response.json().get('response', '')

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract all visible text from an invoice image.

        Args:
            image: Invoice image as numpy array (RGB)

        Returns:
            Extracted text as string
        """
        image_b64 = self._encode_image(image)

        prompt = """Look at this invoice/document image carefully.
Extract ALL visible text exactly as it appears, preserving the layout as much as possible.
Include headers, labels, values, numbers, dates, addresses, and any other text you can see.
Do not add any commentary or explanation - just output the raw text content."""

        logger.info("Extracting text from image using Ollama vision...")
        text = self._call_ollama(prompt, image_b64)
        logger.info(f"Extracted {len(text)} characters of text")

        return text

    def extract_invoice_fields(self, image: np.ndarray) -> Dict:
        """
        Extract structured fields from an invoice image.

        Args:
            image: Invoice image as numpy array (RGB)

        Returns:
            Dictionary with extracted fields
        """
        image_b64 = self._encode_image(image)

        prompt = """Analyze this invoice/document image and extract the following fields.
Return ONLY a valid JSON object with these exact keys:

{
    "dealer_name": "The company/dealer/vendor name",
    "model_name": "Product or model name/number if visible",
    "horse_power": "HP value as number if this is vehicle/equipment related, otherwise null",
    "asset_cost": "Total cost/price as number (without currency symbol)",
    "invoice_number": "Invoice or document number if visible",
    "date": "Invoice date if visible",
    "has_signature": true or false,
    "has_stamp": true or false
}

Important:
- Use null for fields you cannot find
- For numbers, provide just the numeric value without currency symbols or units
- For horse_power, extract just the number (e.g., "45" not "45 HP")
- Look carefully for signatures (handwritten marks) and stamps (official seals/marks)
- Return ONLY the JSON, no other text"""

        logger.info("Extracting invoice fields using Ollama vision...")
        response = self._call_ollama(prompt, image_b64)

        # Parse JSON from response
        try:
            # Try to find JSON in the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1

            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                fields = json.loads(json_str)
            else:
                logger.warning("No JSON found in response, returning empty fields")
                fields = {}

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            fields = {}

        # Normalize the fields
        normalized = {
            'dealer_name': fields.get('dealer_name'),
            'model_name': fields.get('model_name'),
            'horse_power': self._parse_number(fields.get('horse_power')),
            'asset_cost': self._parse_number(fields.get('asset_cost')),
            'invoice_number': fields.get('invoice_number'),
            'date': fields.get('date'),
            'has_signature': bool(fields.get('has_signature', False)),
            'has_stamp': bool(fields.get('has_stamp', False))
        }

        logger.info(f"Extracted fields: {list(k for k, v in normalized.items() if v)}")
        return normalized

    def extract_all(self, image: np.ndarray) -> ExtractionResult:
        """
        Extract both raw text and structured fields from an invoice.

        Args:
            image: Invoice image as numpy array (RGB)

        Returns:
            ExtractionResult with raw_text, fields, and confidence
        """
        # Get raw text first
        raw_text = self.extract_text(image)

        # Get structured fields
        fields = self.extract_invoice_fields(image)

        # Calculate confidence based on how many fields were extracted
        total_fields = 6  # dealer, model, hp, cost, signature, stamp
        found_fields = sum([
            1 if fields.get('dealer_name') else 0,
            1 if fields.get('model_name') else 0,
            1 if fields.get('horse_power') is not None else 0,
            1 if fields.get('asset_cost') is not None else 0,
            1 if fields.get('has_signature') else 0,
            1 if fields.get('has_stamp') else 0
        ])
        confidence = found_fields / total_fields

        return ExtractionResult(
            raw_text=raw_text,
            fields=fields,
            confidence=confidence
        )

    def _parse_number(self, value) -> Optional[float]:
        """Parse a number from various formats"""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            return float(value)

        if isinstance(value, str):
            # Remove common formatting
            cleaned = value.replace(',', '').replace('$', '').replace('₹', '')
            cleaned = cleaned.replace('Rs', '').replace('Rs.', '').strip()

            # Extract number
            import re
            match = re.search(r'[\d.]+', cleaned)
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    pass

        return None


def test_ollama_vision():
    """Test the Ollama vision extractor"""
    print("Testing Ollama Vision Extractor...")

    # Create a simple test image
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255

    # Add some text to the image
    cv2.putText(test_image, "INVOICE #12345", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(test_image, "ABC Company Ltd", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(test_image, "Total: $1,500.00", (50, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    try:
        extractor = OllamaVisionExtractor(model="llava")

        print("\n1. Extracting raw text...")
        text = extractor.extract_text(test_image)
        print(f"Raw text:\n{text}\n")

        print("2. Extracting structured fields...")
        fields = extractor.extract_invoice_fields(test_image)
        print(f"Fields: {json.dumps(fields, indent=2)}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    test_ollama_vision()
