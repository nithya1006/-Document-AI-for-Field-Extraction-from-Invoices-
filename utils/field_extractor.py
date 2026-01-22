"""
Field Extractor - Extracts structured fields from OCR results
Generalized to work with ANY invoice type (retail, industrial, tractor loans, etc.)
Production-ready with comprehensive error handling
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from rapidfuzz import fuzz, process
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FieldExtractor:
    """Extract structured fields from OCR text - works with any invoice type"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.dealer_master = []  # Vendor/dealer master list
        self.model_master = []   # Product/model master list
    
    def _default_config(self) -> Dict:
        """Default configuration for field extraction"""
        return {
            "dealer_name": {
                "keywords": ["dealer", "firm", "seller", "vendor", "proprietor", "name", 
                           "from", "supplier", "company", "store", "shop"],
                "threshold": 0.85  # Fuzzy match threshold
            },
            "model_name": {
                "keywords": ["model", "variant", "type", "product", "item", "description",
                           "tractor", "vehicle", "asset"],
            },
            "horse_power": {
                "keywords": ["hp", "horse power", "horsepower", "h.p", "h.p.", "power"],
                "pattern": r"(\d+\.?\d*)\s*(?:hp|h\.?p\.?|horse\s*power|bhp)",
            },
            "asset_cost": {
                "keywords": ["cost", "price", "amount", "total", "value", "rs", "₹", "rupees",
                           "inr", "grand total", "net amount", "payable"],
                "pattern": r"(?:rs\.?|₹|inr)?\s*(\d+(?:,\d{2,3})*(?:\.\d{2})?)",
            }
        }
    
    def load_master_data(self, dealer_file: str = None, model_file: str = None):
        """
        Load master dealer and model lists for fuzzy matching
        
        Args:
            dealer_file: Path to file with dealer names (one per line)
            model_file: Path to file with model names (one per line)
        """
        if dealer_file:
            try:
                with open(dealer_file, 'r', encoding='utf-8') as f:
                    self.dealer_master = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(self.dealer_master)} dealers from master list")
            except Exception as e:
                logger.warning(f"Could not load dealer master: {e}")
        
        if model_file:
            try:
                with open(model_file, 'r', encoding='utf-8') as f:
                    self.model_master = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded {len(self.model_master)} models from master list")
            except Exception as e:
                logger.warning(f"Could not load model master: {e}")
    
    def extract_all_fields(self, ocr_results: List, full_text: str = None) -> Tuple[Dict, float]:
        """
        Extract all fields from OCR results
        
        Args:
            ocr_results: List of OCRResult objects
            full_text: Optional full text string
            
        Returns:
            Tuple of (extracted_fields_dict, overall_confidence)
        """
        if full_text is None:
            full_text = ' '.join([r.text for r in ocr_results])
        
        # Extract each field
        dealer_name, dealer_conf = self.extract_dealer_name(ocr_results, full_text)
        model_name, model_conf = self.extract_model_name(ocr_results, full_text)
        horse_power, hp_conf = self.extract_horse_power(ocr_results, full_text)
        asset_cost, cost_conf = self.extract_asset_cost(ocr_results, full_text)
        
        fields = {
            "dealer_name": dealer_name,
            "model_name": model_name,
            "horse_power": horse_power,
            "asset_cost": asset_cost
        }
        
        # Calculate overall confidence (weighted average)
        confidences = [dealer_conf, model_conf, hp_conf, cost_conf]
        overall_confidence = sum(confidences) / len(confidences)
        
        logger.info(f"Extracted fields with confidence {overall_confidence:.2%}")
        
        return fields, overall_confidence
    
    def extract_dealer_name(self, ocr_results: List, full_text: str) -> Tuple[str, float]:
        """
        Extract dealer/vendor/company name
        Works for: retail shops, dealers, suppliers, vendors
        """
        keywords = self.config["dealer_name"]["keywords"]
        threshold = self.config["dealer_name"]["threshold"]
        
        candidates = []
        
        # Strategy 1: Look for text near keywords
        lines = full_text.split('\n')
        for i, line in enumerate(lines):
            line_lower = line.lower()
            for keyword in keywords:
                if keyword in line_lower:
                    # Extract nearby text
                    context_lines = lines[max(0, i-1):min(i+3, len(lines))]
                    candidate = ' '.join(context_lines).strip()
                    if len(candidate) > 5:  # Minimum length
                        candidates.append(candidate)
        
        # Strategy 2: Look for proper nouns in top 20% of document
        top_results = ocr_results[:max(10, len(ocr_results)//5)]
        for result in top_results:
            text = result.text.strip()
            # Check if starts with capital and has multiple words
            if text and text[0].isupper() and len(text.split()) >= 2:
                candidates.append(text)
        
        # Strategy 3: Fuzzy match against master list
        if self.dealer_master and candidates:
            best_match = None
            best_score = 0
            
            for candidate in candidates:
                match = process.extractOne(
                    candidate,
                    self.dealer_master,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=threshold * 100
                )
                if match and match[1] > best_score:
                    best_match = match[0]
                    best_score = match[1]
            
            if best_match:
                return best_match, best_score / 100.0
        
        # Return best candidate
        if candidates:
            # Clean and return first good candidate
            cleaned = self._clean_dealer_name(candidates[0])
            return cleaned, 0.6
        
        return "Unknown", 0.0
    
    def extract_model_name(self, ocr_results: List, full_text: str) -> Tuple[str, float]:
        """
        Extract model/product name
        Works for: tractors, vehicles, products, items
        """
        keywords = self.config["model_name"]["keywords"]
        
        # Common brand patterns (expandable)
        brand_patterns = [
            # Tractors
            r"(mahindra|john\s*deere|swaraj|sonalika|new\s*holland|massey\s*ferguson|eicher|farmtrac|powertrac)\s+([a-zA-Z0-9\-\s]+)",
            # General products
            r"(model|product|item)\s*[:#]?\s*([a-zA-Z0-9\-\s]+)",
        ]
        
        # Try brand patterns first
        for pattern in brand_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                # Extract brand + model
                brand = match.group(1).strip()
                model = match.group(2).strip()[:50]  # Limit length
                full_model = f"{brand.title()} {model}"
                return full_model, 0.85
        
        # Try fuzzy matching against master list
        if self.model_master:
            match = process.extractOne(
                full_text,
                self.model_master,
                scorer=fuzz.partial_ratio,
                score_cutoff=75
            )
            if match:
                return match[0], match[1] / 100.0
        
        # Fallback: Look near keywords
        lines = full_text.split('\n')
        for line in lines:
            line_lower = line.lower()
            for keyword in keywords:
                if keyword in line_lower:
                    # Return the line, cleaned
                    cleaned = line.strip()[:100]
                    if len(cleaned) > 3:
                        return cleaned, 0.5
        
        return "Unknown", 0.0
    
    def extract_horse_power(self, ocr_results: List, full_text: str) -> Tuple[float, float]:
        """
        Extract horse power (for vehicles/tractors)
        Returns 0 if not applicable (e.g., retail invoices)
        """
        pattern = self.config["horse_power"].get("pattern",r"(\d+\.?\d*)\s*(?:hp|h\.?p\.?|horse\s*power|bhp)")
        
        # Search for HP pattern
        matches = re.finditer(pattern, full_text, re.IGNORECASE)
        
        hp_values = []
        for match in matches:
            try:
                hp = float(match.group(1))
                # Validate range (10-500 HP for vehicles)
                if 10 <= hp <= 500:
                    hp_values.append(hp)
            except ValueError:
                continue
        
        if hp_values:
            # Return median value
            hp_value = sorted(hp_values)[len(hp_values)//2]
            return hp_value, 0.9
        
        # Alternative: look for numbers near HP keywords
        keywords = self.config["horse_power"]["keywords"]
        lines = full_text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            for keyword in keywords:
                if keyword in line_lower:
                    # Extract numbers from this line
                    numbers = re.findall(r'\d+\.?\d*', line)
                    for num_str in numbers:
                        try:
                            hp = float(num_str)
                            if 10 <= hp <= 500:
                                return hp, 0.7
                        except ValueError:
                            continue
        
        # Not applicable (e.g., retail invoice)
        return 0.0, 0.0
    
    def extract_asset_cost(self, ocr_results: List, full_text: str) -> Tuple[float, float]:
        """
        Extract total cost/amount
        Works for ANY invoice type
        """
        pattern = self.config["asset_cost"].get("pattern",r"(?:rs\.?|₹|inr)?\s*(\d+(?:,\d{2,3})*(?:\.\d{2})?)")

        
        # Search for cost pattern
        matches = list(re.finditer(pattern, full_text, re.IGNORECASE))
        
        cost_values = []
        for match in matches:
            try:
                cost_str = match.group(1).replace(',', '').replace(' ', '')
                cost = float(cost_str)
                # Reasonable range check
                if cost > 10:  # Minimum value
                    cost_values.append(cost)
            except ValueError:
                continue
        
        if cost_values:
            # Return highest value (likely total/grand total)
            cost_value = max(cost_values)
            return cost_value, 0.9
        
        # Fallback: Look near cost keywords (without currency symbol)
        keywords = self.config["asset_cost"]["keywords"]
        lines = full_text.split('\n')
        
        for line in lines:
            line_lower = line.lower()
            # Prioritize "total" keywords
            if any(kw in line_lower for kw in ["total", "grand", "amount payable"]):
                numbers = re.findall(r'\d+(?:,\d{2,3})*(?:\.\d{2})?', line)
                for num_str in numbers:
                    try:
                        cost = float(num_str.replace(',', ''))
                        if cost > 10:
                            return cost, 0.8
                    except ValueError:
                        continue
        
        return 0.0, 0.0
    
    def _clean_dealer_name(self, name: str) -> str:
        """Clean extracted dealer name"""
        # Remove common prefixes/suffixes
        name = re.sub(r'\b(dealer|firm|vendor|seller|from|to)\b', '', name, flags=re.IGNORECASE)
        # Normalize whitespace
        name = ' '.join(name.split())
        return name.strip()
    
    def validate_fields(self, fields: Dict) -> Dict[str, bool]:
        """
        Validate extracted fields
        
        Returns:
            Dict with validation results for each field
        """
        validation = {
            "dealer_name": len(fields.get("dealer_name", "")) > 3 and fields["dealer_name"] != "Unknown",
            "model_name": len(fields.get("model_name", "")) > 3 and fields["model_name"] != "Unknown",
            "horse_power": 0 <= fields.get("horse_power", 0) <= 500,
            "asset_cost": fields.get("asset_cost", 0) > 0
        }
        
        return validation


def test_field_extractor():
    """Test field extraction"""
    # Sample invoice text
    sample_text = """
    ABC MOTORS PVT LTD
    Authorized Dealer - Mahindra Tractors
    
    Invoice No: INV-2024-001
    Date: 15-Jan-2024
    
    Product Details:
    Model: Mahindra 575 DI
    Horse Power: 50 HP
    Category: Agriculture Tractor
    
    Pricing:
    Ex-Showroom: Rs. 5,00,000
    Registration: Rs. 25,000
    Total Amount: Rs. 5,25,000
    """
    
    # Mock OCR results
    from collections import namedtuple
    OCRResult = namedtuple('OCRResult', ['text', 'bbox', 'confidence'])
    
    lines = [line.strip() for line in sample_text.split('\n') if line.strip()]
    ocr_results = [OCRResult(line, (0, i*20, 500, (i+1)*20), 0.9) for i, line in enumerate(lines)]
    
    # Extract fields
    extractor = FieldExtractor()
    fields, confidence = extractor.extract_all_fields(ocr_results, sample_text)
    
    print("="*60)
    print("FIELD EXTRACTION TEST")
    print("="*60)
    print(f"\nDealer Name: {fields['dealer_name']}")
    print(f"Model Name: {fields['model_name']}")
    print(f"Horse Power: {fields['horse_power']} HP")
    print(f"Asset Cost: Rs. {fields['asset_cost']:,.2f}")
    print(f"\nOverall Confidence: {confidence:.2%}")
    
    # Validate
    validation = extractor.validate_fields(fields)
    print("\nValidation Results:")
    for field, is_valid in validation.items():
        status = "✓" if is_valid else "✗"
        print(f"  {status} {field}: {'Valid' if is_valid else 'Invalid'}")


if __name__ == "__main__":
    test_field_extractor()