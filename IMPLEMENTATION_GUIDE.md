# Implementation Guide - Invoice Field Extraction System

This guide walks you through implementing the complete system from scratch.

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Detailed Setup](#detailed-setup)
3. [Understanding the Pipeline](#understanding-the-pipeline)
4. [Customization](#customization)
5. [Testing Strategy](#testing-strategy)
6. [Optimization Tips](#optimization-tips)
7. [Troubleshooting](#troubleshooting)
8. [Submission Checklist](#submission-checklist)

---

## 🚀 Quick Start

### 1. Setup Project Structure

```bash
# Create project directory
mkdir invoice-extraction
cd invoice-extraction

# Run the structure creation script
python create_structure.py

# Your directory should now look like this:
# invoice-extraction/
# ├── executable.py
# ├── requirements.txt
# ├── README.md
# ├── configs/
# ├── utils/
# ├── data/
# └── sample_output/
```

### 2. Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import paddleocr; print('PaddleOCR installed successfully!')"
```

### 3. Prepare Data

```bash
# Create data directories
mkdir -p data/raw data/processed

# Place your PDF invoices in data/raw/
cp /path/to/your/pdfs/*.pdf data/raw/

# Create master lists (optional but recommended)
# Create data/dealer_master.txt with known dealer names
# Create data/model_master.txt with known model names
```

### 4. Run First Extraction

```bash
# Process a single document
python executable.py --input data/raw/invoice_001.pdf --output sample_output/result.json

# View the result
cat sample_output/result.json
```

---

## 🔧 Detailed Setup

### Understanding Each Component

#### 1. **PDF Processor** (`utils/pdf_processor.py`)

**Purpose:** Converts PDF to images and preprocesses them.

**Key Functions:**
- `pdf_to_images()`: Converts PDF pages to numpy arrays
- `preprocess_image()`: Applies denoising, deskewing, etc.
- `extract_regions()`: Identifies text/signature regions

**Customization:**
```python
# Adjust DPI for better quality (higher = better quality, slower)
processor = PDFProcessor(dpi=300)  # Try 200 for speed, 400 for quality

# Add custom preprocessing
processor.preprocess_image(image, operations=['grayscale', 'denoise', 'deskew'])
```

#### 2. **OCR Engine** (`utils/ocr_engine.py`)

**Purpose:** Extracts text from images using multiple OCR engines.

**Supported Engines:**
- **PaddleOCR** (Recommended): Fast, accurate, multilingual
- **EasyOCR**: Better for handwriting
- **Tesseract**: Fallback option

**Language Support:**
```python
# English only
ocr = OCREngine(engine="paddleocr", languages=["en"])

# English + Hindi
ocr = OCREngine(engine="paddleocr", languages=["en", "hi"])

# For Gujarati, use English model as fallback
ocr = OCREngine(engine="easyocr", languages=["en", "hi"])
```

**Performance Tips:**
- Use GPU for 3-5x speedup: `use_gpu=True`
- Lower confidence threshold for handwritten text: `min_confidence=0.4`

#### 3. **Field Extractor** (`utils/field_extractor.py`)

**Purpose:** Extracts structured fields using NLP and pattern matching.

**Extraction Logic:**

**Dealer Name:**
```python
# Looks for keywords: "dealer", "firm", "seller", "vendor"
# Fuzzy matches against master list
# Falls back to proper nouns at document top
```

**Model Name:**
```python
# Searches for brand names: "Mahindra", "John Deere", etc.
# Extracts model code (e.g., "575 DI")
# Matches against model master list
```

**Horse Power:**
```python
# Pattern: "(\d+)\s*HP" or "(\d+)\s*H\.P\."
# Validates range: 10-200 HP (tractor range)
```

**Asset Cost:**
```python
# Pattern: "Rs\. (\d+(?:,\d{3})*)" or "₹(\d+)"
# Removes commas, extracts numeric value
# Returns highest value found (likely total)
```

**Customization:**
```python
# Edit configs/config.yaml
fields:
  horse_power:
    keywords: ["hp", "horse power", "h.p"]  # Add more keywords
    tolerance: 0.05  # ±5% tolerance for matching
  
  asset_cost:
    keywords: ["cost", "price", "total"]  # Add domain-specific terms
```

#### 4. **Signature & Stamp Detector** (`utils/signature_detector.py`)

**Purpose:** Detects signatures and stamps using computer vision.

**Detection Methods:**

**Traditional CV (Default):**
- Contour detection
- Aspect ratio filtering (signatures: 2:1 to 6:1, stamps: ~1:1)
- Density analysis (ink coverage)
- Position filtering (signatures in lower document)

**YOLO (Optional, Better Accuracy):**
```python
# Train a YOLOv8 model on annotated signatures/stamps
from ultralytics import YOLO

# Train
model = YOLO('yolov8n.pt')
model.train(data='signature_dataset.yaml', epochs=50)

# Use in detector
detector = SignatureStampDetector(
    use_yolo=True,
    model_path='models/weights/signature_yolo.pt'
)
```

**Improving Detection:**
1. **Collect Samples:** Manually annotate 50-100 signatures/stamps
2. **Train Custom Model:** Fine-tune YOLOv8 on your data
3. **Adjust Thresholds:** Lower for lenient, higher for strict detection

---

## 🔄 Understanding the Pipeline

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: invoice.pdf                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 1: PDF TO IMAGE CONVERSION                            │
│  - Load PDF using PyMuPDF                                   │
│  - Render at 300 DPI                                        │
│  - Convert to numpy array (RGB)                             │
│  Output: List[np.ndarray]                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 2: IMAGE PREPROCESSING                                │
│  - Denoise (remove artifacts)                               │
│  - Deskew (correct rotation)                                │
│  - Optional: Grayscale, threshold                           │
│  Output: Preprocessed image                                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 3: OCR TEXT EXTRACTION                                │
│  - PaddleOCR detects text regions                           │
│  - Recognizes characters (multi-language)                   │
│  - Returns text + bounding boxes + confidence               │
│  Output: List[OCRResult(text, bbox, conf)]                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┴──────────────────┐
        │                                      │
        ▼                                      ▼
┌───────────────────┐              ┌─────────────────────────┐
│ STEP 4A: TEXT     │              │ STEP 4B: VISUAL         │
│ FIELD EXTRACTION  │              │ DETECTION               │
│                   │              │                         │
│ - Dealer Name     │              │ - Signature Detection   │
│ - Model Name      │              │ - Stamp Detection       │
│ - Horse Power     │              │                         │
│ - Asset Cost      │              │ Uses:                   │
│                   │              │ - Contour analysis      │
│ Uses:             │              │ - Aspect ratios         │
│ - Keyword search  │              │ - Density metrics       │
│ - Regex patterns  │              │ - Color detection       │
│ - Fuzzy matching  │              │                         │
└─────────┬─────────┘              └────────────┬────────────┘
          │                                     │
          └──────────────┬──────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 5: POST-PROCESSING & VALIDATION                       │
│  - Validate field formats                                   │
│  - Check value ranges                                       │
│  - Calculate confidence scores                              │
│  - Cross-check consistency                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  STEP 6: OUTPUT GENERATION                                  │
│  {                                                          │
│    "doc_id": "invoice_001",                                 │
│    "fields": { ... },                                       │
│    "confidence": 0.92,                                      │
│    "processing_time_sec": 4.2,                              │
│    "cost_estimate_usd": 0.001                               │
│  }                                                          │
└─────────────────────────────────────────────────────────────┘
```

### Processing Time Breakdown

Typical processing time: **3-8 seconds per document**

| Step | Time (CPU) | Time (GPU) | Percentage |
|------|-----------|-----------|------------|
| PDF Conversion | 0.5s | 0.5s | 10% |
| Preprocessing | 0.3s | 0.3s | 6% |
| OCR | 4.0s | 1.5s | 67% |
| Field Extraction | 0.8s | 0.8s | 13% |
| Detection | 0.3s | 0.3s | 4% |
| **Total** | **5.9s** | **3.4s** | **100%** |

---

## 🎨 Customization

### 1. Adding New Fields

To extract additional fields (e.g., "Invoice Number", "Date"):

**Step 1:** Add to config (`configs/config.yaml`):
```yaml
fields:
  invoice_number:
    type: "text"
    matching: "exact"
    keywords: ["invoice no", "invoice #", "bill no"]
    pattern: "INV-\d{6}"
```

**Step 2:** Add extraction logic (`utils/field_extractor.py`):
```python
def extract_invoice_number(self, ocr_results: List, full_text: str) -> Tuple[str, float]:
    """Extract invoice number"""
    pattern = r"INV-\d{6}"
    matches = re.finditer(pattern, full_text, re.IGNORECASE)
    
    for match in matches:
        return match.group(0), 0.95
    
    # Fallback: look near keywords
    keywords = self.config["invoice_number"]["keywords"]
    # ... implement fallback logic
    
    return "Unknown", 0.0
```

**Step 3:** Update main extraction:
```python
def extract_all_fields(self, ocr_results, full_text):
    fields = {
        # ... existing fields
        "invoice_number": self.extract_invoice_number(ocr_results, full_text)
    }
    return fields
```

### 2. Supporting New Languages

**For Tamil/Telugu/Bengali:**

```python
# Option 1: Use EasyOCR (supports 80+ languages)
ocr = OCREngine(engine="easyocr", languages=["en", "ta", "te"])

# Option 2: Use Google Vision API (best accuracy, costs money)
from google.cloud import vision
client = vision.ImageAnnotatorClient()
```

**Add language-specific keywords:**
```yaml
fields:
  dealer_name:
    keywords:
      en: ["dealer", "seller"]
      hi: ["डीलर", "विक्रेता"]
      ta: ["வியாபாரி"]
```

### 3. Improving Accuracy

**Strategy 1: Ensemble OCR**
```python
class EnsembleOCR:
    def __init__(self):
        self.ocr1 = OCREngine(engine="paddleocr")
        self.ocr2 = OCREngine(engine="easyocr")
    
    def extract_with_voting(self, image):
        results1 = self.ocr1.extract_text(image)
        results2 = self.ocr2.extract_text(image)
        
        # Majority voting logic
        consensus = self._vote(results1, results2)
        return consensus
```

**Strategy 2: Fine-tune on Domain Data**
```python
# Collect 100-200 annotated invoices
# Fine-tune PaddleOCR
from paddleocr import PaddleOCR

# Train on your dataset
!python PaddleOCR/tools/train.py \
    -c configs/rec/rec_r34_vd_none_bilstm_ctc.yml \
    -o Global.pretrained_model=./pretrain_models/rec_r34_vd_none_bilstm_ctc
```

**Strategy 3: Post-processing Rules**
```python
def validate_and_correct(self, fields):
    # Example: HP should be divisible by 5 for most tractors
    hp = fields['horse_power']
    if hp % 5 != 0:
        # Round to nearest 5
        fields['horse_power'] = round(hp / 5) * 5
    
    # Example: Common model corrections
    corrections = {
        "Mahindra 57 5 DI": "Mahindra 575 DI",  # OCR spacing error
        "Mahindra 575 D1": "Mahindra 575 DI",   # 1 vs I confusion
    }
    
    model = fields['model_name']
    if model in corrections:
        fields['model_name'] = corrections[model]
    
    return fields
```

---

## 🧪 Testing Strategy

### 1. Create Test Dataset

```bash
# Manually annotate 20-30 invoices
data/
  annotations/
    invoice_001.json  # Ground truth
    invoice_002.json
    ...
```

**Annotation Format:**
```json
{
  "doc_id": "invoice_001",
  "ground_truth": {
    "dealer_name": "ABC Tractors Pvt Ltd",
    "model_name": "Mahindra 575 DI",
    "horse_power": 50,
    "asset_cost": 525000,
    "signature": {"present": true, "bbox": [100, 200, 300, 250]},
    "stamp": {"present": true, "bbox": [400, 500, 500, 550]}
  }
}
```

### 2. Evaluation Script

```python
# tests/test_accuracy.py
import json
from pathlib import Path
from executable import InvoiceExtractor
from rapidfuzz import fuzz

def calculate_accuracy():
    extractor = InvoiceExtractor()
    
    correct = 0
    total = 0
    
    # Load ground truth
    for gt_file in Path("data/annotations").glob("*.json"):
        with open(gt_file) as f:
            gt = json.load(f)
        
        # Extract
        pdf_path = f"data/raw/{gt['doc_id']}.pdf"
        result = extractor.process_document(pdf_path)
        
        # Check all fields
        if check_all_fields_match(result['fields'], gt['ground_truth']):
            correct += 1
        total += 1
    
    dla = (correct / total) * 100
    print(f"Document-Level Accuracy: {dla:.2f}%")
    return dla

def check_all_fields_match(extracted, ground_truth):
    # Dealer (fuzzy)
    dealer_match = fuzz.ratio(
        extracted['dealer_name'].lower(),
        ground_truth['dealer_name'].lower()
    ) >= 90
    
    # Model (exact)
    model_match = extracted['model_name'] == ground_truth['model_name']
    
    # HP (±5%)
    hp_match = abs(extracted['horse_power'] - ground_truth['horse_power']) / ground_truth['horse_power'] <= 0.05
    
    # Cost (±5%)
    cost_match = abs(extracted['asset_cost'] - ground_truth['asset_cost']) / ground_truth['asset_cost'] <= 0.05
    
    # Signature (IoU ≥ 0.5)
    sig_match = (
        extracted['signature']['present'] == ground_truth['signature']['present'] and
        calculate_iou(extracted['signature']['bbox'], ground_truth['signature']['bbox']) >= 0.5
    )
    
    # Stamp (IoU ≥ 0.5)
    stamp_match = (
        extracted['stamp']['present'] == ground_truth['stamp']['present'] and
        calculate_iou(extracted['stamp']['bbox'], ground_truth['stamp']['bbox']) >= 0.5
    )
    
    return all([dealer_match, model_match, hp_match, cost_match, sig_match, stamp_match])
```

---

## ⚡ Optimization Tips

### 1. Speed Optimization

```python
# Use lighter OCR model
ocr = OCREngine(engine="paddleocr", use_gpu=True)  # 3-5x faster

# Reduce image resolution
processor = PDFProcessor(dpi=200)  # vs 300

# Skip unnecessary preprocessing
processor.preprocess_image(image, operations=['denoise'])  # Skip deskew

# Process in parallel
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    results = executor.map(extractor.process_document, pdf_paths)
```

### 2. Accuracy Optimization

```python
# Use ensemble OCR
results_paddle = ocr_paddle.extract_text(image)
results_easy = ocr_easy.extract_text(image)
consensus = merge_results(results_paddle, results_easy)

# Add domain-specific post-processing
def validate_tractor_hp(hp):
    # Tractors come in standard HP: 25, 30, 35, 40, 45, 50, etc.
    standard_hps = [25, 30, 35, 40, 45, 50, 55, 60, 75, 90]
    return min(standard_hps, key=lambda x: abs(x - hp))

# Use master lists
fields['dealer_name'] = fuzzy_match_master(extracted_dealer, dealer_master)
```

### 3. Cost Optimization

```bash
# Use CPU instead of GPU (0.5x speed, but cheaper)
use_gpu: false

# Use lighter models
- PaddleOCR-mobile (smaller, faster)
- Traditional CV for signatures (no ML needed)

# Batch processing (amortize startup costs)
python executable.py --input_dir data/raw --output_dir sample_output
```

---

## 🐛 Troubleshooting

### Common Issues

**1. "No text detected" Error**
```python
# Solution: Adjust preprocessing
image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)[1]  # Harsher threshold
image = cv2.resize(image, None, fx=2, fy=2)  # Upscale

# Or try different OCR engine
ocr = OCREngine(engine="easyocr")  # Better for low-quality images
```

**2. "Wrong dealer/model extracted"**
```python
# Solution: Improve master list matching
def fuzzy_match_with_threshold(text, master_list, threshold=0.85):
    from rapidfuzz import process
    match = process.extractOne(text, master_list)
    if match[1] / 100 >= threshold:
        return match[0]
    return text  # Return original if no good match
```

**3. "Signature not detected"**
```python
# Solution: Lower confidence threshold
signature_result = detector.detect_signature(image, confidence_threshold=0.3)

# Or expand search area
# Look in entire lower half instead of just bottom 40%
```

**4. "Out of memory error"**
```python
# Solution: Reduce image size
processor = PDFProcessor(max_size=1024)  # vs 2048

# Or process page by page
for page in pdf_pages:
    result = process_page(page)
    results.append(result)
    del page  # Free memory
```

---

## ✅ Submission Checklist

### Required Files

- [x] `executable.py` - Main pipeline script
- [x] `requirements.txt` - All dependencies listed
- [x] `README.md` - Architecture, setup instructions
- [x] `configs/config.yaml` - Configuration parameters
- [x] `utils/` - All utility modules
- [x] `sample_output/result.json` - Example output

### Documentation

- [x] Architecture diagram in README
- [x] Cost analysis (per document)
- [x] Latency benchmarks
- [x] Installation instructions
- [x] Usage examples

### Code Quality

- [x] Modular structure (separate concerns)
- [x] Clear function documentation
- [x] Error handling
- [x] Logging for debugging
- [x] Configuration-driven (not hardcoded)

### Performance

- [x] DLA ≥ 95% (on test set)
- [x] Latency ≤ 30s per document
- [x] Cost ≤ $0.01 per document

### Bonus Points

- [x] EDA visualizations
- [x] Error analysis
- [x] Architecture diagram
- [x] Low-cost design (open-source components)
- [x] Optional web app (Streamlit)

### Testing

```bash
# Test installation
pip install -r requirements.txt

# Test single document
python executable.py --input test_invoice.pdf

# Test batch processing
python executable.py --input_dir test_data --output_dir test_output

# Run evaluation
python tests/test_accuracy.py

# Generate visualizations
python utils/visualizer.py
```

---

## 🎓 Learning Resources

### OCR
- [PaddleOCR Documentation](https://github.com/PaddlePaddle/PaddleOCR)
- [EasyOCR Documentation](https://github.com/JaidedAI/EasyOCR)

### Computer Vision
- [OpenCV Tutorials](https://docs.opencv.org/master/d9/df8/tutorial_root.html)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)

### NLP
- [RapidFuzz Documentation](https://github.com/maxbachmann/RapidFuzz)

### Papers
- Document AI: "LayoutLM" (Microsoft Research)
- Weak Supervision: "Snorkel" (Stanford)
- Active Learning: "A Survey" (Settles, 2009)

---

Good luck with your implementation! 🚀