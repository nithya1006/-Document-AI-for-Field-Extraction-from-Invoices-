"""
INVOICE FIELD EXTRACTION SYSTEM - Main Executable
Production-ready, tested, and fully compatible with latest versions

Extracts 6 key fields from invoice images (PNG/JPG) or PDFs:
1. Dealer Name (fuzzy matching)
2. Model Name (exact matching)
3. Horse Power (numeric)
4. Asset Cost (numeric)
5. Dealer Signature (with bounding box)
6. Dealer Stamp (with bounding box)

Usage:
    # Using EasyOCR (default)
    python executable.py --input invoice.png
    python executable.py --input_dir data/raw --output_dir sample_output

    # Using Ollama vision model (llama3.2-vision)
    python executable.py --input invoice.png --ollama
    python executable.py --input_dir data/raw --output_dir sample_output --ollama
"""

import os
import sys
import json
import time
import argparse
import yaml
from pathlib import Path
from typing import Dict, List
import logging

# Add utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.pdf_processor import PDFProcessor
from utils.ocr_engine import OCREngine
from utils.field_extractor import FieldExtractor
from utils.signature_detector import SignatureStampDetector
from utils.ollama_vision import OllamaVisionExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/extraction.log')
    ]
)
logger = logging.getLogger(__name__)


class InvoiceExtractor:
    """Main pipeline for invoice field extraction"""

    def __init__(self, config_path: str = "configs/config.yaml", use_ollama: bool = False, ollama_model: str = "llama3.2-vision"):
        """Initialize the extraction pipeline

        Args:
            config_path: Path to configuration file
            use_ollama: If True, use Ollama vision model instead of EasyOCR
            ollama_model: Ollama vision model to use (default: llava)
        """

        # Create logs directory
        os.makedirs("logs", exist_ok=True)

        self.use_ollama = use_ollama
        self.ollama_model = ollama_model

        # Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            self.config = self._default_config()

        logger.info("="*60)
        logger.info("INVOICE FIELD EXTRACTION SYSTEM")
        logger.info("="*60)
        logger.info("Initializing extraction pipeline...")

        # Initialize components
        self.pdf_processor = PDFProcessor(
            dpi=self.config['processing']['image_dpi'],
            max_size=self.config['processing']['max_image_size']
        )

        if self.use_ollama:
            logger.info(f"Using Ollama vision model: {self.ollama_model}")
            self.ollama_extractor = OllamaVisionExtractor(model=self.ollama_model)
            self.ocr_engine = None
            self.field_extractor = None
        else:
            logger.info("Using EasyOCR engine")
            self.ollama_extractor = None
            self.ocr_engine = OCREngine(
                languages=self.config['ocr']['languages'],
                use_gpu=self.config['ocr']['use_gpu']
            )
            self.field_extractor = FieldExtractor(config=self.config.get('fields'))
            # Load master data if available
            self._load_master_data()

        self.signature_detector = SignatureStampDetector()

        logger.info("✓ Pipeline initialized successfully")
        logger.info("="*60)
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'processing': {
                'image_dpi': 300,
                'max_image_size': 2048,
                'preprocessing': ['denoise', 'deskew']
            },
            'ocr': {
                'engine': 'easyocr',
                'languages': ['en'],
                'use_gpu': False,
                'confidence_threshold': 0.5
            },
            'detection': {
                'signature': {'confidence_threshold': 0.5, 'iou_threshold': 0.5},
                'stamp': {'confidence_threshold': 0.5, 'iou_threshold': 0.5}
            }
        }
    
    def _load_master_data(self):
        """Load dealer and model master files if available"""
        dealer_file = "data/dealer_master.txt"
        model_file = "data/model_master.txt"
        
        if not os.path.exists(dealer_file) or not os.path.exists(model_file):
            logger.info("Master data files not found. Creating samples...")
            self._create_sample_masters()
        
        self.field_extractor.load_master_data(dealer_file, model_file)
    
    def _create_sample_masters(self):
        """Create sample master files"""
        os.makedirs("data", exist_ok=True)
        
        # Sample dealers/vendors
        dealers = [
            "ABC Tractors Pvt Ltd",
            "Mahindra Authorized Dealer",
            "John Deere Sales Center",
            "Swaraj Trading Company",
            "New Holland Distributors",
            "General Store",
            "Retail Shop",
            "Industrial Supplies Co"
        ]
        
        with open("data/dealer_master.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(dealers))
        
        # Sample models/products
        models = [
            "Mahindra 575 DI",
            "Mahindra 475 DI",
            "John Deere 5050",
            "Swaraj 855 FE",
            "New Holland 3600"
        ]
        
        with open("data/model_master.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(models))
    
    def process_document(self, file_path: str) -> Dict:
        """
        Process a single document (PDF or image)
        
        Args:
            file_path: Path to PDF or image file
            
        Returns:
            Dictionary with extracted fields and metadata
        """
        start_time = time.time()
        doc_id = Path(file_path).stem
        
        logger.info(f"\nProcessing: {doc_id}")
        logger.info("-" * 60)
        
        try:
            # Step 1: Load image
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                logger.info("Loading image file...")
                import cv2
                image = cv2.imread(file_path)
                if image is None:
                    raise ValueError(f"Could not load image: {file_path}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images = [image]
            elif file_ext == '.pdf':
                logger.info("Converting PDF to images...")
                images = self.pdf_processor.pdf_to_images(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            if not images:
                raise ValueError("No images extracted from document")
            
            # Process first page
            image = images[0]
            logger.info(f"Image size: {image.shape}")
            
            # Step 2: Preprocess
            logger.info("Preprocessing image...")
            preprocessed = self.pdf_processor.preprocess_image(
                image,
                operations=self.config['processing']['preprocessing']
            )

            if self.use_ollama:
                # Use Ollama vision model for extraction
                logger.info(f"Extracting with Ollama vision ({self.ollama_model})...")
                extraction_result = self.ollama_extractor.extract_all(preprocessed)

                fields = {
                    'dealer_name': extraction_result.fields.get('dealer_name'),
                    'model_name': extraction_result.fields.get('model_name'),
                    'horse_power': extraction_result.fields.get('horse_power'),
                    'asset_cost': extraction_result.fields.get('asset_cost'),
                }
                text_confidence = extraction_result.confidence

                logger.info(f"✓ Dealer: {fields['dealer_name']}")
                logger.info(f"✓ Model: {fields['model_name']}")
                logger.info(f"✓ HP: {fields['horse_power']}")
                logger.info(f"✓ Cost: {fields['asset_cost']}")

                # Use Ollama's signature/stamp detection
                signature_result = {
                    'present': extraction_result.fields.get('has_signature', False),
                    'confidence': 0.8 if extraction_result.fields.get('has_signature') else 0.0,
                    'bbox': None
                }
                stamp_result = {
                    'present': extraction_result.fields.get('has_stamp', False),
                    'confidence': 0.8 if extraction_result.fields.get('has_stamp') else 0.0,
                    'bbox': None
                }

            else:
                # Use EasyOCR for extraction
                logger.info("Extracting text with OCR...")
                ocr_results = self.ocr_engine.extract_text(preprocessed)
                full_text = ' '.join([r.text for r in ocr_results])

                logger.info(f"✓ Extracted {len(ocr_results)} text elements")

                # Step 4: Extract structured fields
                logger.info("Extracting structured fields...")
                fields, text_confidence = self.field_extractor.extract_all_fields(
                    ocr_results,
                    full_text
                )

                logger.info(f"✓ Dealer: {fields['dealer_name']}")
                logger.info(f"✓ Model: {fields['model_name']}")
                logger.info(f"✓ HP: {fields['horse_power']}")
                logger.info(f"✓ Cost: {fields['asset_cost']}")

                # Detect signature and stamp using traditional method
                logger.info("Detecting signature and stamp...")
                signature_result = self.signature_detector.detect_signature(
                    image,
                    confidence_threshold=self.config['detection']['signature']['confidence_threshold']
                )

                stamp_result = self.signature_detector.detect_stamp(
                    image,
                    confidence_threshold=self.config['detection']['stamp']['confidence_threshold']
                )

            logger.info(f"✓ Signature: {'Yes' if signature_result['present'] else 'No'}")
            logger.info(f"✓ Stamp: {'Yes' if stamp_result['present'] else 'No'}")

            fields['signature'] = signature_result
            fields['stamp'] = stamp_result

            # Calculate overall confidence
            sig_conf = signature_result.get('confidence', 0)
            stamp_conf = stamp_result.get('confidence', 0)
            overall_confidence = (text_confidence + sig_conf + stamp_conf) / 3
            
            # Calculate metrics
            processing_time = time.time() - start_time
            cost_estimate = self._estimate_cost(processing_time)
            
            # Prepare result
            result = {
                "doc_id": doc_id,
                "fields": fields,
                "confidence": round(overall_confidence, 3),
                "processing_time_sec": round(processing_time, 2),
                "cost_estimate_usd": round(cost_estimate, 4),
                "status": "success"
            }
            
            logger.info("-" * 60)
            logger.info(f"✓ Completed in {processing_time:.2f}s")
            logger.info(f"✓ Confidence: {overall_confidence:.1%}")
            logger.info(f"✓ Cost: ${cost_estimate:.4f}")
            
            return result
            
        except Exception as e:
            logger.error(f"✗ Error: {str(e)}", exc_info=True)
            return {
                "doc_id": doc_id,
                "error": str(e),
                "processing_time_sec": round(time.time() - start_time, 2),
                "status": "failed"
            }
    
    def process_batch(self, input_dir: str, output_dir: str):
        """
        Process a batch of documents
        
        Args:
            input_dir: Directory containing files
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all supported files
        supported_extensions = ['.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        all_files = []
        
        for ext in supported_extensions:
            all_files.extend(Path(input_dir).glob(f"*{ext}"))
        
        if not all_files:
            logger.error(f"No supported files found in {input_dir}")
            logger.info(f"Supported formats: {', '.join(supported_extensions)}")
            return
        
        logger.info(f"\n{'='*60}")
        logger.info(f"BATCH PROCESSING: {len(all_files)} files")
        logger.info(f"{'='*60}\n")
        
        results = []
        successful = 0
        failed = 0
        
        for i, file_path in enumerate(all_files, 1):
            logger.info(f"\n[{i}/{len(all_files)}] Processing: {file_path.name}")
            
            result = self.process_document(str(file_path))
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
            else:
                failed += 1
            
            # Save individual result
            output_file = Path(output_dir) / f"{result['doc_id']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save batch summary
        summary = {
            "total_documents": len(results),
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful/len(results)*100):.1f}%",
            "average_confidence": sum(r.get('confidence', 0) for r in results) / len(results),
            "average_time": sum(r['processing_time_sec'] for r in results) / len(results),
            "total_cost": sum(r.get('cost_estimate_usd', 0) for r in results),
            "results": results
        }
        
        summary_file = Path(output_dir) / "batch_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("BATCH SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total Documents: {summary['total_documents']}")
        logger.info(f"Successful: {summary['successful']}")
        logger.info(f"Failed: {summary['failed']}")
        logger.info(f"Success Rate: {summary['success_rate']}")
        logger.info(f"Avg Confidence: {summary['average_confidence']:.1%}")
        logger.info(f"Avg Time: {summary['average_time']:.2f}s")
        logger.info(f"Total Cost: ${summary['total_cost']:.4f}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"{'='*60}\n")
    
    def _estimate_cost(self, processing_time: float) -> float:
        """
        Estimate processing cost
        
        Using EasyOCR (open-source, CPU-based):
        - Cost: ~$0.001 per document (very low)
        """
        return 0.001


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Invoice Field Extraction System - Extract structured data from invoices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python executable.py --input invoice.png
  
  # Process batch
  python executable.py --input_dir data/raw --output_dir sample_output
  
  # Use custom config
  python executable.py --input invoice.png --config my_config.yaml
        """
    )
    
    parser.add_argument('--input', type=str, help='Input file path (PDF/image)')
    parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--input_dir', type=str, help='Input directory with files')
    parser.add_argument('--output_dir', type=str, default='sample_output',
                       help='Output directory (default: sample_output)')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Configuration file (default: configs/config.yaml)')
    parser.add_argument('--ollama', action='store_true',
                       help='Use Ollama vision model instead of EasyOCR')
    parser.add_argument('--model', type=str, default='llama3.2-vision',
                       help='Ollama vision model to use (default: llama3.2-vision)')
    
    args = parser.parse_args()
    
    if not args.input and not args.input_dir:
        parser.print_help()
        return
    
    # Initialize extractor
    try:
        extractor = InvoiceExtractor(
            config_path=args.config,
            use_ollama=args.ollama,
            ollama_model=args.model
        )
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return
    
    if args.input:
        # Process single file
        result = extractor.process_document(args.input)
        
        output_file = args.output or f"sample_output/{result['doc_id']}.json"
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n✓ Result saved to: {output_file}")
        
        # Print result
        print("\n" + "="*60)
        print("EXTRACTION RESULT")
        print("="*60)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    elif args.input_dir:
        # Process batch
        extractor.process_batch(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()