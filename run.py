"""
Document AI Invoice Extraction System - Project Structure

Run this script to create the complete project structure.
"""

import os

def create_project_structure():
    """Create the complete directory structure for the project"""
    
    directories = [
        'data/raw',
        'data/processed',
        'data/annotations',
        'models/weights',
        'utils',
        'sample_output',
        'notebooks',
        'configs',
        'logs',
        'tests'
    ]
    
    files = {
        'executable.py': '# Main execution file',
        'requirements.txt': '',
        'README.md': '# Invoice Field Extraction System',
        'utils/__init__.py': '',
        'utils/pdf_processor.py': '# PDF to image conversion',
        'utils/ocr_engine.py': '# OCR extraction',
        'utils/field_extractor.py': '# Field extraction logic',
        'utils/signature_detector.py': '# Signature/stamp detection',
        'utils/post_processor.py': '# Post-processing and validation',
        'utils/visualizer.py': '# Visualization utilities',
        'configs/config.yaml': '# Configuration parameters',
        'tests/test_pipeline.py': '# Unit tests',
        '.gitignore': 'data/\n*.pyc\n__pycache__/\nlogs/\n*.log'
    }
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created directory: {directory}")
    
    # Create files
    for filepath, content in files.items():
        if not os.path.exists(filepath):
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✓ Created file: {filepath}")
    
    print("\n✅ Project structure created successfully!")
    print("\nNext steps:")
    print("1. Place your PDF invoices in data/raw/")
    print("2. Install dependencies: pip install -r requirements.txt")
    print("3. Run the pipeline: python executable.py")

if __name__ == "__main__":
    create_project_structure()