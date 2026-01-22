# Visualization utilities
"""
Visualizer - EDA and result visualization utilities
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import cv2

sns.set_style("whitegrid")


class Visualizer:
    """Visualization utilities for EDA and results"""
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def visualize_extraction_result(self, image: np.ndarray, result: Dict, 
                                   output_path: str = None):
        """
        Visualize extraction results on image
        
        Args:
            image: Original image
            result: Extraction result dictionary
            output_path: Where to save visualization
        """
        vis_image = image.copy()
        
        # Draw signature bbox if present
        if result['fields']['signature']['present']:
            bbox = result['fields']['signature']['bbox']
            cv2.rectangle(vis_image, 
                         (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (0, 255, 0), 3)
            cv2.putText(vis_image, "Signature", 
                       (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw stamp bbox if present
        if result['fields']['stamp']['present']:
            bbox = result['fields']['stamp']['bbox']
            cv2.rectangle(vis_image,
                         (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                         (255, 0, 0), 3)
            cv2.putText(vis_image, "Stamp",
                       (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Add text overlay with extracted fields
        y_offset = 30
        texts = [
            f"Dealer: {result['fields']['dealer_name']}",
            f"Model: {result['fields']['model_name']}",
            f"HP: {result['fields']['horse_power']}",
            f"Cost: {result['fields']['asset_cost']}",
            f"Confidence: {result['confidence']:.2%}"
        ]
        
        for text in texts:
            cv2.putText(vis_image, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(vis_image, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            y_offset += 30
        
        if output_path:
            cv2.imwrite(output_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image
    
    def plot_batch_metrics(self, results: List[Dict]):
        """
        Plot metrics from batch processing results
        
        Args:
            results: List of result dictionaries
        """
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'doc_id': r['doc_id'],
                'confidence': r.get('confidence', 0),
                'processing_time': r.get('processing_time_sec', 0),
                'cost': r.get('cost_estimate_usd', 0),
                'has_error': 'error' in r
            }
            for r in results
        ])
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Confidence distribution
        axes[0, 0].hist(df['confidence'], bins=20, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(df['confidence'].mean(), color='red', 
                          linestyle='--', label=f'Mean: {df["confidence"].mean():.2%}')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Confidence Score Distribution')
        axes[0, 0].legend()
        
        # 2. Processing time distribution
        axes[0, 1].hist(df['processing_time'], bins=20, 
                       edgecolor='black', alpha=0.7, color='green')
        axes[0, 1].axvline(df['processing_time'].mean(), color='red',
                          linestyle='--', 
                          label=f'Mean: {df["processing_time"].mean():.2f}s')
        axes[0, 1].set_xlabel('Processing Time (seconds)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Processing Time Distribution')
        axes[0, 1].legend()
        
        # 3. Success rate
        success_rate = (~df['has_error']).sum() / len(df) * 100
        axes[1, 0].bar(['Success', 'Failed'], 
                      [(~df['has_error']).sum(), df['has_error'].sum()],
                      color=['green', 'red'], alpha=0.7)
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Success Rate: {success_rate:.1f}%')
        
        # 4. Cost analysis
        total_cost = df['cost'].sum()
        avg_cost = df['cost'].mean()
        axes[1, 1].bar(['Total Cost', 'Avg Cost'], 
                      [total_cost, avg_cost],
                      color='blue', alpha=0.7)
        axes[1, 1].set_ylabel('USD')
        axes[1, 1].set_title('Cost Analysis')
        axes[1, 1].text(0, total_cost, f'${total_cost:.4f}', 
                       ha='center', va='bottom')
        axes[1, 1].text(1, avg_cost, f'${avg_cost:.4f}',
                       ha='center', va='bottom')
        
        plt.tight_layout()
        output_path = self.output_dir / "batch_metrics.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved batch metrics to {output_path}")
    
    def plot_field_accuracy(self, results: List[Dict], ground_truth: List[Dict] = None):
        """
        Plot field-level accuracy if ground truth is available
        
        Args:
            results: Extraction results
            ground_truth: Ground truth annotations
        """
        if not ground_truth:
            print("Ground truth not available for accuracy calculation")
            return
        
        # Calculate field-level accuracy
        fields = ['dealer_name', 'model_name', 'horse_power', 
                 'asset_cost', 'signature', 'stamp']
        
        accuracies = {field: 0 for field in fields}
        
        for result, gt in zip(results, ground_truth):
            if 'error' in result:
                continue
            
            for field in fields:
                if field in ['signature', 'stamp']:
                    # Check presence
                    if result['fields'][field]['present'] == gt['fields'][field]['present']:
                        accuracies[field] += 1
                else:
                    # Check value match
                    if self._fields_match(result['fields'][field], 
                                         gt['fields'][field], field):
                        accuracies[field] += 1
        
        # Convert to percentages
        accuracies = {k: (v / len(results)) * 100 for k, v in accuracies.items()}
        
        # Plot
        plt.figure(figsize=(10, 6))
        bars = plt.bar(accuracies.keys(), accuracies.values(), 
                      color='skyblue', edgecolor='black', alpha=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%',
                    ha='center', va='bottom')
        
        plt.axhline(y=95, color='red', linestyle='--', label='Target: 95%')
        plt.xlabel('Field')
        plt.ylabel('Accuracy (%)')
        plt.title('Field-Level Accuracy')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        output_path = self.output_dir / "field_accuracy.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved field accuracy plot to {output_path}")
    
    def plot_error_analysis(self, results: List[Dict]):
        """
        Analyze and visualize error patterns
        
        Args:
            results: Extraction results
        """
        errors = []
        
        for result in results:
            if 'error' in result:
                errors.append({
                    'doc_id': result['doc_id'],
                    'error_type': self._categorize_error(result['error'])
                })
        
        if not errors:
            print("No errors to analyze!")
            return
        
        error_df = pd.DataFrame(errors)
        error_counts = error_df['error_type'].value_counts()
        
        # Plot error distribution
        plt.figure(figsize=(10, 6))
        error_counts.plot(kind='bar', color='coral', edgecolor='black', alpha=0.7)
        plt.xlabel('Error Type')
        plt.ylabel('Count')
        plt.title('Error Type Distribution')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = self.output_dir / "error_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved error analysis to {output_path}")
    
    def create_eda_report(self, results: List[Dict]):
        """
        Create comprehensive EDA report
        
        Args:
            results: Extraction results
        """
        print("Creating EDA report...")
        
        # 1. Batch metrics
        self.plot_batch_metrics(results)
        
        # 2. Error analysis
        self.plot_error_analysis(results)
        
        # 3. Confidence vs Processing Time
        self._plot_confidence_vs_time(results)
        
        # 4. Summary statistics
        self._generate_summary_stats(results)
        
        print(f"EDA report saved to {self.output_dir}")
    
    def _plot_confidence_vs_time(self, results: List[Dict]):
        """Plot relationship between confidence and processing time"""
        data = [(r.get('confidence', 0), r.get('processing_time_sec', 0))
                for r in results if 'error' not in r]
        
        if not data:
            return
        
        confidences, times = zip(*data)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(times, confidences, alpha=0.6, edgecolors='black')
        plt.xlabel('Processing Time (seconds)')
        plt.ylabel('Confidence Score')
        plt.title('Confidence vs Processing Time')
        plt.grid(True, alpha=0.3)
        
        # Add trend line
        z = np.polyfit(times, confidences, 1)
        p = np.poly1d(z)
        plt.plot(sorted(times), p(sorted(times)), 
                "r--", alpha=0.8, label='Trend')
        plt.legend()
        
        output_path = self.output_dir / "confidence_vs_time.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_stats(self, results: List[Dict]):
        """Generate summary statistics"""
        df = pd.DataFrame([
            {
                'confidence': r.get('confidence', 0),
                'processing_time': r.get('processing_time_sec', 0),
                'cost': r.get('cost_estimate_usd', 0),
                'success': 'error' not in r
            }
            for r in results
        ])
        
        summary = {
            "total_documents": len(results),
            "successful": df['success'].sum(),
            "failed": (~df['success']).sum(),
            "success_rate": f"{(df['success'].sum() / len(df)) * 100:.2f}%",
            "avg_confidence": f"{df['confidence'].mean():.2%}",
            "std_confidence": f"{df['confidence'].std():.2%}",
            "avg_time": f"{df['processing_time'].mean():.2f}s",
            "total_cost": f"${df['cost'].sum():.4f}",
            "avg_cost": f"${df['cost'].mean():.4f}"
        }
        
        # Save to JSON
        output_path = self.output_dir / "summary_stats.json"
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        for key, value in summary.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
    
    def _fields_match(self, extracted, ground_truth, field_type):
        """Check if extracted field matches ground truth"""
        if field_type in ['dealer_name', 'model_name']:
            from rapidfuzz import fuzz
            return fuzz.ratio(str(extracted).lower(), 
                            str(ground_truth).lower()) >= 90
        else:
            # Numeric fields with 5% tolerance
            return abs(float(extracted) - float(ground_truth)) / float(ground_truth) <= 0.05
    
    def _categorize_error(self, error_msg: str) -> str:
        """Categorize error message"""
        error_msg = error_msg.lower()
        
        if 'pdf' in error_msg:
            return 'PDF Processing Error'
        elif 'ocr' in error_msg:
            return 'OCR Error'
        elif 'memory' in error_msg:
            return 'Memory Error'
        elif 'timeout' in error_msg:
            return 'Timeout Error'
        else:
            return 'Other Error'


def test_visualizer():
    """Test visualization functions"""
    # Create sample results
    sample_results = [
        {
            "doc_id": f"invoice_{i:03d}",
            "confidence": np.random.uniform(0.7, 0.98),
            "processing_time_sec": np.random.uniform(2, 8),
            "cost_estimate_usd": 0.001,
            "fields": {
                "dealer_name": "ABC Tractors",
                "model_name": "Mahindra 575 DI",
                "horse_power": 50,
                "asset_cost": 525000,
                "signature": {"present": True, "bbox": [100, 200, 300, 250]},
                "stamp": {"present": True, "bbox": [400, 500, 500, 550]}
            }
        }
        for i in range(50)
    ]
    
    # Add some errors
    sample_results.append({
        "doc_id": "invoice_error",
        "error": "OCR extraction failed"
    })
    
    visualizer = Visualizer()
    visualizer.create_eda_report(sample_results)
    print("Test visualizations created successfully!")


if __name__ == "__main__":
    test_visualizer()