import os
import sys
import cv2
import glob
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Tuple

base_dir = '../swimming_in_salt_data/images/'
sample_name = '20250404_24.4_14_01'  # Example sample name

# Configuration - Edit these variables as needed
CONFIG = {
    # Sample configuration
    'sample_name': sample_name,  # This will create a subdirectory in detection_results
    
    # Input directories (should match the sample name)
    'left_images': os.path.join(base_dir, sample_name, "right/"),
    'right_images': os.path.join(base_dir, sample_name, "left/"),

    # Calibration directory (where ROI info is stored)
    'calibration_dir': 'calibration_results',
    
    # Output directory (base - sample subdirectory will be created)
    'output_dir_base': '../swimming_in_salt_data/results/detection_results',
    
    # Image parameters
    'image_pattern': '*.jpg',
    
    # Detection parameters
    'threshold': 10,
    'min_area': 4,
    'background_frames': 100,  # Number of frames to use for background calculation
    
    # Camera names (should match calibration)
    'left_camera_name': 'lower',
    'right_camera_name': 'upper',
    
    # Processing options
    'save_processed_images': False,
    'save_detection_images': True,
    'create_visualizations': True,
    'verbose': True,
}

# Dynamically set detection output directory based on sample name
CONFIG['output_dir'] = os.path.join(CONFIG['output_dir_base'], CONFIG['sample_name'])

def load_roi_coordinates(calibration_dir: str):
    """Load ROI coordinates from calibration results"""
    roi_file = os.path.join(calibration_dir, 'saved_roi.npz')
    
    if not os.path.exists(roi_file):
        print(f"ROI file not found: {roi_file}")
        return None, None
    
    try:
        data = np.load(roi_file, allow_pickle=True)
        left_roi = tuple(data['left_roi'])
        right_roi = tuple(data['right_roi'])
        
        print(f"✓ Loaded ROI coordinates:")
        print(f"  Left ROI: {left_roi}")
        print(f"  Right ROI: {right_roi}")
        
        return left_roi, right_roi
        
    except Exception as e:
        print(f"Error loading ROI coordinates: {e}")
        return None, None

def crop_image(image: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop image using ROI coordinates"""
    x1, y1, x2, y2 = roi
    return image[y1:y2, x1:x2]

def extract_timestamp(filename: str) -> str:
    """Extract timestamp from filename"""
    match = re.search(r'frame_(\d{8})_(\d{6})_(\d+)', os.path.basename(filename))
    if match:
        date, time, microseconds = match.groups()
        return f"{date}_{time}_{microseconds}"
    
    # Fallback: use just the filename without extension
    return os.path.splitext(os.path.basename(filename))[0]

def compute_background(image_stack: List[np.ndarray], method='mean') -> np.ndarray:
    """Compute background from image stack"""
    if not image_stack:
        raise ValueError("Empty image stack")
    
    image_array = np.array(image_stack, dtype=np.float32)
    
    if method == 'mean':
        background = np.mean(image_array, axis=0)
    elif method == 'median':
        background = np.median(image_array, axis=0)
    else:
        raise ValueError(f"Unknown background method: {method}")
    
    return background.astype(np.uint8)

def detect_objects(binary_image: np.ndarray, min_area: int = 5) -> List[Dict]:
    """Detect objects in binary image using connected components"""
    num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    
    detections = []
    for i in range(1, num_labels):  # Skip background (label 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]
            
            detections.append({
                'area': area,
                'centroid_x': cx,
                'centroid_y': cy,
                'bbox_x': x,
                'bbox_y': y,
                'bbox_width': width,
                'bbox_height': height
            })
    
    return detections

def create_detection_visualization(original: np.ndarray, binary: np.ndarray, 
                                 detections: List[Dict], title: str) -> np.ndarray:
    """Create visualization showing detections on original image"""
    vis_img = cv2.cvtColor(original, cv2.COLOR_GRAY2RGB) if len(original.shape) == 2 else original.copy()
    
    # Draw bounding boxes and centroids
    for det in detections:
        x, y, w, h = det['bbox_x'], det['bbox_y'], det['bbox_width'], det['bbox_height']
        cx, cy = int(det['centroid_x']), int(det['centroid_y'])
        
        # Bounding box
        cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Centroid
        cv2.circle(vis_img, (cx, cy), 3, (255, 0, 0), -1)
        
        # Area text
        cv2.putText(vis_img, f'{det["area"]}px', (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    
    return vis_img

def save_detection_report(output_dir: str, detection_stats: Dict, total_frames: int):
    """Save comprehensive detection report"""
    report_file = os.path.join(output_dir, 'detection_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("OBJECT DETECTION REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Sample information
        f.write("SAMPLE INFORMATION:\n")
        f.write("-"*30 + "\n")
        f.write(f"Sample name:              {CONFIG['sample_name']}\n")
        f.write(f"Output directory:         {CONFIG['output_dir']}\n\n")
        
        # Parameters
        f.write("DETECTION PARAMETERS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Left images directory:    {CONFIG['left_images']}\n")
        f.write(f"Right images directory:   {CONFIG['right_images']}\n")
        f.write(f"Calibration directory:    {CONFIG['calibration_dir']}\n")
        f.write(f"Threshold:                {CONFIG['threshold']}\n")
        f.write(f"Minimum area:             {CONFIG['min_area']} pixels\n")
        f.write(f"Background frames:        {CONFIG['background_frames']}\n")
        f.write(f"Total frames processed:   {total_frames}\n\n")
        
        # Statistics
        left_total = sum(detection_stats['left_counts'])
        right_total = sum(detection_stats['right_counts'])
        total_detections = left_total + right_total
        
        f.write("DETECTION STATISTICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"★ Total objects detected: {total_detections}\n")
        f.write(f"★ Left camera detections: {left_total}\n")
        f.write(f"★ Right camera detections: {right_total}\n")
        
        if total_frames > 0:
            f.write(f"★ Average objects/frame:  {total_detections/total_frames:.2f}\n")
            f.write(f"★ Average left/frame:     {left_total/total_frames:.2f}\n")
            f.write(f"★ Average right/frame:    {right_total/total_frames:.2f}\n")
        
        if detection_stats['left_areas'] and detection_stats['right_areas']:
            all_areas = detection_stats['left_areas'] + detection_stats['right_areas']
            f.write(f"★ Average object area:    {np.mean(all_areas):.1f} pixels\n")
            f.write(f"★ Largest object:         {max(all_areas):.0f} pixels\n")
            f.write(f"★ Smallest object:        {min(all_areas):.0f} pixels\n")
        
        f.write("\n")
        
        # Frame-by-frame summary (first 20 frames)
        f.write("FRAME-BY-FRAME SUMMARY (first 20 frames):\n")
        f.write("-"*50 + "\n")
        f.write("Frame | Left Count | Right Count | Total\n")
        f.write("-"*50 + "\n")
        
        for i in range(min(20, len(detection_stats['left_counts']))):
            left_count = detection_stats['left_counts'][i]
            right_count = detection_stats['right_counts'][i]
            total_count = left_count + right_count
            f.write(f"{i:5d} | {left_count:10d} | {right_count:11d} | {total_count:5d}\n")
        
        if len(detection_stats['left_counts']) > 20:
            f.write("... (showing first 20 frames only)\n")
        
        f.write("\nOUTPUT FILES:\n")
        f.write("-"*20 + "\n")
        f.write("Detection CSVs (per frame):\n")
        f.write(f"  - {CONFIG['left_camera_name']}_*_objects.csv\n")
        f.write(f"  - {CONFIG['right_camera_name']}_*_objects.csv\n")
        f.write("Master CSV:\n")
        f.write("  - all_detections.csv\n")
        f.write("Background images:\n")
        f.write("  - left_background.png\n")
        f.write("  - right_background.png\n")
        if CONFIG['save_detection_images']:
            f.write("Detection visualizations:\n")
            f.write("  - detections/ folder\n")
        if CONFIG['save_processed_images']:
            f.write("Processed images:\n")
            f.write("  - processed/ folder\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Detection report saved to: {report_file}")

def create_detection_summary_plots(output_dir: str, detection_stats: Dict):
    """Create summary plots of detection statistics"""
    if not CONFIG['create_visualizations']:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Detection Summary - {CONFIG["sample_name"]}', fontsize=16, fontweight='bold')
    
    # Detection counts per frame
    frames = range(len(detection_stats['left_counts']))
    axes[0, 0].plot(frames, detection_stats['left_counts'], 'b-', label='Left camera', alpha=0.7)
    axes[0, 0].plot(frames, detection_stats['right_counts'], 'r-', label='Right camera', alpha=0.7)
    axes[0, 0].set_title('Objects Detected per Frame')
    axes[0, 0].set_xlabel('Frame Number')
    axes[0, 0].set_ylabel('Object Count')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Total counts
    total_counts = [l + r for l, r in zip(detection_stats['left_counts'], detection_stats['right_counts'])]
    axes[0, 1].plot(frames, total_counts, 'g-', alpha=0.7)
    axes[0, 1].set_title('Total Objects per Frame')
    axes[0, 1].set_xlabel('Frame Number')
    axes[0, 1].set_ylabel('Total Object Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Area distribution
    if detection_stats['left_areas'] and detection_stats['right_areas']:
        all_areas = detection_stats['left_areas'] + detection_stats['right_areas']
        axes[1, 0].hist(all_areas, bins=50, alpha=0.7, color='purple')
        axes[1, 0].set_title('Object Area Distribution')
        axes[1, 0].set_xlabel('Area (pixels)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Detection rate comparison
    left_total = sum(detection_stats['left_counts'])
    right_total = sum(detection_stats['right_counts'])
    total = left_total + right_total
    
    if total > 0:
        labels = ['Left Camera', 'Right Camera']
        sizes = [left_total, right_total]
        colors = ['lightblue', 'lightcoral']
        
        axes[1, 1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        axes[1, 1].set_title('Detection Distribution by Camera')
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'detection_summary.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Detection summary plots saved to: {plot_file}")

def process_detection_pipeline():
    """Main detection processing pipeline"""
    print("OBJECT DETECTION PIPELINE")
    print("="*60)
    print(f"Sample name:      {CONFIG['sample_name']}")
    print(f"Left images:      {CONFIG['left_images']}")
    print(f"Right images:     {CONFIG['right_images']}")
    print(f"Calibration dir:  {CONFIG['calibration_dir']}")
    print(f"Output directory: {CONFIG['output_dir']}")
    print(f"Threshold:        {CONFIG['threshold']}")
    print(f"Min area:         {CONFIG['min_area']} pixels")
    print(f"Background frames: {CONFIG['background_frames']}")
    
    # Create output directories with sample subdirectory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"✓ Created sample directory: {CONFIG['output_dir']}")
    
    if CONFIG['save_processed_images']:
        os.makedirs(os.path.join(CONFIG['output_dir'], 'processed'), exist_ok=True)
    
    if CONFIG['save_detection_images']:
        os.makedirs(os.path.join(CONFIG['output_dir'], 'detections'), exist_ok=True)
    
    # Load ROI coordinates
    left_roi, right_roi = load_roi_coordinates(CONFIG['calibration_dir'])
    if left_roi is None or right_roi is None:
        print("✗ Failed to load ROI coordinates. Cannot proceed.")
        return False
    
    # Get image lists
    left_images = sorted(glob.glob(os.path.join(CONFIG['left_images'], CONFIG['image_pattern'])))
    right_images = sorted(glob.glob(os.path.join(CONFIG['right_images'], CONFIG['image_pattern'])))
    
    if not left_images or not right_images:
        print("✗ No images found in input directories")
        return False
    
    if len(left_images) != len(right_images):
        print(f"⚠ Warning: Different number of images (left: {len(left_images)}, right: {len(right_images)})")
    
    num_frames = min(len(left_images), len(right_images))
    print(f"Processing {num_frames} image pairs...")
    
    # Initialize statistics
    detection_stats = {
        'left_counts': [],
        'right_counts': [],
        'left_areas': [],
        'right_areas': [],
        'timestamps': []
    }
    
    # Step 1: Load and crop images for background calculation
    print(f"\n{'='*60}")
    print("STEP 1: COMPUTING BACKGROUND IMAGES")
    print(f"{'='*60}")
    
    left_cropped_stack = []
    right_cropped_stack = []
    
    background_end = min(CONFIG['background_frames'], num_frames)
    
    for i in tqdm(range(background_end), desc="Loading images for background"):
        # Load images
        left_img = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)
        
        if left_img is None or right_img is None:
            continue
        
        # Crop using ROI
        left_cropped = crop_image(left_img, left_roi)
        right_cropped = crop_image(right_img, right_roi)
        
        left_cropped_stack.append(left_cropped)
        right_cropped_stack.append(right_cropped)
    
    # Compute backgrounds
    print(f"Computing background from {len(left_cropped_stack)} frames...")
    left_background = compute_background(left_cropped_stack)
    right_background = compute_background(right_cropped_stack)
    
    print(f"✓ Background computed")
    print(f"  Left background shape: {left_background.shape}")
    print(f"  Right background shape: {right_background.shape}")
    
    # Step 2: Process all frames for detection
    print(f"\n{'='*60}")
    print("STEP 2: OBJECT DETECTION")
    print(f"{'='*60}")
    
    all_detections = []
    
    for i in tqdm(range(num_frames), desc="Processing frames"):
        # Load images
        left_img = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)
        
        if left_img is None or right_img is None:
            continue
        
        # Extract timestamp
        timestamp = extract_timestamp(left_images[i])
        detection_stats['timestamps'].append(timestamp)
        
        # Crop images
        left_cropped = crop_image(left_img, left_roi)
        right_cropped = crop_image(right_img, right_roi)
        
        # Background subtraction
        left_diff = cv2.absdiff(left_cropped, left_background)
        right_diff = cv2.absdiff(right_cropped, right_background)
        
        # Thresholding
        _, left_binary = cv2.threshold(left_diff, CONFIG['threshold'], 255, cv2.THRESH_BINARY)
        _, right_binary = cv2.threshold(right_diff, CONFIG['threshold'], 255, cv2.THRESH_BINARY)
        
        # Object detection
        left_detections = detect_objects(left_binary, CONFIG['min_area'])
        right_detections = detect_objects(right_binary, CONFIG['min_area'])
        
        # Update statistics
        detection_stats['left_counts'].append(len(left_detections))
        detection_stats['right_counts'].append(len(right_detections))
        
        for det in left_detections:
            detection_stats['left_areas'].append(det['area'])
        for det in right_detections:
            detection_stats['right_areas'].append(det['area'])
        
        # Save detections to CSV
        for camera, detections, prefix in [
            ('left', left_detections, CONFIG['left_camera_name']),
            ('right', right_detections, CONFIG['right_camera_name'])
        ]:
            # Add frame info to detections
            frame_detections = []
            for j, det in enumerate(detections):
                det_copy = det.copy()
                det_copy['object_id'] = f"{prefix}_{timestamp}_obj{j:03d}"
                det_copy['frame'] = i
                det_copy['timestamp'] = timestamp
                det_copy['camera'] = camera
                frame_detections.append(det_copy)
                all_detections.append(det_copy)
            
            # Save frame CSV
            if frame_detections:
                df = pd.DataFrame(frame_detections)
                csv_file = os.path.join(CONFIG['output_dir'], f"{prefix}_{timestamp}_objects.csv")
                df.to_csv(csv_file, index=False)
        
        # Save processed images if requested
        if CONFIG['save_processed_images']:
            cv2.imwrite(os.path.join(CONFIG['output_dir'], 'processed', f"left_diff_{timestamp}.png"), left_diff)
            cv2.imwrite(os.path.join(CONFIG['output_dir'], 'processed', f"right_diff_{timestamp}.png"), right_diff)
            cv2.imwrite(os.path.join(CONFIG['output_dir'], 'processed', f"left_binary_{timestamp}.png"), left_binary)
            cv2.imwrite(os.path.join(CONFIG['output_dir'], 'processed', f"right_binary_{timestamp}.png"), right_binary)
        
        # Save detection visualizations if requested
        if CONFIG['save_detection_images']:
            left_vis = create_detection_visualization(left_cropped, left_binary, left_detections, f"Left {timestamp}")
            right_vis = create_detection_visualization(right_cropped, right_binary, right_detections, f"Right {timestamp}")
            
            cv2.imwrite(os.path.join(CONFIG['output_dir'], 'detections', f"left_detections_{timestamp}.png"), left_vis)
            cv2.imwrite(os.path.join(CONFIG['output_dir'], 'detections', f"right_detections_{timestamp}.png"), right_vis)
    
    # Step 3: Save comprehensive results
    print(f"\n{'='*60}")
    print("STEP 3: SAVING RESULTS")
    print(f"{'='*60}")
    
    # Save all detections to master CSV
    if all_detections:
        master_df = pd.DataFrame(all_detections)
        master_csv = os.path.join(CONFIG['output_dir'], 'all_detections.csv')
        master_df.to_csv(master_csv, index=False)
        print(f"✓ Master detections CSV saved: {master_csv}")
    
    # Save background images
    cv2.imwrite(os.path.join(CONFIG['output_dir'], 'left_background.png'), left_background)
    cv2.imwrite(os.path.join(CONFIG['output_dir'], 'right_background.png'), right_background)
    print(f"✓ Background images saved")
    
    # Save detection report
    save_detection_report(CONFIG['output_dir'], detection_stats, num_frames)
    
    # Create visualization plots
    if CONFIG['create_visualizations']:
        create_detection_summary_plots(CONFIG['output_dir'], detection_stats)
    
    # Print summary
    total_detections = sum(detection_stats['left_counts']) + sum(detection_stats['right_counts'])
    print(f"\n✓ Detection pipeline completed successfully!")
    print(f"✓ Sample: {CONFIG['sample_name']}")
    print(f"✓ Processed {num_frames} frames")
    print(f"✓ Total objects detected: {total_detections}")
    print(f"✓ Average objects per frame: {total_detections/num_frames:.2f}")
    print(f"✓ Results saved in: {CONFIG['output_dir']}")
    
    return True

def main():
    """Main function"""
    try:
        success = process_detection_pipeline()
        if success:
            print("\n🎉 Object detection completed successfully!")
        else:
            print("\n❌ Object detection failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()