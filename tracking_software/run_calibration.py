import os
import sys
import cv2
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add current directory to path so we can import our modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import functions from our calibration scripts
from single_camera_calib import single_camera_calibrate, save_calibration_report
from stereo_camera_calib import stereo_calibrate, save_stereo_calibration_report, load_single_camera_calibration

# Configuration - Edit these variables as needed
CONFIG = {
    # Image directories
    'left_images': '../swimming_in_salt_data/planktracker_2ndcalibration_August2025/small_water/lower/',
    'right_images': '../swimming_in_salt_data/planktracker_2ndcalibration_August2025/small_water/upper/',
    
    # Output directory
    'output_dir': './calibration_results',
    
    # Calibration parameters
    'chess_rows': 8,
    'chess_cols': 11,
    'square_size': 1.5,  # mm
    'image_pattern': '*.jpg',
    
    # Camera names
    'left_camera_name': 'lower',
    'right_camera_name': 'upper',
    
    # ROI selection
    'use_roi': True,  # Set to False to skip ROI selection
    'roi_margin': 10,  # Additional margin around selected ROI (pixels)
    'reuse_saved_roi': True,  # Set to True to automatically reuse saved ROI
    'roi_save_file': 'saved_roi.npz',  # File to save/load ROI coordinates
    
    # Visualization
    'create_visualizations': True,  # Set to False to skip visualizations
    'max_vis_images': 5,  # Maximum number of images to show in pipeline visualization
}

class ROISelector:
    """Interactive ROI selection tool"""
    
    def __init__(self):
        self.roi = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_image = None
        
    def mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for ROI selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                # Draw rectangle on temporary image
                self.temp_image = self.original_image.copy()
                cv2.rectangle(self.temp_image, self.start_point, self.end_point, (0, 255, 0), 2)
                cv2.imshow('Select ROI', self.temp_image)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)
            
            # Calculate ROI coordinates
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            
            # Add margin
            margin = CONFIG['roi_margin']
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(self.original_image.shape[1], x2 + margin)
            y2 = min(self.original_image.shape[0], y2 + margin)
            
            self.roi = (x1, y1, x2, y2)
            
            # Draw final rectangle
            self.temp_image = self.original_image.copy()
            cv2.rectangle(self.temp_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(self.temp_image, f'ROI: {x2-x1}x{y2-y1}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Select ROI', self.temp_image)
    
    def select_roi(self, image_path: str):
        """Select ROI from an example image"""
        print(f"\nROI Selection Tool")
        print("=" * 40)
        print(f"Loading example image: {os.path.basename(image_path)}")
        
        # Load image
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            print(f"Error: Cannot load image {image_path}")
            return None
            
        # Resize image if too large for display
        height, width = self.original_image.shape[:2]
        max_display_size = 1200
        
        if width > max_display_size or height > max_display_size:
            scale = max_display_size / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            display_image = cv2.resize(self.original_image, (new_width, new_height))
            self.scale_factor = scale
        else:
            display_image = self.original_image.copy()
            self.scale_factor = 1.0
        
        self.original_image = display_image.copy()
        self.temp_image = display_image.copy()
        
        # Create window and set mouse callback
        cv2.namedWindow('Select ROI', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('Select ROI', self.mouse_callback)
        
        print("\nInstructions:")
        print("- Click and drag to select ROI")
        print("- Press 'r' to reset selection")
        print("- Press 'ENTER' to confirm selection")
        print("- Press 'ESC' to skip ROI selection")
        
        cv2.imshow('Select ROI', self.temp_image)
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter key
                if self.roi is not None:
                    # Scale ROI back to original image size
                    x1, y1, x2, y2 = self.roi
                    if self.scale_factor != 1.0:
                        x1 = int(x1 / self.scale_factor)
                        y1 = int(y1 / self.scale_factor)
                        x2 = int(x2 / self.scale_factor)
                        y2 = int(y2 / self.scale_factor)
                    
                    cv2.destroyAllWindows()
                    print(f"ROI selected: ({x1}, {y1}) to ({x2}, {y2})")
                    print(f"ROI size: {x2-x1} x {y2-y1} pixels")
                    return (x1, y1, x2, y2)
                else:
                    print("Please select an ROI first!")
                    
            elif key == 27:  # ESC key
                cv2.destroyAllWindows()
                print("ROI selection skipped")
                return None
                
            elif key == ord('r'):  # Reset
                self.roi = None
                self.temp_image = self.original_image.copy()
                cv2.imshow('Select ROI', self.temp_image)
                print("Selection reset")

def get_example_image(image_dir: str, pattern: str) -> str:
    """Get the first image from directory for ROI selection"""
    images = sorted(glob.glob(os.path.join(image_dir, pattern)))
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir} with pattern {pattern}")
    return images[0]

def crop_images_in_directory(input_dir: str, output_dir: str, roi: tuple, pattern: str):
    """Crop all images in a directory using the specified ROI"""
    x1, y1, x2, y2 = roi
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images
    images = sorted(glob.glob(os.path.join(input_dir, pattern)))
    
    print(f"Cropping {len(images)} images from {os.path.basename(input_dir)}...")
    
    for img_path in images:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Cannot load {img_path}")
            continue
        
        # Crop image
        cropped = img[y1:y2, x1:x2]
        
        # Save cropped image
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, cropped)
    
    print(f"✓ Cropped images saved to: {output_dir}")

def visualize_calibration_coverage(left_images_dir: str, right_images_dir: str):
    """Create calibration coverage visualization"""
    if not CONFIG['create_visualizations']:
        return
    
    print(f"\n{'='*60}")
    print("CREATING CALIBRATION COVERAGE VISUALIZATION")
    print(f"{'='*60}")
    
    # Get calibration images
    left_images = sorted(glob.glob(os.path.join(left_images_dir, CONFIG['image_pattern'])))
    right_images = sorted(glob.glob(os.path.join(right_images_dir, CONFIG['image_pattern'])))
    
    if not left_images or not right_images:
        print("No images found for coverage visualization")
        return
    
    # Prepare chessboard detection
    chessboard_size = (CONFIG['chess_rows'], CONFIG['chess_cols'])
    
    # Get image size
    sample_img = cv2.imread(left_images[0])
    if sample_img is None:
        return
    
    img_height, img_width = sample_img.shape[:2]
    
    # Create coverage maps
    left_coverage = np.zeros((img_height, img_width), dtype=np.float32)
    right_coverage = np.zeros((img_height, img_width), dtype=np.float32)
    
    # Process each image pair
    valid_detections = 0
    all_left_corners = []
    all_right_corners = []
    
    print("Analyzing calibration coverage...")
    
    for left_path, right_path in zip(left_images, right_images):
        # Load images
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is None or right_img is None:
            continue
        
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        left_ret, left_corners = cv2.findChessboardCorners(left_gray, chessboard_size, None)
        right_ret, right_corners = cv2.findChessboardCorners(right_gray, chessboard_size, None)
        
        if left_ret and right_ret:
            valid_detections += 1
            
            # Add corners to coverage maps
            for corner in left_corners:
                x, y = int(corner[0][0]), int(corner[0][1])
                if 0 <= x < img_width and 0 <= y < img_height:
                    cv2.circle(left_coverage, (x, y), 20, 1, -1)
                    all_left_corners.append([x, y])
            
            for corner in right_corners:
                x, y = int(corner[0][0]), int(corner[0][1])
                if 0 <= x < img_width and 0 <= y < img_height:
                    cv2.circle(right_coverage, (x, y), 20, 1, -1)
                    all_right_corners.append([x, y])
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Calibration Coverage Analysis', fontsize=16, fontweight='bold')
    
    # Left camera coverage
    im1 = axes[0, 0].imshow(left_coverage, cmap='hot', alpha=0.7)
    axes[0, 0].set_title(f'Left Camera Coverage\n({valid_detections} valid detections)')
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Right camera coverage
    im2 = axes[0, 1].imshow(right_coverage, cmap='hot', alpha=0.7)
    axes[0, 1].set_title(f'Right Camera Coverage\n({valid_detections} valid detections)')
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Corner distribution
    if all_left_corners:
        all_left_corners = np.array(all_left_corners)
        axes[1, 0].scatter(all_left_corners[:, 0], all_left_corners[:, 1], 
                          alpha=0.6, s=10, c='blue')
        axes[1, 0].set_title('Left Camera Corner Distribution')
        axes[1, 0].set_xlabel('X (pixels)')
        axes[1, 0].set_ylabel('Y (pixels)')
        axes[1, 0].invert_yaxis()
        axes[1, 0].grid(True, alpha=0.3)
    
    if all_right_corners:
        all_right_corners = np.array(all_right_corners)
        axes[1, 1].scatter(all_right_corners[:, 0], all_right_corners[:, 1], 
                          alpha=0.6, s=10, c='red')
        axes[1, 1].set_title('Right Camera Corner Distribution')
        axes[1, 1].set_xlabel('X (pixels)')
        axes[1, 1].set_ylabel('Y (pixels)')
        axes[1, 1].invert_yaxis()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save coverage visualization
    coverage_file = os.path.join(CONFIG['output_dir'], 'calibration_coverage.png')
    plt.savefig(coverage_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Coverage visualization saved to: {coverage_file}")

def visualize_pipeline_summary():
    """Create a comprehensive pipeline summary visualization"""
    if not CONFIG['create_visualizations']:
        return
    
    print(f"\n{'='*60}")
    print("CREATING PIPELINE SUMMARY VISUALIZATION")
    print(f"{'='*60}")
    
    # Get sample images
    try:
        # Original images
        left_orig = get_example_image(CONFIG['left_images'], CONFIG['image_pattern'])
        right_orig = get_example_image(CONFIG['right_images'], CONFIG['image_pattern'])
        
        # Cropped images (if ROI was used)
        if CONFIG['use_roi']:
            cropped_left_dir = os.path.join(CONFIG['output_dir'], 'cropped_left')
            cropped_right_dir = os.path.join(CONFIG['output_dir'], 'cropped_right')
            left_cropped = get_example_image(cropped_left_dir, CONFIG['image_pattern'])
            right_cropped = get_example_image(cropped_right_dir, CONFIG['image_pattern'])
        else:
            left_cropped = left_orig
            right_cropped = right_orig
        
    except FileNotFoundError as e:
        print(f"Error loading images for visualization: {e}")
        return
    
    # Load stereo calibration results
    stereo_file = os.path.join(CONFIG['output_dir'], 'stereo_calibration.npz')
    if not os.path.exists(stereo_file):
        print("Stereo calibration file not found for visualization")
        return
    
    stereo_data = np.load(stereo_file)
    
    # Load images
    left_img_orig = cv2.imread(left_orig)
    right_img_orig = cv2.imread(right_orig)
    left_img_cropped = cv2.imread(left_cropped)
    right_img_cropped = cv2.imread(right_cropped)
    
    if any(img is None for img in [left_img_orig, right_img_orig, left_img_cropped, right_img_cropped]):
        print("Error loading images for pipeline visualization")
        return
    
    # Convert BGR to RGB for matplotlib
    left_img_orig = cv2.cvtColor(left_img_orig, cv2.COLOR_BGR2RGB)
    right_img_orig = cv2.cvtColor(right_img_orig, cv2.COLOR_BGR2RGB)
    left_img_cropped = cv2.cvtColor(left_img_cropped, cv2.COLOR_BGR2RGB)
    right_img_cropped = cv2.cvtColor(right_img_cropped, cv2.COLOR_BGR2RGB)
    
    # Create rectified images
    left_map1 = stereo_data['left_map1']
    left_map2 = stereo_data['left_map2']
    right_map1 = stereo_data['right_map1']
    right_map2 = stereo_data['right_map2']
    
    left_rectified = cv2.remap(left_img_cropped, left_map1, left_map2, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_img_cropped, right_map1, right_map2, cv2.INTER_LINEAR)
    
    # Create the visualization
    if CONFIG['use_roi']:
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        fig.suptitle('Calibration Pipeline Summary', fontsize=20, fontweight='bold')
        
        # Original images
        axes[0, 0].imshow(left_img_orig)
        axes[0, 0].set_title(f'Original Left ({CONFIG["left_camera_name"]})', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(right_img_orig)
        axes[0, 1].set_title(f'Original Right ({CONFIG["right_camera_name"]})', fontsize=14)
        axes[0, 1].axis('off')
        
        # Cropped images
        axes[1, 0].imshow(left_img_cropped)
        axes[1, 0].set_title('Cropped Left (ROI Applied)', fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(right_img_cropped)
        axes[1, 1].set_title('Cropped Right (ROI Applied)', fontsize=14)
        axes[1, 1].axis('off')
        
        # Rectified images
        axes[2, 0].imshow(left_rectified)
        axes[2, 0].set_title('Rectified Left', fontsize=14)
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(right_rectified)
        axes[2, 1].set_title('Rectified Right', fontsize=14)
        axes[2, 1].axis('off')
        
    else:
        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        fig.suptitle('Calibration Pipeline Summary', fontsize=20, fontweight='bold')
        
        # Original images
        axes[0, 0].imshow(left_img_orig)
        axes[0, 0].set_title(f'Original Left ({CONFIG["left_camera_name"]})', fontsize=14)
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(right_img_orig)
        axes[0, 1].set_title(f'Original Right ({CONFIG["right_camera_name"]})', fontsize=14)
        axes[0, 1].axis('off')
        
        # Rectified images
        axes[1, 0].imshow(left_rectified)
        axes[1, 0].set_title('Rectified Left', fontsize=14)
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(right_rectified)
        axes[1, 1].set_title('Rectified Right', fontsize=14)
        axes[1, 1].axis('off')
    
    # Add epipolar lines to rectified images
    add_epipolar_lines(axes[-1, 0], left_rectified.shape)
    add_epipolar_lines(axes[-1, 1], right_rectified.shape)
    
    plt.tight_layout()
    
    # Save pipeline visualization
    pipeline_file = os.path.join(CONFIG['output_dir'], 'pipeline_summary.png')
    plt.savefig(pipeline_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Pipeline summary saved to: {pipeline_file}")
    
    # Create rectified stereo pair for depth visualization
    create_stereo_pair_visualization(left_rectified, right_rectified)

def add_epipolar_lines(ax, img_shape):
    """Add horizontal epipolar lines to show rectification quality"""
    height, width = img_shape[:2]
    
    # Add horizontal lines every 50 pixels
    for y in range(50, height, 100):
        ax.axhline(y=y, color='lime', linewidth=1, alpha=0.7)
    
    ax.text(10, 30, 'Epipolar Lines', color='lime', fontsize=10, 
           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

def create_stereo_pair_visualization(left_rectified, right_rectified):
    """Create side-by-side stereo pair for cross-eyed viewing"""
    
    # Resize images for comfortable viewing
    target_width = 400
    height, width = left_rectified.shape[:2]
    scale = target_width / width
    new_height = int(height * scale)
    
    left_small = cv2.resize(left_rectified, (target_width, new_height))
    right_small = cv2.resize(right_rectified, (target_width, new_height))
    
    # Create side-by-side stereo pair
    stereo_pair = np.hstack([left_small, right_small])
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.imshow(stereo_pair)
    ax.set_title('Rectified Stereo Pair\n(For cross-eyed stereo viewing)', fontsize=14)
    ax.axis('off')
    
    # Add center line
    center_x = target_width
    ax.axvline(x=center_x, color='white', linewidth=2, alpha=0.8)
    
    # Add labels
    ax.text(target_width//2, 20, 'Left', ha='center', va='top', color='white', 
           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    ax.text(target_width + target_width//2, 20, 'Right', ha='center', va='top', color='white',
           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    plt.tight_layout()
    
    # Save stereo pair
    stereo_file = os.path.join(CONFIG['output_dir'], 'stereo_pair.png')
    plt.savefig(stereo_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Stereo pair visualization saved to: {stereo_file}")

def create_calibration_summary_report():
    """Create a comprehensive calibration summary with key metrics"""
    
    summary_file = os.path.join(CONFIG['output_dir'], 'calibration_summary.txt')
    
    try:
        # Load calibration data
        left_calib_file = os.path.join(CONFIG['output_dir'], f"{CONFIG['left_camera_name']}_calibration.npz")
        right_calib_file = os.path.join(CONFIG['output_dir'], f"{CONFIG['right_camera_name']}_calibration.npz")
        stereo_file = os.path.join(CONFIG['output_dir'], 'stereo_calibration.npz')
        
        left_data = np.load(left_calib_file)
        right_data = np.load(right_calib_file)
        stereo_data = np.load(stereo_file)
        
        with open(summary_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE CALIBRATION SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            # Configuration
            f.write("CONFIGURATION:\n")
            f.write("-"*40 + "\n")
            f.write(f"Left camera:     {CONFIG['left_camera_name']}\n")
            f.write(f"Right camera:    {CONFIG['right_camera_name']}\n")
            f.write(f"Chessboard:      {CONFIG['chess_rows']}x{CONFIG['chess_cols']}\n")
            f.write(f"Square size:     {CONFIG['square_size']} mm\n")
            f.write(f"ROI used:        {CONFIG['use_roi']}\n")
            f.write(f"Image size:      {stereo_data['image_size']}\n\n")
            
            # Single camera results
            f.write("SINGLE CAMERA CALIBRATION RESULTS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Left camera RMS error:   {left_data['calibration_error']:.4f} pixels\n")
            f.write(f"Right camera RMS error:  {right_data['calibration_error']:.4f} pixels\n")
            f.write(f"Left mean reproj error:  {left_data['mean_reprojection_error']:.4f} pixels\n")
            f.write(f"Right mean reproj error: {right_data['mean_reprojection_error']:.4f} pixels\n\n")
            
            # Stereo results
            f.write("STEREO CALIBRATION RESULTS:\n")
            f.write("-"*40 + "\n")
            f.write(f"Stereo RMS error:        {stereo_data['stereo_error']:.4f} pixels\n")
            f.write(f"Baseline:                {stereo_data['baseline']:.2f} mm\n")
            f.write(f"Valid stereo pairs:      {stereo_data['valid_pairs']}\n\n")
            
            # Quality assessment
            f.write("QUALITY ASSESSMENT:\n")
            f.write("-"*40 + "\n")
            
            max_single_error = max(left_data['calibration_error'], right_data['calibration_error'])
            stereo_error = stereo_data['stereo_error']
            
            if max_single_error < 0.5 and stereo_error < 1.0:
                f.write("Overall Quality: ★★★ EXCELLENT\n")
            elif max_single_error < 1.0 and stereo_error < 2.0:
                f.write("Overall Quality: ★★☆ GOOD\n")
            elif max_single_error < 2.0 and stereo_error < 3.0:
                f.write("Overall Quality: ★☆☆ ACCEPTABLE\n")
            else:
                f.write("Overall Quality: ☆☆☆ NEEDS IMPROVEMENT\n")
            
            f.write(f"\nRecommendation: ")
            if stereo_error < 2.0:
                f.write("Calibration is ready for production use.\n")
            else:
                f.write("Consider recalibrating with more images or better coverage.\n")
            
            f.write("\n" + "="*80 + "\n")
        
        print(f"✓ Calibration summary saved to: {summary_file}")
        
    except Exception as e:
        print(f"Warning: Could not create calibration summary: {e}")

def save_roi_coordinates(left_roi: tuple, right_roi: tuple):
    """Save ROI coordinates for future reuse"""
    roi_file = os.path.join(CONFIG['output_dir'], CONFIG['roi_save_file'])
    
    np.savez(roi_file,
             left_roi=np.array(left_roi),
             right_roi=np.array(right_roi),
             left_camera_name=CONFIG['left_camera_name'],
             right_camera_name=CONFIG['right_camera_name'],
             image_pattern=CONFIG['image_pattern'],
             roi_margin=CONFIG['roi_margin'])
    
    print(f"✓ ROI coordinates saved to: {roi_file}")

def load_roi_coordinates():
    """Load previously saved ROI coordinates"""
    roi_file = os.path.join(CONFIG['output_dir'], CONFIG['roi_save_file'])
    
    if not os.path.exists(roi_file):
        return None, None
    
    try:
        data = np.load(roi_file, allow_pickle=True)
        
        # Verify the saved ROI is compatible with current configuration
        if (str(data['left_camera_name']) == CONFIG['left_camera_name'] and
            str(data['right_camera_name']) == CONFIG['right_camera_name'] and
            str(data['image_pattern']) == CONFIG['image_pattern']):
            
            left_roi = tuple(data['left_roi'])
            right_roi = tuple(data['right_roi'])
            saved_margin = int(data['roi_margin'])
            
            print(f"✓ Found saved ROI coordinates:")
            print(f"  Left ROI: {left_roi}")
            print(f"  Right ROI: {right_roi}")
            print(f"  Saved margin: {saved_margin} pixels")
            print(f"  Current margin: {CONFIG['roi_margin']} pixels")
            
            if saved_margin != CONFIG['roi_margin']:
                print(f"⚠ Warning: ROI margin changed from {saved_margin} to {CONFIG['roi_margin']}")
            
            return left_roi, right_roi
        else:
            print("⚠ Saved ROI configuration doesn't match current settings")
            return None, None
            
    except Exception as e:
        print(f"⚠ Error loading saved ROI: {e}")
        return None, None

def prompt_roi_reuse(left_roi: tuple, right_roi: tuple) -> bool:
    """Ask user if they want to reuse saved ROI"""
    print(f"\nFound saved ROI coordinates:")
    print(f"Left camera ({CONFIG['left_camera_name']}): {left_roi}")
    print(f"  Size: {left_roi[2]-left_roi[0]} x {left_roi[3]-left_roi[1]} pixels")
    print(f"Right camera ({CONFIG['right_camera_name']}): {right_roi}")
    print(f"  Size: {right_roi[2]-right_roi[0]} x {right_roi[3]-right_roi[1]} pixels")
    
    while True:
        response = input("\nReuse saved ROI? (y/n/preview): ").lower().strip()
        
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        elif response in ['p', 'preview']:
            preview_saved_roi(left_roi, right_roi)
        else:
            print("Please enter 'y' for yes, 'n' for no, or 'p' to preview")

def preview_saved_roi(left_roi: tuple, right_roi: tuple):
    """Show preview of saved ROI on example images"""
    try:
        left_example = get_example_image(CONFIG['left_images'], CONFIG['image_pattern'])
        right_example = get_example_image(CONFIG['right_images'], CONFIG['image_pattern'])
        
        # Load and show left image with ROI
        left_img = cv2.imread(left_example)
        if left_img is not None:
            x1, y1, x2, y2 = left_roi
            left_preview = left_img.copy()
            cv2.rectangle(left_preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(left_preview, f'Saved ROI: {x2-x1}x{y2-y1}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Resize for display if needed
            height, width = left_preview.shape[:2]
            if width > 1200 or height > 1200:
                scale = 1200 / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                left_preview = cv2.resize(left_preview, (new_width, new_height))
            
            cv2.imshow('Left Camera - Saved ROI Preview', left_preview)
        
        # Load and show right image with ROI
        right_img = cv2.imread(right_example)
        if right_img is not None:
            x1, y1, x2, y2 = right_roi
            right_preview = right_img.copy()
            cv2.rectangle(right_preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(right_preview, f'Saved ROI: {x2-x1}x{y2-y1}', (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Resize for display if needed
            height, width = right_preview.shape[:2]
            if width > 1200 or height > 1200:
                scale = 1200 / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                right_preview = cv2.resize(right_preview, (new_width, new_height))
            
            cv2.imshow('Right Camera - Saved ROI Preview', right_preview)
        
        print("Press any key to close preview windows...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error showing ROI preview: {e}")

def setup_roi_and_crop():
    """Setup ROI selection and crop images"""
    if not CONFIG['use_roi']:
        print("ROI selection disabled. Using original images.")
        return CONFIG['left_images'], CONFIG['right_images']
    
    print(f"\n{'='*60}")
    print("ROI SELECTION AND IMAGE CROPPING")
    print(f"{'='*60}")
    
    # Try to load saved ROI if enabled
    left_roi = None
    right_roi = None
    
    if CONFIG['reuse_saved_roi']:
        left_roi, right_roi = load_roi_coordinates()
        
        if left_roi is not None and right_roi is not None:
            # Ask user if they want to reuse saved ROI
            if prompt_roi_reuse(left_roi, right_roi):
                print("✓ Using saved ROI coordinates")
            else:
                print("Using new ROI selection")
                left_roi = None
                right_roi = None
    
    # If no saved ROI or user chose new selection, do interactive selection
    if left_roi is None or right_roi is None:
        # Get example images for ROI selection
        try:
            left_example = get_example_image(CONFIG['left_images'], CONFIG['image_pattern'])
            right_example = get_example_image(CONFIG['right_images'], CONFIG['image_pattern'])
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return None, None
        
        # ROI selection for left camera
        print(f"\nSelect ROI for LEFT camera ({CONFIG['left_camera_name']}):")
        roi_selector = ROISelector()
        left_roi = roi_selector.select_roi(left_example)
        
        if left_roi is None:
            print("No ROI selected for left camera. Using original images.")
            return CONFIG['left_images'], CONFIG['right_images']
        
        # ROI selection for right camera
        print(f"\nSelect ROI for RIGHT camera ({CONFIG['right_camera_name']}):")
        roi_selector = ROISelector()
        right_roi = roi_selector.select_roi(right_example)
        
        if right_roi is None:
            print("No ROI selected for right camera. Using original images.")
            return CONFIG['left_images'], CONFIG['right_images']
        
        # Save the new ROI coordinates
        save_roi_coordinates(left_roi, right_roi)
    
    # Create cropped image directories
    cropped_left_dir = os.path.join(CONFIG['output_dir'], 'cropped_left')
    cropped_right_dir = os.path.join(CONFIG['output_dir'], 'cropped_right')
    
    # Check if cropped images already exist and are up to date
    if (os.path.exists(cropped_left_dir) and os.path.exists(cropped_right_dir) and
        len(glob.glob(os.path.join(cropped_left_dir, CONFIG['image_pattern']))) > 0):
        
        print(f"✓ Found existing cropped images")
        
        # Ask if user wants to use existing cropped images
        while True:
            response = input("Use existing cropped images? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                print("✓ Using existing cropped images")
                return cropped_left_dir, cropped_right_dir
            elif response in ['n', 'no']:
                print("Recreating cropped images...")
                break
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    # Crop images
    crop_images_in_directory(CONFIG['left_images'], cropped_left_dir, left_roi, CONFIG['image_pattern'])
    crop_images_in_directory(CONFIG['right_images'], cropped_right_dir, right_roi, CONFIG['image_pattern'])
    
    # Save ROI information (text file for human reading)
    roi_info_file = os.path.join(CONFIG['output_dir'], 'roi_info.txt')
    with open(roi_info_file, 'w') as f:
        f.write("ROI INFORMATION\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Left camera ({CONFIG['left_camera_name']}) ROI:\n")
        f.write(f"  Top-left: ({left_roi[0]}, {left_roi[1]})\n")
        f.write(f"  Bottom-right: ({left_roi[2]}, {left_roi[3]})\n")
        f.write(f"  Size: {left_roi[2]-left_roi[0]} x {left_roi[3]-left_roi[1]} pixels\n\n")
        
        f.write(f"Right camera ({CONFIG['right_camera_name']}) ROI:\n")
        f.write(f"  Top-left: ({right_roi[0]}, {right_roi[1]})\n")
        f.write(f"  Bottom-right: ({right_roi[2]}, {right_roi[3]})\n")
        f.write(f"  Size: {right_roi[2]-right_roi[0]} x {right_roi[3]-right_roi[1]} pixels\n\n")
        
        f.write(f"Margin applied: {CONFIG['roi_margin']} pixels\n")
    
    print(f"✓ ROI information saved to: {roi_info_file}")
    
    return cropped_left_dir, cropped_right_dir

class SimpleArgs:
    """Simple class to mimic argparse arguments"""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def create_args_for_single_camera(camera_name: str, image_dir: str) -> SimpleArgs:
    """Create arguments object for single camera calibration"""
    return SimpleArgs(
        image_dir=image_dir,
        camera_name=camera_name,
        output_dir=CONFIG['output_dir'],
        chess_rows=CONFIG['chess_rows'],
        chess_cols=CONFIG['chess_cols'],
        square_size=CONFIG['square_size'],
        image_pattern=CONFIG['image_pattern']
    )

def create_args_for_stereo(left_images: str, right_images: str) -> SimpleArgs:
    """Create arguments object for stereo calibration"""
    left_calib_file = os.path.join(CONFIG['output_dir'], f"{CONFIG['left_camera_name']}_calibration.npz")
    right_calib_file = os.path.join(CONFIG['output_dir'], f"{CONFIG['right_camera_name']}_calibration.npz")
    
    return SimpleArgs(
        left_images=left_images,
        right_images=right_images,
        left_calib=left_calib_file,
        right_calib=right_calib_file,
        output_dir=CONFIG['output_dir'],
        chess_rows=CONFIG['chess_rows'],
        chess_cols=CONFIG['chess_cols'],
        square_size=CONFIG['square_size'],
        image_pattern=CONFIG['image_pattern']
    )

def validate_directories():
    """Validate that input directories exist"""
    errors = []
    
    if not os.path.exists(CONFIG['left_images']):
        errors.append(f"Left image directory does not exist: {CONFIG['left_images']}")
    
    if not os.path.exists(CONFIG['right_images']):
        errors.append(f"Right image directory does not exist: {CONFIG['right_images']}")
    
    if errors:
        for error in errors:
            print(f"✗ {error}")
        return False
    
    return True

def run_single_calibration(camera_name: str, image_dir: str) -> bool:
    """Run single camera calibration"""
    print(f"\n{'='*60}")
    print(f"RUNNING SINGLE CAMERA CALIBRATION: {camera_name.upper()}")
    print(f"{'='*60}")
    
    try:
        args = create_args_for_single_camera(camera_name, image_dir)
        result = single_camera_calibrate(args)
        
        if result is not None:
            print(f"✓ {camera_name} camera calibration completed successfully!")
            return True
        else:
            print(f"✗ {camera_name} camera calibration failed!")
            return False
            
    except Exception as e:
        print(f"✗ Error during {camera_name} camera calibration: {str(e)}")
        return False

def run_stereo_calibration(left_images: str, right_images: str) -> bool:
    """Run stereo camera calibration"""
    print(f"\n{'='*60}")
    print(f"RUNNING STEREO CAMERA CALIBRATION")
    print(f"{'='*60}")
    
    try:
        args = create_args_for_stereo(left_images, right_images)
        result = stereo_calibrate(args)
        
        if result is not None:
            print("✓ Stereo calibration completed successfully!")
            return True
        else:
            print("✗ Stereo calibration failed!")
            return False
            
    except Exception as e:
        print(f"✗ Error during stereo calibration: {str(e)}")
        return False

def print_configuration():
    """Print current configuration"""
    print("CAMERA CALIBRATION PIPELINE")
    print("="*60)
    print(f"Left images:      {CONFIG['left_images']}")
    print(f"Right images:     {CONFIG['right_images']}")
    print(f"Output directory: {CONFIG['output_dir']}")
    print(f"Chessboard:       {CONFIG['chess_rows']}x{CONFIG['chess_cols']}")
    print(f"Square size:      {CONFIG['square_size']} mm")
    print(f"Image pattern:    {CONFIG['image_pattern']}")
    print(f"Left camera:      {CONFIG['left_camera_name']}")
    print(f"Right camera:     {CONFIG['right_camera_name']}")
    print(f"Use ROI:          {CONFIG['use_roi']}")
    print(f"Create visualizations: {CONFIG['create_visualizations']}")
    if CONFIG['use_roi']:
        print(f"ROI margin:       {CONFIG['roi_margin']} pixels")

def print_results():
    """Print final results summary"""
    print(f"\n{'='*60}")
    print("CALIBRATION PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*60}")
    
    left_calib = os.path.join(CONFIG['output_dir'], f"{CONFIG['left_camera_name']}_calibration.npz")
    right_calib = os.path.join(CONFIG['output_dir'], f"{CONFIG['right_camera_name']}_calibration.npz")
    stereo_calib = os.path.join(CONFIG['output_dir'], 'stereo_calibration.npz')
    
    print(f"✓ Left camera calibration:  {left_calib}")
    print(f"✓ Right camera calibration: {right_calib}")
    print(f"✓ Stereo calibration:       {stereo_calib}")
    print(f"✓ All results saved in:     {CONFIG['output_dir']}")
    
    print(f"\nGenerated files:")
    print(f"  - {CONFIG['left_camera_name']}_calibration.npz")
    print(f"  - {CONFIG['left_camera_name']}_calibration_report.txt")
    print(f"  - {CONFIG['right_camera_name']}_calibration.npz")
    print(f"  - {CONFIG['right_camera_name']}_calibration_report.txt")
    print(f"  - stereo_calibration.npz")
    print(f"  - stereo_calibration_report.txt")
    print(f"  - calibration_summary.txt")
    
    if CONFIG['use_roi']:
        print(f"  - roi_info.txt")
        print(f"  - cropped_left/ (cropped images)")
        print(f"  - cropped_right/ (cropped images)")
    
    if CONFIG['create_visualizations']:
        print(f"  - pipeline_summary.png")
        print(f"  - calibration_coverage.png")
        print(f"  - stereo_pair.png")

def main():
    """Main calibration pipeline"""
    print_configuration()
    
    # Create output directory
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # Validate input directories
    if not validate_directories():
        return False
    
    # Setup ROI and get image directories to use
    left_images_dir, right_images_dir = setup_roi_and_crop()
    
    if left_images_dir is None or right_images_dir is None:
        print("✗ ROI setup failed. Stopping pipeline.")
        return False
    
    # Step 1: Single camera calibration for left camera
    if not run_single_calibration(CONFIG['left_camera_name'], left_images_dir):
        print(f"✗ Left camera calibration failed. Stopping pipeline.")
        return False
    
    # Step 2: Single camera calibration for right camera
    if not run_single_calibration(CONFIG['right_camera_name'], right_images_dir):
        print(f"✗ Right camera calibration failed. Stopping pipeline.")
        return False
    
    # Step 3: Stereo calibration
    if not run_stereo_calibration(left_images_dir, right_images_dir):
        print(f"✗ Stereo calibration failed. Stopping pipeline.")
        return False
    
    # Step 4: Create visualizations
    if CONFIG['create_visualizations']:
        visualize_calibration_coverage(left_images_dir, right_images_dir)
        visualize_pipeline_summary()
    
    # Step 5: Create comprehensive summary
    create_calibration_summary_report()
    
    # Success!
    print_results()
    print("\n🎉 All calibrations completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Calibration pipeline failed!")
        sys.exit(1)