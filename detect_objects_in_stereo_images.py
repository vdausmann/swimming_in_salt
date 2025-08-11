import os
import cv2
import glob
import numpy as np
import pandas as pd
import re
from tqdm import tqdm
from typing import Dict

def load_rectification(rectification_file):
    rect_data = np.load(rectification_file, allow_pickle=True)
    return rect_data['map1_x'], rect_data['map1_y'], rect_data['map2_x'], rect_data['map2_y']

def load_roi_coordinates(roi_file):
    roi_data = np.load(roi_file, allow_pickle=True)
    horizontal_lines_upper = roi_data['horizontal_lines_upper']
    horizontal_lines_lower = roi_data['horizontal_lines_lower']
    vertical_lines = roi_data['vertical_lines']
    return (
        int(vertical_lines[0]),              # left_bound
        int(horizontal_lines_upper[0]),      # upper_bound_top
        int(horizontal_lines_upper[1]),      # lower_bound_top
        int(horizontal_lines_lower[0]),      # upper_bound_bottom
        int(horizontal_lines_lower[1]),      # lower_bound_bottom
        int(vertical_lines[1])               # right_bound
    )

def get_valid_roi(map1_x, map1_y, map2_x, map2_y, roi_file=None):
    if roi_file:
        return load_roi_coordinates(roi_file)
    h, w = map1_x.shape[:2]
    valid1 = (map1_x >= 0) & (map1_x < w) & (map1_y >= 0) & (map1_y < h)
    valid2 = (map2_x >= 0) & (map2_x < w) & (map2_y >= 0) & (map2_y < h)
    valid = valid1 & valid2
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return (xmin, ymin, xmax, ymax)

def compute_average_background(image_stack):
    image_stack = np.array(image_stack, dtype=np.float32)
    return np.mean(image_stack, axis=0).astype(np.uint8)

def extract_timestamp(filename):
    """Extract timestamp from filename in format frame_YYYYMMDD_HHMMSS_microseconds_*.jpg"""
    match = re.search(r'frame_(\d{8})_(\d{6})_(\d+)_', os.path.basename(filename))
    if match:
        date, time, microseconds = match.groups()
        return f"{date}_{time}_{microseconds}"
    return None

def save_detection_report(
    output_file: str,
    parameters: Dict,
    detection_stats: Dict,
    total_frames: int
):
    """Save a detailed detection report with parameters and statistics."""
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("OBJECT DETECTION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Parameters section
        f.write("DETECTION PARAMETERS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Input directory: {parameters['input_dir']}\n")
        f.write(f"Output directory: {parameters['output_dir']}\n")
        f.write(f"Threshold value: {parameters['threshold']}\n")
        f.write(f"Minimum area: {parameters['min_area']} pixels²\n")
        f.write(f"Rectification file: {os.path.basename(parameters['rectification_file'])}\n")
        f.write(f"ROI file: {os.path.basename(parameters.get('roi_file', 'N/A'))}\n")
        f.write(f"Total frames processed: {total_frames}\n\n")
        
        # Overall statistics
        f.write("DETECTION STATISTICS:\n")
        f.write("-" * 30 + "\n")
        
        # Calculate totals
        total_detections = sum(detection_stats['upper_detections'] + detection_stats['lower_detections'])
        upper_total = sum(detection_stats['upper_detections'])
        lower_total = sum(detection_stats['lower_detections'])
        
        f.write(f"★ Total objects detected: {total_detections}\n")
        f.write(f"★ Upper chamber detections: {upper_total}\n")
        f.write(f"★ Lower chamber detections: {lower_total}\n")
        
        if total_frames > 0:
            f.write(f"★ Average objects per frame: {total_detections/total_frames:.2f}\n")
            f.write(f"★ Average upper per frame: {upper_total/total_frames:.2f}\n")
            f.write(f"★ Average lower per frame: {lower_total/total_frames:.2f}\n")
        
        # Frame-by-frame statistics
        if detection_stats['upper_detections']:
            f.write(f"★ Max objects in single frame (upper): {max(detection_stats['upper_detections'])}\n")
            f.write(f"★ Max objects in single frame (lower): {max(detection_stats['lower_detections'])}\n")
        
        # Area statistics
        if detection_stats['upper_areas']:
            f.write(f"★ Average object area (upper): {np.mean(detection_stats['upper_areas']):.1f} pixels²\n")
            f.write(f"★ Average object area (lower): {np.mean(detection_stats['lower_areas']):.1f} pixels²\n")
            f.write(f"★ Largest object area: {max(detection_stats['upper_areas'] + detection_stats['lower_areas']):.1f} pixels²\n")
        
        f.write("\n")
        
        # Frame-by-frame breakdown
        f.write("FRAME-BY-FRAME BREAKDOWN:\n")
        f.write("-" * 30 + "\n")
        f.write("Frame | Upper Count | Lower Count | Total\n")
        f.write("-" * 40 + "\n")
        
        for i in range(min(20, len(detection_stats['upper_detections']))):  # Show first 20 frames
            upper_count = detection_stats['upper_detections'][i]
            lower_count = detection_stats['lower_detections'][i]
            total_count = upper_count + lower_count
            f.write(f"{i:5d} | {upper_count:11d} | {lower_count:11d} | {total_count:5d}\n")
        
        if len(detection_stats['upper_detections']) > 20:
            f.write("... (showing first 20 frames only)\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Report generated successfully!\n")

def detect_and_save(
    input_dir,
    output_dir,
    rectification_file,
    roi_file=None,
    threshold=30,
    min_area=2
):
    os.makedirs(output_dir, exist_ok=True)
    
    # Store parameters for report
    detection_params = {
        'input_dir': input_dir,
        'output_dir': output_dir,
        'rectification_file': rectification_file,
        'roi_file': roi_file,
        'threshold': threshold,
        'min_area': min_area
    }
    
    # Initialize statistics tracking
    detection_stats = {
        'upper_detections': [],
        'lower_detections': [],
        'upper_areas': [],
        'lower_areas': []
    }
    
    left_dir = os.path.join(input_dir, 'left')
    right_dir = os.path.join(input_dir, 'right')
    left_images = sorted(glob.glob(os.path.join(left_dir, '*.jpg')))
    right_images = sorted(glob.glob(os.path.join(right_dir, '*.jpg')))
    if not left_images or not right_images:
        print("No images found!")
        return

    map1_x, map1_y, map2_x, map2_y = load_rectification(rectification_file)
    roi = get_valid_roi(map1_x, map1_y, map2_x, map2_y, roi_file)
    left_bound, upper_bound_top, lower_bound_top, upper_bound_bottom, lower_bound_bottom, right_bound = roi

    # Rectify and stack for background
    upper_rect_stack, lower_rect_stack = [], []
    for u_file, l_file in tqdm(zip(left_images, right_images), desc="Rectifying"):
        u_img = cv2.imread(u_file)
        l_img = cv2.imread(l_file)
        if u_img is None or l_img is None:
            continue
        u_rect = cv2.remap(u_img, map1_x, map1_y, cv2.INTER_LINEAR)
        l_rect = cv2.remap(l_img, map2_x, map2_y, cv2.INTER_LINEAR)
        u_gray = cv2.cvtColor(u_rect, cv2.COLOR_BGR2GRAY)
        l_gray = cv2.cvtColor(l_rect, cv2.COLOR_BGR2GRAY)
        upper_rect_stack.append(u_gray)
        lower_rect_stack.append(l_gray)
    avg_bg_upper = compute_average_background(upper_rect_stack)
    avg_bg_lower = compute_average_background(lower_rect_stack)

    for idx, (u_file, l_file) in tqdm(enumerate(zip(left_images, right_images)), total=len(left_images), desc="Processing"):
        # Extract timestamp from filename
        timestamp = extract_timestamp(u_file)
        if not timestamp:
            timestamp = f"{idx:04d}"  # Fallback if timestamp extraction fails
            
        u_rect_gray = upper_rect_stack[idx]
        l_rect_gray = lower_rect_stack[idx]
        u_corrected = cv2.absdiff(u_rect_gray, avg_bg_upper)
        l_corrected = cv2.absdiff(l_rect_gray, avg_bg_lower)
        _, u_thresh = cv2.threshold(u_corrected, threshold, 255, cv2.THRESH_BINARY)
        _, l_thresh = cv2.threshold(l_corrected, threshold, 255, cv2.THRESH_BINARY)
        u_thresh = cv2.rotate(u_thresh, cv2.ROTATE_90_COUNTERCLOCKWISE)
        l_thresh = cv2.rotate(l_thresh, cv2.ROTATE_90_COUNTERCLOCKWISE)
        u_crop = u_thresh[upper_bound_top+90:lower_bound_top, left_bound+120:right_bound-70]
        l_crop = l_thresh[upper_bound_bottom:lower_bound_bottom-20, left_bound+120:right_bound-70]

        # Save processed images with timestamp in filename
        u_out = os.path.join(output_dir, f"upper_{timestamp}.png")
        l_out = os.path.join(output_dir, f"lower_{timestamp}.png")
        cv2.imwrite(u_out, u_crop)
        cv2.imwrite(l_out, l_crop)

        # Initialize frame counters
        frame_upper_count = 0
        frame_lower_count = 0

        # Detect objects and save CSV
        for label, crop, prefix in [
            ('upper', u_crop, 'upper'),
            ('lower', l_crop, 'lower')
        ]:
            num_labels, labels_img, stats, centroids = cv2.connectedComponentsWithStats(crop)
            detections = []
            for i in range(1, num_labels):  # skip background
                area = stats[i, cv2.CC_STAT_AREA]
                if area >= min_area:
                    cx, cy = map(int, centroids[i])
                    cxf, cyf = centroids[i]
                    object_id = f"{prefix}_{timestamp}_obj{i-1}"
                    detections.append({'object_id': object_id, 'x': cxf, 'y': cyf, 'area': area})
                    
                    # Track statistics
                    if prefix == 'upper':
                        frame_upper_count += 1
                        detection_stats['upper_areas'].append(area)
                    else:
                        frame_lower_count += 1
                        detection_stats['lower_areas'].append(area)
            
            df = pd.DataFrame(detections)
            csv_out = os.path.join(output_dir, f"{prefix}_{timestamp}_objects.csv")
            df.to_csv(csv_out, index=False)
        
        # Record frame statistics
        detection_stats['upper_detections'].append(frame_upper_count)
        detection_stats['lower_detections'].append(frame_lower_count)
    
    # Save detection report
    save_detection_report(
        os.path.join(output_dir, "detection_report.txt"),
        detection_params,
        detection_stats,
        len(left_images)
    )
    
    print(f"✓ Detection completed: {len(left_images)} frames processed")
    print(f"✓ Total objects detected: {sum(detection_stats['upper_detections']) + sum(detection_stats['lower_detections'])}")
    print(f"✓ Report saved to: detection_report.txt")

if __name__ == "__main__":
    detect_and_save(
        input_dir="/Users/vdausmann/oyster_project/images/20250404_24.4_24_01",
        output_dir="/Users/vdausmann/swimming_in_salt/detection_results/20250404_24.4_24_01",
        rectification_file="/Users/vdausmann/swimming_in_salt/calibration/stereo_rectification.npz",
        roi_file="/Users/vdausmann/swimming_in_salt/calibration/roi_coordinates.npz",
        threshold=10,
        min_area=5
    )