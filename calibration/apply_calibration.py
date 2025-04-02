import os
import cv2
import glob
import numpy as np
import math
from typing import List, Dict, Tuple
from tqdm import tqdm

def compute_average_background(image_stack):
    image_stack = np.array(image_stack, dtype=np.float32)
    return np.mean(image_stack, axis=0).astype(np.uint8)

def load_roi_coordinates(roi_file):
    """Load the manually defined ROI coordinates"""
    roi_data = np.load(roi_file)
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
    """Get the valid region of interest after rectification"""
    if roi_file:
        return load_roi_coordinates(roi_file)
    
    # Fallback to automatic ROI detection if no file provided
    h, w = map1_x.shape[:2]
    valid1 = (map1_x >= 0) & (map1_x < w) & (map1_y >= 0) & (map1_y < h)
    valid2 = (map2_x >= 0) & (map2_x < w) & (map2_y >= 0) & (map2_y < h)
    valid = valid1 & valid2
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    return (xmin, ymin, xmax, ymax)

class TrackedObject:
    def __init__(self, match: Tuple[Dict, Dict], frame_idx: int):#, calib_params: Dict):
        self.upper_positions = [(match[0]['x'], match[0]['y'])]
        self.lower_positions = [(match[1]['x'], match[1]['y'])]
        self.areas = [(match[0]['area'], match[1]['area'])]
        self.last_frame = frame_idx
        self.timestamps = []
        self.velocities = []  # (vx, vy, vz) in mm/s
        
        # Store calibration parameters
        # self.focal_length = calib_params['focal_length']
        # self.baseline = calib_params['baseline']
        # self.pixel_size = calib_params['pixel_size']
        
    def add_match(self, match: Tuple[Dict, Dict], frame_idx: int):
        self.upper_positions.append((match[0]['x'], match[0]['y']))
        self.lower_positions.append((match[1]['x'], match[1]['y']))
        self.areas.append((match[0]['area'], match[1]['area']))
        self.last_frame = frame_idx
        
    def get_last_position(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (self.upper_positions[-1], self.lower_positions[-1])
    
    def get_last_areas(self) -> Tuple[float, float]:
        return self.areas[-1]
    
    # def calculate_3d_position(self, upper_pos: Tuple[int, int], lower_pos: Tuple[int, int]) -> Tuple[float, float, float]:
    #     """Calculate 3D position from stereo matches"""
    #     disparity = upper_pos[0] - lower_pos[0]  # x-disparity
    #     if disparity == 0:
    #         return None
            
    #     # Calculate depth (Z) using disparity
    #     Z = (self.focal_length * self.baseline) / disparity
        
    #     # Calculate X and Y world coordinates
    #     X = Z * (upper_pos[0] - self.focal_length) * self.pixel_size
    #     Y = Z * upper_pos[1] * self.pixel_size
        
    #     return (X, Y, Z)

def predict_next_position(positions):
    """Predict next position based on velocity from last two positions"""
    if len(positions) < 2:
        return positions[-1]
    
    last_pos = positions[-1]
    prev_pos = positions[-2]
    velocity = (
        last_pos[0] - prev_pos[0],
        last_pos[1] - prev_pos[1]
    )
    return (
        last_pos[0] + velocity[0],
        last_pos[1] + velocity[1]
    )

def match_to_tracks(
    current_matches: List[Tuple[Dict, Dict]], 
    tracks: List[TrackedObject], 
    frame_idx: int,
    max_distance: int = 50,
    max_frames_gap: int = 10,
    continuity_weight: float = 0.2  # Weight for trajectory continuity
) -> List[TrackedObject]:
    # Remove old tracks
    active_tracks = [t for t in tracks if frame_idx - t.last_frame <= max_frames_gap]
    
    # Match current detections to existing tracks
    matched_indices = set()
    new_tracks = []
    
    for match in current_matches:
        upper_pos = (match[0]['x'], match[0]['y'])
        lower_pos = (match[1]['x'], match[1]['y'])
        areas = (match[0]['area'], match[1]['area'])
        
        best_track = None
        best_score = float('inf')
        
        for track in active_tracks:
            if track.last_frame == frame_idx:
                continue
                
            last_upper, last_lower = track.get_last_position()
            last_areas = track.get_last_areas()
            
            # Position distance
            dist_upper = math.sqrt((upper_pos[0] - last_upper[0])**2 + 
                                 (upper_pos[1] - last_upper[1])**2)
            dist_lower = math.sqrt((lower_pos[0] - last_lower[0])**2 + 
                                 (lower_pos[1] - last_lower[1])**2)
            
            # Area similarity
            area_similarity = min(areas[0], last_areas[0]) / max(areas[0], last_areas[0]) * \
                            min(areas[1], last_areas[1]) / max(areas[1], last_areas[1])
            
            # Trajectory continuity score
            continuity_score = 0
            if len(track.upper_positions) >= 2:
                # Predict positions based on current trajectory
                pred_upper = predict_next_position(track.upper_positions)
                pred_lower = predict_next_position(track.lower_positions)
                
                # Calculate deviation from predicted trajectory
                pred_dist_upper = math.sqrt((upper_pos[0] - pred_upper[0])**2 + 
                                         (upper_pos[1] - pred_upper[1])**2)
                pred_dist_lower = math.sqrt((lower_pos[0] - pred_lower[0])**2 + 
                                         (lower_pos[1] - pred_lower[1])**2)
                
                continuity_score = (pred_dist_upper + pred_dist_lower) * continuity_weight
            
            # Combined score (lower is better)
            total_score = (dist_upper + dist_lower) * (1 + (1 - area_similarity)) + continuity_score
            
            if total_score < best_score and total_score < max_distance:
                best_score = total_score
                best_track = track
        
        if best_track is not None:
            best_track.add_match(match, frame_idx)
            matched_indices.add(id(best_track))
        else:
            new_tracks.append(TrackedObject(match, frame_idx))#, calib_params=calib_params))
    
    # Keep unmatched active tracks and add new tracks
    final_tracks = [t for t in active_tracks if id(t) in matched_indices]
    final_tracks.extend(new_tracks)
    
    return final_tracks

def load_calibration_params(stereo_calib_file: str) -> Dict:
    """Load and process calibration parameters"""
    calib_data = np.load(stereo_calib_file)
    
    # Get camera matrices
    mtx_upper = calib_data['cameraMatrixUpper']
    mtx_lower = calib_data['cameraMatrixLower']
    
    # Focal length is the average of fx and fy from the camera matrix
    # Camera matrix structure:
    # [[fx  0  cx]
    #  [0   fy cy]
    #  [0   0   1]]
    focal_length_upper = (mtx_upper[0,0] + mtx_upper[1,1]) / 2
    focal_length_lower = (mtx_lower[0,0] + mtx_lower[1,1]) / 2
    
    # Use average of both cameras
    focal_length_pixels = (focal_length_upper + focal_length_lower) / 2
    
    return {
        'focal_length': focal_length_pixels,  # in pixels
        'baseline': 38.0,  # in mm
        'pixel_size': 0.00155,  # in mm (1.55 micrometers)
        'camera_matrices': (mtx_upper, mtx_lower),
        'R': calib_data['R'],  # Rotation matrix between cameras
        'T': calib_data['T']   # Translation vector between cameras
    }

def rectify_align_and_correct_images(
    input_dir,
    output_dir,
    stereo_calib_file,
    rectification_file,
    threshold=30,
    roi_file=None,
    window_size=10,
    min_area=2
):
    # Load calibration parameters
    calib_params = load_calibration_params(stereo_calib_file)
    rect_data = np.load(rectification_file)
    
    #F = calib_data['F']  # assuming 'F' is the key in the calibration file
    
    map1_x, map1_y = rect_data['map1_x'], rect_data['map1_y']
    map2_x, map2_y = rect_data['map2_x'], rect_data['map2_y']
    
    os.makedirs(output_dir, exist_ok=True)
    
    upper_images = sorted(glob.glob(os.path.join(input_dir, 'left', '*.jpg')))
    lower_images = sorted(glob.glob(os.path.join(input_dir, 'right', '*.jpg')))
    
    # Rectify and store all images for background averaging
    upper_rect_stack, lower_rect_stack = [], []
    print("Rectifying images...")
    for u_file, l_file in tqdm(zip(upper_images, lower_images)):
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
    print("Done")
    
    # Compute average backgrounds
    if upper_rect_stack and lower_rect_stack:
        print("Computing average background images...")
        avg_bg_upper = compute_average_background(upper_rect_stack)
        avg_bg_lower = compute_average_background(lower_rect_stack)
        print("Done")
    else:
        return
    
    # Initialize tracks list
    tracks: List[TrackedObject] = []
    
    # Perform correction and output results
    print("Processing images...")
    for idx, (u_file, l_file) in tqdm(enumerate(zip(upper_images, lower_images))):
        u_rect_gray = upper_rect_stack[idx]
        l_rect_gray = lower_rect_stack[idx]
        
        # Background subtraction
        u_corrected = cv2.absdiff(u_rect_gray, avg_bg_upper)
        l_corrected = cv2.absdiff(l_rect_gray, avg_bg_lower)
        
        # Threshold
        _, u_out = cv2.threshold(u_corrected, threshold, 255, cv2.THRESH_BINARY)
        _, l_out = cv2.threshold(l_corrected, threshold, 255, cv2.THRESH_BINARY)
        
        # Convert to 3-channel for line drawing
        # u_out = cv2.cvtColor(u_thresh, cv2.COLOR_GRAY2BGR)
        # l_out = cv2.cvtColor(l_thresh, cv2.COLOR_GRAY2BGR)
        
        # Get valid ROI from rectification maps
        roi = get_valid_roi(map1_x, map1_y, map2_x, map2_y, roi_file)
        left_bound, upper_bound_top, lower_bound_top, upper_bound_bottom, lower_bound_bottom, right_bound = roi
        #print(f"ROI: left={left_bound}, right={right_bound}, upper_bound_top={upper_bound_top}, lower_bound_top={lower_bound_top}, upper_bound_bottom={upper_bound_bottom}, lower_bound_bottom={lower_bound_bottom}") 

        # Rotate 90° CCW and stack cropped images
        u_rot = cv2.rotate(u_out, cv2.ROTATE_90_COUNTERCLOCKWISE)
        l_rot = cv2.rotate(l_out, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #print(f"Upper shape: {u_rot.shape}, Lower shape: {l_rot.shape}")  

        # # Draw optical axes before cropping
        # optical_center_x = map1_y.shape[0] // 2
        # optical_center_y_upper = map1_x.shape[1] // 2
        # optical_center_y_lower = map2_x.shape[1] // 2

        # # Draw vertical lines (red)
        # cv2.line(u_rot, 
        #          (optical_center_x, 0),
        #          (optical_center_x, u_rot.shape[0]),
        #          (0, 0, 255),  # Red color
        #          1)
        # cv2.line(l_rot,
        #          (optical_center_x, 0),
        #          (optical_center_x, l_rot.shape[0]),
        #          (0, 0, 255),  # Red color
        #          1)

        # # Draw horizontal lines (blue)
        # cv2.line(u_rot,
        #          (0, optical_center_y_upper),
        #          (u_rot.shape[1], optical_center_y_upper),
        #          (255, 0, 0),  # Blue color
        #          1)
        # cv2.line(l_rot,
        #          (0, optical_center_y_lower),
        #          (l_rot.shape[1], optical_center_y_lower),
        #          (255, 0, 0),  # Blue color
        #          1)

        u_rot = u_rot[upper_bound_top+90:lower_bound_top, left_bound+100:right_bound-70]
        l_rot = l_rot[upper_bound_bottom:lower_bound_bottom-20, left_bound+100:right_bound-70]   

        # Replace contour finding with connected components analysis
        num_labels_u, labels_u, stats_u, centroids_u = cv2.connectedComponentsWithStats(u_rot)
        num_labels_l, labels_l, stats_l, centroids_l = cv2.connectedComponentsWithStats(l_rot)

        # Convert back to BGR for drawing colored matches
        u_color = cv2.cvtColor(u_rot, cv2.COLOR_GRAY2BGR)
        l_color = cv2.cvtColor(l_rot, cv2.COLOR_GRAY2BGR)

        # Stack the color images (before drawing anything)
        clean_stacked = np.concatenate([u_color, l_color], axis=0)
        split_line_y = u_color.shape[0]

        # Process connected components to get object locations
        upper_objects = []
        lower_objects = []
        
        # Skip label 0 as it's the background
        for label in range(1, num_labels_u):
            area = stats_u[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                cx, cy = map(int, centroids_u[label])
                upper_objects.append({'x': cx, 'y': cy, 'area': area})
                
        for label in range(1, num_labels_l):
            area = stats_l[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                cx, cy = map(int, centroids_l[label])
                lower_objects.append({'x': cx, 'y': cy, 'area': area})

        # Stack the color images (after drawing detections)
        stacked = clean_stacked.copy()
        
        # Draw green separator line
        cv2.line(stacked, 
                 (0, split_line_y), 
                 (stacked.shape[1], split_line_y), 
                 (0, 255, 0), 
                 2)

        # Match objects within column windows
        matches = []
        
        for u_obj in upper_objects:
            column_matches = []
            for l_obj in lower_objects:
                if abs(u_obj['x'] - l_obj['x']) <= window_size:
                    # Calculate area similarity
                    area_ratio = min(u_obj['area'], l_obj['area']) / max(u_obj['area'], l_obj['area'])
                    
                    # Calculate position score
                    base_score = 1.0 - area_ratio  # Lower is better
                    
                    # Add trajectory prediction if we have matching tracks
                    trajectory_score = 0
                    for track in tracks:
                        if track.last_frame == idx - 1:  # Track from previous frame
                            # Predict next positions
                            pred_upper = predict_next_position(track.upper_positions)
                            pred_lower = predict_next_position(track.lower_positions)
                            
                            # Calculate distance to predictions
                            upper_dist = math.sqrt((u_obj['x'] - pred_upper[0])**2 + 
                                                 (u_obj['y'] - pred_upper[1])**2)
                            lower_dist = math.sqrt((l_obj['x'] - pred_lower[0])**2 + 
                                                 (l_obj['y'] - pred_lower[1])**2)
                            
                            # Update trajectory score if this is better than previous
                            new_score = (upper_dist + lower_dist) * 0.2  # 0.3 is trajectory weight
                            if trajectory_score == 0 or new_score < trajectory_score:
                                trajectory_score = new_score
                    
                    total_score = base_score + trajectory_score
                    column_matches.append((l_obj, total_score))
            
            # Select best match in column based on combined score (lower is better)
            if column_matches:
                best_match = min(column_matches, key=lambda x: x[1])[0]
                matches.append((u_obj, best_match))
                # Draw yellow line connecting matches
                # cv2.line(stacked,
                #         (u_obj['x'], u_obj['y']),
                #         (best_match['x'], best_match['y'] + split_line_y),
                #         (0, 255, 255),  # Yellow color
                #         1)

        # Draw detection circles
        for u_obj in upper_objects:
            cv2.circle(stacked, (u_obj['x'], u_obj['y']), 3, (0, 0, 255), -1)
        for l_obj in lower_objects:
            cv2.circle(stacked, (l_obj['x'], l_obj['y'] + split_line_y), 3, (0, 0, 255), -1)

        # Add tracking across frames
        tracks = match_to_tracks(matches, tracks, idx)
        
        # Visualize tracks
        for track in tracks:
            if len(track.upper_positions) > 1 and track.last_frame == idx:
                # # Calculate 3D velocity
                # pos3d_current = track.calculate_3d_position(
                #     track.upper_positions[-1], 
                #     track.lower_positions[-1]
                # )
                # pos3d_prev = track.calculate_3d_position(
                #     track.upper_positions[-2],
                #     track.lower_positions[-2]
                # )
                
                # if pos3d_current and pos3d_prev:
                #     dt = 1/30.0  # Assuming 30 fps, adjust as needed
                #     velocity = [
                #         (pos3d_current[i] - pos3d_prev[i])/dt 
                #         for i in range(3)
                #     ]
                    
                #     # Display velocity
                #     last_pos = track.upper_positions[-1]
                #     cv2.putText(stacked,
                #         f"v={np.linalg.norm(velocity):.1f}mm/s",
                #         (int(last_pos[0]), int(last_pos[1]) - 10),
                #         cv2.FONT_HERSHEY_SIMPLEX,
                #         0.5,
                #         (255, 255, 255),
                #         1)
                
                # Draw track history
                color = tuple(np.random.randint(0, 255, 3).tolist())  # Random color for each track
                
                # Draw upper track
                for i in range(len(track.upper_positions) - 1):
                    pt1 = (track.upper_positions[i][0], track.upper_positions[i][1])
                    pt2 = (track.upper_positions[i+1][0], track.upper_positions[i+1][1])
                    cv2.line(stacked, pt1, pt2, color, 2)
                
                # Draw lower track
                for i in range(len(track.lower_positions) - 1):
                    pt1 = (track.lower_positions[i][0], track.lower_positions[i][1] + split_line_y)
                    pt2 = (track.lower_positions[i+1][0], track.lower_positions[i+1][1] + split_line_y)
                    cv2.line(stacked, pt1, pt2, color, 2)
        
        # Write both outputs
        base_filename = os.path.splitext(os.path.basename(u_file))[0]
        # Save visualization with tracking
        vis_file = os.path.join(output_dir, base_filename + "_tracked.jpg")
        cv2.imwrite(vis_file, stacked)
        # Save clean stacked image
        # clean_file = os.path.join(output_dir, base_filename + "_clean.jpg")
        # cv2.imwrite(clean_file, clean_stacked)

if __name__ == "__main__":
    # Example usage
    rectify_align_and_correct_images(
        input_dir="/Users/vdausmann/oyster_project/images/20250312_21.2_01",
        output_dir="/Users/vdausmann/oyster_project/result_images/20250312_21.2_01_old",
        stereo_calib_file="/Users/vdausmann/oyster_project/planktracker3D/calibration/stereo_calibration.npz",
        rectification_file="/Users/vdausmann/oyster_project/planktracker3D/calibration/stereo_rectification.npz",
        threshold=10,
        roi_file="/Users/vdausmann/oyster_project/planktracker3D/calibration/roi_coordinates.npz",
        window_size=15,
        min_area=1
    )