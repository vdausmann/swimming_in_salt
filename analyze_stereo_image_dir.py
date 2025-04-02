import os
import cv2
import glob
import numpy as np
import math
from typing import List, Dict, Tuple
from tqdm import tqdm
from dataclasses import dataclass
from scipy.spatial.distance import cdist
import pandas as pd

@dataclass
class SingleTrack:
    """Track object in a single image sequence"""
    positions: List[Tuple[int, int]]
    areas: List[float]
    frame_indices: List[int]
    
    @property
    def is_active(self) -> bool:
        return len(self.positions) > 0
    
    @property
    def last_position(self) -> Tuple[int, int]:
        return self.positions[-1]
    
    @property
    def last_frame(self) -> int:
        return self.frame_indices[-1]
        
    @property
    def velocity(self) -> Tuple[float, float]:
        """Calculate current velocity vector"""
        if len(self.positions) < 2:
            return (0, 0)
        return (
            self.positions[-1][0] - self.positions[-2][0],
            self.positions[-1][1] - self.positions[-2][1]
        )
    
    def get_trajectory_vector(self) -> np.ndarray:
        """Calculate normalized trajectory vector from last N positions"""
        if len(self.positions) < 2:
            return np.array([0, 0])
        
        positions = np.array(self.positions[-5:])  # Use last 5 positions
        diffs = np.diff(positions, axis=0)
        mean_vector = np.mean(diffs, axis=0)
        norm = np.linalg.norm(mean_vector)
        return mean_vector / norm if norm > 0 else mean_vector
    
    def predict_next_position(self) -> Tuple[int, int]:
        if len(self.positions) < 2:
            return self.last_position
        velocity = self.velocity
        return (
            int(self.last_position[0] + velocity[0]),
            int(self.last_position[1] + velocity[1])
        )

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

def track_objects_in_image(
    detections: List[Dict],
    existing_tracks: List[SingleTrack],
    frame_idx: int,
    max_distance: float = 20,
    max_frames_gap: int = 5
) -> List[SingleTrack]:
    """Track objects in a single image sequence"""
    
    # Remove old tracks
    active_tracks = [t for t in existing_tracks 
                    if frame_idx - t.last_frame <= max_frames_gap]
    
    # Convert current detections to numpy array for efficient distance calculation
    if not detections:
        return active_tracks
        
    current_positions = np.array([[d['x'], d['y']] for d in detections])
    current_areas = np.array([d['area'] for d in detections])
    
    # Calculate distances between all current detections and predicted positions
    matched_track_indices = set()
    matched_detection_indices = set()
    
    if active_tracks:
        predicted_positions = np.array([t.predict_next_position() for t in active_tracks])
        distances = cdist(current_positions, predicted_positions)
        
        # Match detections to tracks
        while True:
            if len(matched_detection_indices) == len(detections) or \
               len(matched_track_indices) == len(active_tracks):
                break
                
            # Find minimum distance
            min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)
            min_dist = distances[min_dist_idx]
            
            if min_dist > max_distance:
                break
                
            det_idx, track_idx = min_dist_idx
            
            if det_idx not in matched_detection_indices and \
               track_idx not in matched_track_indices:
                # Update track
                track = active_tracks[track_idx]
                detection = detections[det_idx]
                
                track.positions.append((detection['x'], detection['y']))
                track.areas.append(detection['area'])
                track.frame_indices.append(frame_idx)
                
                matched_detection_indices.add(det_idx)
                matched_track_indices.add(track_idx)
            
            # Mark this pair as invalid for future iterations
            distances[det_idx, track_idx] = float('inf')
    
    # Create new tracks for unmatched detections
    for i, detection in enumerate(detections):
        if i not in matched_detection_indices:
            new_track = SingleTrack(
                positions=[(detection['x'], detection['y'])],
                areas=[detection['area']],
                frame_indices=[frame_idx]
            )
            active_tracks.append(new_track)
    
    return active_tracks

def match_tracks_between_images(
    upper_tracks: List[SingleTrack],
    lower_tracks: List[SingleTrack],
    max_x_distance: float = 15,
    min_trajectory_similarity: float = 0.2
) -> List[Tuple[SingleTrack, SingleTrack]]:
    """Match tracks between upper and lower images based on:
    1. X-position window constraint (primary)
    2. Track shape similarity (secondary)
    """
    
    matches = []
    matched_lower_indices = set()

    def calculate_trajectory_similarity(track1: SingleTrack, track2: SingleTrack) -> float:
        """Calculate similarity between two track histories"""
        # Get last 5 positions or all if less
        positions1 = np.array(track1.positions[-5:])
        positions2 = np.array(track2.positions[-5:])
        
        # Normalize positions to start at (0,0)
        positions1 = positions1 - positions1[0]
        positions2 = positions2 - positions2[0]
        
        # Calculate shape similarity using trajectory vectors
        if len(positions1) >= 2 and len(positions2) >= 2:
            # Calculate direction changes
            diffs1 = np.diff(positions1, axis=0)
            diffs2 = np.diff(positions2, axis=0)
            
            # Normalize vectors
            norms1 = np.linalg.norm(diffs1, axis=1, keepdims=True)
            norms2 = np.linalg.norm(diffs2, axis=1, keepdims=True)
            
            # Avoid division by zero
            norms1[norms1 == 0] = 1
            norms2[norms2 == 0] = 1
            
            vectors1 = diffs1 / norms1
            vectors2 = diffs2 / norms2
            
            # Compare direction changes
            min_len = min(len(vectors1), len(vectors2))
            similarities = np.sum(vectors1[:min_len] * vectors2[:min_len], axis=1)
            return np.mean(similarities)
            
        return 0.0
    
    for upper_track in upper_tracks:
        if not upper_track.is_active:
            continue
            
        best_match = None
        best_score = float('inf')
        
        upper_x = upper_track.last_position[0]
        
        # First, find candidates within x-window
        candidates = []
        for i, lower_track in enumerate(lower_tracks):
            if i in matched_lower_indices or not lower_track.is_active:
                continue
                
            lower_x = lower_track.last_position[0]
            x_dist = abs(upper_x - lower_x)
            
            # Primary criterion: must be within x-window
            if x_dist <= max_x_distance:
                candidates.append((i, lower_track, x_dist))
        
        # Then evaluate track shape similarity for candidates
        for i, lower_track, x_dist in candidates:
            # Calculate trajectory similarity
            traj_similarity = calculate_trajectory_similarity(upper_track, lower_track)
            
            if traj_similarity >= min_trajectory_similarity:
                # Combined score (lower is better)
                # Weight x_distance more heavily than trajectory similarity
                score = (x_dist * 2.0) * (2.0 - traj_similarity)
                
                if score < best_score:
                    best_score = score
                    best_match = i
        
        if best_match is not None:
            matches.append((upper_track, lower_tracks[best_match]))
            matched_lower_indices.add(best_match)
    
    return matches

def save_tracks_to_csv(
    output_dir: str,
    upper_tracks: List[SingleTrack],
    lower_tracks: List[SingleTrack],
    matched_pairs: List[Tuple[SingleTrack, SingleTrack]]
):
    """Save tracking results to CSV files"""
    # Create DataFrames for individual tracks
    upper_data = []
    lower_data = []
    matched_data = []
    
    # Process upper tracks
    for track_id, track in enumerate(upper_tracks):
        for pos, area, frame in zip(track.positions, track.areas, track.frame_indices):
            upper_data.append({
                'track_id': f'upper_{track_id}',
                'frame': frame,
                'x': pos[0],
                'y': pos[1],
                'area': area,
                'is_matched': any(track == pair[0] for pair in matched_pairs)
            })
    
    # Process lower tracks
    for track_id, track in enumerate(lower_tracks):
        for pos, area, frame in zip(track.positions, track.areas, track.frame_indices):
            lower_data.append({
                'track_id': f'lower_{track_id}',
                'frame': frame,
                'x': pos[0],
                'y': pos[1],
                'area': area,
                'is_matched': any(track == pair[1] for pair in matched_pairs)
            })
    
    # Process matched pairs
    for pair_id, (upper_track, lower_track) in enumerate(matched_pairs):
        # Find all frames where both tracks exist
        common_frames = set(upper_track.frame_indices) & set(lower_track.frame_indices)
        
        for frame in common_frames:
            u_idx = upper_track.frame_indices.index(frame)
            l_idx = lower_track.frame_indices.index(frame)
            
            u_pos = upper_track.positions[u_idx]
            l_pos = lower_track.positions[l_idx]
            
            matched_data.append({
                'pair_id': f'pair_{pair_id}',
                'frame': frame,
                'upper_x': u_pos[0],
                'upper_y': u_pos[1],
                'upper_area': upper_track.areas[u_idx],
                'lower_x': l_pos[0],
                'lower_y': l_pos[1],
                'lower_area': lower_track.areas[l_idx],
                'x_distance': abs(u_pos[0] - l_pos[0])
            })
    
    # Create DataFrames and save to CSV
    if upper_data:
        df_upper = pd.DataFrame(upper_data)
        df_upper.to_csv(os.path.join(output_dir, 'upper_tracks.csv'), index=False)
    
    if lower_data:
        df_lower = pd.DataFrame(lower_data)
        df_lower.to_csv(os.path.join(output_dir, 'lower_tracks.csv'), index=False)
    
    if matched_data:
        df_matched = pd.DataFrame(matched_data)
        df_matched.to_csv(os.path.join(output_dir, 'matched_tracks.csv'), index=False)

def main_processing_loop(
    input_dir: str,
    output_dir: str,
    stereo_calib_file: str,
    rectification_file: str,
    threshold: int = 30,
    roi_file: str = None,
    window_size: int = 10,
    trajectory_similarity: float = 0.2,
    min_area: int = 2
):
    """Main processing loop for tracking objects in stereo images"""
    
    # Load calibration parameters

    rect_data = np.load(rectification_file)
        
    map1_x, map1_y = rect_data['map1_x'], rect_data['map1_y']
    map2_x, map2_y = rect_data['map2_x'], rect_data['map2_y']
    
    os.makedirs(output_dir, exist_ok=True)
    
    upper_images = sorted(glob.glob(os.path.join(input_dir, 'left', '*.jpg')))
    lower_images = sorted(glob.glob(os.path.join(input_dir, 'right', '*.jpg')))

    if not upper_images or not lower_images:
        print("No images found!")
        return
    
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
    def compute_average_background(image_stack):
        image_stack = np.array(image_stack, dtype=np.float32)
        return np.mean(image_stack, axis=0).astype(np.uint8)

    if upper_rect_stack and lower_rect_stack:
        print("Computing average background images...")
        avg_bg_upper = compute_average_background(upper_rect_stack)
        avg_bg_lower = compute_average_background(lower_rect_stack)
        print("Done")
    else:
        return
    
    # Initialize tracking
    upper_tracks: List[SingleTrack] = []
    lower_tracks: List[SingleTrack] = []
    track_colors = {}  # For visualization
    
    # Process each frame
    print("Processing images...")
    for idx, (u_file, l_file) in tqdm(enumerate(zip(upper_images, lower_images))):
        
        u_rect_gray = upper_rect_stack[idx]
        l_rect_gray = lower_rect_stack[idx]
        
        # Background subtraction
        u_corrected = cv2.absdiff(u_rect_gray, avg_bg_upper)
        l_corrected = cv2.absdiff(l_rect_gray, avg_bg_lower)
        
        # Threshold
        _, u_thresh = cv2.threshold(u_corrected, threshold, 255, cv2.THRESH_BINARY)
        _, l_thresh = cv2.threshold(l_corrected, threshold, 255, cv2.THRESH_BINARY)

        # Rotate 90° CCW and stack cropped images
        u_thresh = cv2.rotate(u_thresh, cv2.ROTATE_90_COUNTERCLOCKWISE)
        l_thresh = cv2.rotate(l_thresh, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Get valid ROI from rectification maps
        roi = get_valid_roi(map1_x, map1_y, map2_x, map2_y, roi_file)
        left_bound, upper_bound_top, lower_bound_top, upper_bound_bottom, lower_bound_bottom, right_bound = roi

        u_crop = u_thresh[upper_bound_top+90:lower_bound_top, left_bound+100:right_bound-70]
        l_crop = l_thresh[upper_bound_bottom:lower_bound_bottom-20, left_bound+100:right_bound-70] 

        # Stack the color images (before drawing anything)
        #clean_stacked = np.concatenate([u_crop, l_crop], axis=0)
        split_line_y = u_crop.shape[0]
        
        # Find connected components
        num_labels_u, labels_u, stats_u, centroids_u = cv2.connectedComponentsWithStats(u_crop)
        num_labels_l, labels_l, stats_l, centroids_l = cv2.connectedComponentsWithStats(l_crop)
        
        # Convert detections to list of dictionaries
        upper_objects = []
        lower_objects = []
        
        for label in range(1, num_labels_u):  # Skip background (label 0)
            area = stats_u[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                cx, cy = map(int, centroids_u[label])
                upper_objects.append({'x': cx, 'y': cy, 'area': area})
                
        for label in range(1, num_labels_l):
            area = stats_l[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                cx, cy = map(int, centroids_l[label])
                lower_objects.append({'x': cx, 'y': cy, 'area': area})
        
        # Track objects in each image
        upper_tracks = track_objects_in_image(upper_objects, upper_tracks, idx)
        lower_tracks = track_objects_in_image(lower_objects, lower_tracks, idx)
        
        # Match tracks between images
        matched_tracks = match_tracks_between_images(upper_tracks, lower_tracks, window_size, trajectory_similarity)
        
        # Visualize results
        u_vis = cv2.cvtColor(u_crop, cv2.COLOR_GRAY2BGR)
        l_vis = cv2.cvtColor(l_crop, cv2.COLOR_GRAY2BGR)
        
        # Draw tracks
        # First, assign colors to matched tracks
        for upper_track, lower_track in matched_tracks:
            if id(upper_track) not in track_colors:
                color = tuple(np.random.randint(0, 255, 3).tolist())
                track_colors[id(upper_track)] = color
                track_colors[id(lower_track)] = color  # Same color for matched pair
        
        # Then assign colors to unmatched tracks
        for track in upper_tracks + lower_tracks:
            if id(track) not in track_colors:
                track_colors[id(track)] = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Draw all tracks
        for track in upper_tracks + lower_tracks:
            color = track_colors[id(track)]
            
            # Draw track history
            for i in range(len(track.positions) - 1):
                pt1 = tuple(map(int, track.positions[i]))
                pt2 = tuple(map(int, track.positions[i + 1]))
                if track in upper_tracks:
                    cv2.line(u_vis, pt1, pt2, color, 2)
                else:
                    cv2.line(l_vis, pt1, pt2, color, 2)

        # Stack images
        stacked = np.concatenate([u_vis, l_vis], axis=0)

        # Draw green separator line
        cv2.line(stacked, 
                 (0, split_line_y), 
                 (stacked.shape[1], split_line_y), 
                 (0, 255, 0), 
                 2)
        
        # Draw detection circles
        for u_obj in upper_objects:
            cv2.circle(stacked, (u_obj['x'], u_obj['y']), 3, (0, 0, 255), -1)
        for l_obj in lower_objects:
            cv2.circle(stacked, (l_obj['x'], l_obj['y'] + split_line_y), 3, (0, 0, 255), -1)
        
        # Draw matches
        # split_line_y = u_vis.shape[0]
        # for upper_track, lower_track in matched_tracks:
        #     u_pos = upper_track.last_position
        #     l_pos = lower_track.last_position
        #     cv2.line(stacked,
        #             u_pos,
        #             (l_pos[0], l_pos[1] + split_line_y),
        #             (0, 255, 255),  # Yellow
        #             1)
        
        # Save output
        output_path = os.path.join(output_dir, f"tracked_{idx:04d}.jpg")
        cv2.imwrite(output_path, stacked)
    
    # Save tracking results to CSV
    save_tracks_to_csv(output_dir, upper_tracks, lower_tracks, matched_tracks)

if __name__ == "__main__":
    main_processing_loop(
        input_dir="/Users/vdausmann/oyster_project/images/20250328_24.4_14_01",
        output_dir="/Users/vdausmann/oyster_project/result_images/20250328_24.4_14_01",
        stereo_calib_file="/Users/vdausmann/oyster_project/planktracker3D/calibration/stereo_calibration.npz",
        rectification_file="/Users/vdausmann/oyster_project/planktracker3D/calibration/stereo_rectification.npz",
        threshold=10,
        roi_file="/Users/vdausmann/oyster_project/planktracker3D/calibration/roi_coordinates.npz",
        window_size=15, #window in x direction for matching objects in stereo image pairs
        trajectory_similarity=0.2, #minimum trajectory similarity for matching objects in stereo image pairs
        min_area=1
    )