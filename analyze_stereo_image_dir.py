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
from scipy.optimize import curve_fit

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
    
    # def predict_next_position(self) -> Tuple[int, int]:
    #     if len(self.positions) < 2:
    #         return self.last_position
    #     velocity = self.velocity
    #     return (
    #         int(self.last_position[0] + velocity[0]),
    #         int(self.last_position[1] + velocity[1])
    #     )

    def predict_next_position(self) -> Tuple[int, int]:
        """Predict next position using linear or sine fit, fallback to velocity."""
        from scipy.optimize import curve_fit
        import numpy as np

        def linear_func(x, a, b):
            return a * x + b

        def sine_func(x, A, w, phi, c):
            return A * np.sin(w * x + phi) + c

        if len(self.positions) < 4:
            # Not enough points, fallback to velocity
            if len(self.positions) < 2:
                return self.last_position
            velocity = self.velocity
            return (
                int(self.last_position[0] + velocity[0]),
                int(self.last_position[1] + velocity[1])
            )
        positions = np.array(self.positions)
        x = np.arange(len(positions))
        pred = []
        for dim in range(2):  # x and y
            y = positions[:, dim]
            # Try linear fit
            try:
                popt_lin, _ = curve_fit(linear_func, x, y)
                y_pred_lin = linear_func(x, *popt_lin)
                lin_err = np.mean((y - y_pred_lin) ** 2)
            except Exception:
                lin_err = np.inf
            # Try sine fit
            try:
                guess_freq = 2 * np.pi / max(1, len(x))
                guess_amp = (np.max(y) - np.min(y)) / 2
                guess_offset = np.mean(y)
                guess = [guess_amp, guess_freq, 0, guess_offset]
                popt_sin, _ = curve_fit(sine_func, x, y, p0=guess, maxfev=10000)
                y_pred_sin = sine_func(x, *popt_sin)
                sin_err = np.mean((y - y_pred_sin) ** 2)
            except Exception:
                sin_err = np.inf
            # Choose best model
            next_x = x[-1] + 1
            if lin_err < sin_err:
                pred.append(linear_func(next_x, *popt_lin))
            else:
                pred.append(sine_func(next_x, *popt_sin))
        return tuple(int(round(v)) for v in pred)    
    

# Compare the full trajectory shape using direction vectors
def trajectory_shape_similarity(track: SingleTrack, detection: Dict, N: int = 10) -> float:
    """
    Compare the last N direction vectors of the track with the direction to the detection.
    Returns the mean cosine similarity (range -1 to 1).
    """
    positions = np.array(track.positions[-N:])
    if len(positions) < 2:
        return 1.0  # No trajectory, treat as perfect match

    # Compute direction vectors for the track
    diffs = np.diff(positions, axis=0)
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    track_dirs = diffs / norms

    # Add the direction to the detection as the last vector
    last_pos = positions[-1]
    det_vec = np.array([detection['x'], detection['y']]) - last_pos
    det_norm = np.linalg.norm(det_vec)
    if det_norm == 0:
        det_dir = track_dirs[-1]
    else:
        det_dir = det_vec / det_norm

    # Compare the last direction in the track to the direction to the detection
    similarities = np.dot(track_dirs[-1], det_dir)
    # Optionally, you can average over several last directions:
    # similarities = np.dot(track_dirs, det_dir)
    # return np.mean(similarities)

    return similarities  # or abs(similarities) if you want to allow reversals

def load_roi_coordinates(roi_file):
    """Load the manually defined ROI coordinates"""
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

# def track_objects_in_image(
#     detections: List[Dict],
#     existing_tracks: List[SingleTrack],
#     frame_idx: int,
#     max_distance: float = 20,
#     max_frames_gap: int = 5
# ) -> List[SingleTrack]:
#     """Track objects in a single image sequence"""
    
#     # Remove old tracks
#     active_tracks = [t for t in existing_tracks 
#                     if frame_idx - t.last_frame <= max_frames_gap]
    
#     # Convert current detections to numpy array for efficient distance calculation
#     if not detections:
#         return active_tracks
        
#     current_positions = np.array([[d['x'], d['y']] for d in detections])
#     current_areas = np.array([d['area'] for d in detections])
    
#     # Calculate distances between all current detections and predicted positions
#     matched_track_indices = set()
#     matched_detection_indices = set()
    
#     if active_tracks:
#         predicted_positions = np.array([t.predict_next_position() for t in active_tracks])
#         distances = cdist(current_positions, predicted_positions)
        
#         # Match detections to tracks
#         while True:
#             if len(matched_detection_indices) == len(detections) or \
#                len(matched_track_indices) == len(active_tracks):
#                 break
                
#             # Find minimum distance
#             min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)
#             min_dist = distances[min_dist_idx]
            
#             if min_dist > max_distance:
#                 break
                
#             det_idx, track_idx = min_dist_idx
            
#             if det_idx not in matched_detection_indices and \
#                track_idx not in matched_track_indices:
#                 # Update track
#                 track = active_tracks[track_idx]
#                 detection = detections[det_idx]
                
#                 track.positions.append((detection['x'], detection['y']))
#                 track.areas.append(detection['area'])
#                 track.frame_indices.append(frame_idx)
                
#                 matched_detection_indices.add(det_idx)
#                 matched_track_indices.add(track_idx)
            
#             # Mark this pair as invalid for future iterations
#             distances[det_idx, track_idx] = float('inf')
    
#     # Create new tracks for unmatched detections
#     for i, detection in enumerate(detections):
#         if i not in matched_detection_indices:
#             new_track = SingleTrack(
#                 positions=[(detection['x'], detection['y'])],
#                 areas=[detection['area']],
#                 frame_indices=[frame_idx]
#             )
#             active_tracks.append(new_track)
    
#     return active_tracks

from scipy.optimize import linear_sum_assignment

# def track_objects_in_image(
#     detections: List[Dict],
#     existing_tracks: List[SingleTrack],
#     frame_idx: int,
#     max_distance: float = 20,
#     max_frames_gap: int = 5,
#     min_traj_similarity: float = 0.2
# ) -> List[SingleTrack]:
#     """Hybrid tracking: use distance unless ambiguous, then use trajectory shape."""
#     active_tracks = [t for t in existing_tracks if frame_idx - t.last_frame <= max_frames_gap]
#     if not detections:
#         return active_tracks

#     detection_positions = np.array([[d['x'], d['y']] for d in detections])
#     matched_detection_indices = set()
#     matched_track_indices = set()

#     # For each track, find candidate detections
#     for track_idx, track in enumerate(active_tracks):
#         pred_pos = np.array(track.predict_next_position())
#         dists = np.linalg.norm(detection_positions - pred_pos, axis=1)
#         candidates = np.where(dists < max_distance)[0]

#         if len(candidates) == 1:
#             # Only one candidate: assign directly
#             det_idx = candidates[0]
#             if det_idx not in matched_detection_indices:
#                 detection = detections[det_idx]
#                 track.positions.append((detection['x'], detection['y']))
#                 track.areas.append(detection['area'])
#                 track.frame_indices.append(frame_idx)
#                 matched_detection_indices.add(det_idx)
#                 matched_track_indices.add(track_idx)
#         elif len(candidates) > 1:
#             # Ambiguous: use distance to predicted next position (from advanced model)
#             pred_pos = np.array(track.predict_next_position())
#             best_dist = float('inf')
#             best_det_idx = None
#             for det_idx in candidates:
#                 if det_idx in matched_detection_indices:
#                     continue
#                 detection = detections[det_idx]
#                 det_pos = np.array([detection['x'], detection['y']])
#                 dist = np.linalg.norm(det_pos - pred_pos)
#                 # Optionally, combine with trajectory similarity:
#                 # traj_sim = trajectory_shape_similarity(track, detection, N=10)
#                 # score = dist - alpha * traj_sim
#                 if dist < best_dist:
#                     best_dist = dist
#                     best_det_idx = det_idx
#             if best_det_idx is not None:
#                 detection = detections[best_det_idx]
#                 track.positions.append((detection['x'], detection['y']))
#                 track.areas.append(detection['area'])
#                 track.frame_indices.append(frame_idx)
#                 matched_detection_indices.add(best_det_idx)
#                 matched_track_indices.add(track_idx)
#         # else: no candidates, do nothing (track may die if not updated)

#     # Create new tracks for unmatched detections
#     for i, detection in enumerate(detections):
#         if i not in matched_detection_indices:
#             new_track = SingleTrack(
#                 positions=[(detection['x'], detection['y'])],
#                 areas=[detection['area']],
#                 frame_indices=[frame_idx]
#             )
#             active_tracks.append(new_track)
#     return active_tracks

def track_objects_in_image(
    detections: List[Dict],
    existing_tracks: List[SingleTrack],
    frame_idx: int,
    short_track_ids: dict,
    focus_id: int,
    vicinity_radius: float = 50,
    max_distance: float = 20,
    max_frames_gap: int = 5,
    min_traj_similarity: float = 0.2
) -> List[SingleTrack]:
    """
    Track only the focus track and tracks in its vicinity.
    Print likelihoods (distance to predicted position) for each candidate.
    """
    # Find the focus track's object id
    focus_tid = None
    for tid, short_id in short_track_ids.items():
        if short_id == focus_id:
            focus_tid = tid
            break

    # Find the focus track object
    focus_track = None
    for track in existing_tracks:
        if id(track) == focus_tid:
            focus_track = track
            break

    # If no focus track, fallback to normal tracking for all tracks
    if focus_track is None:
        print(f"Focus track with id {focus_id} not found, using all existing tracks.")
        return existing_tracks

    focus_pos = np.array(focus_track.last_position)

    # Only keep tracks in the vicinity of the focus track (including itself)
    filtered_tracks = []
    for track in existing_tracks:
        pos = np.array(track.last_position)
        if np.linalg.norm(pos - focus_pos) < vicinity_radius or id(track) == focus_tid:
            filtered_tracks.append(track)

    # Only keep detections in the vicinity of the focus track
    filtered_detections = []
    for det in detections:
        pos = np.array([det['x'], det['y']])
        if np.linalg.norm(pos - focus_pos) < vicinity_radius:
            filtered_detections.append(det)
            print(f"Detection at {det['x']}, {det['y']} in vicinity of focus track {focus_id}")
    # Always include the detection closest to the focus track (in case none are within radius)
    if not filtered_detections and detections:
        dists = [np.linalg.norm(np.array([det['x'], det['y']]) - focus_pos) for det in detections]
        min_idx = int(np.argmin(dists))
        filtered_detections.append(detections[min_idx])
        print(f"No detections in vicinity of focus track {focus_id}, using closest detection at {filtered_detections[0]['x']}, {filtered_detections[0]['y']}")

    # Now do the usual matching, but only for filtered_tracks and filtered_detections
    active_tracks = [t for t in filtered_tracks if frame_idx - t.last_frame <= max_frames_gap]
    if not filtered_detections:
        print(f"No detections in vicinity of focus track {focus_id}, returning existing tracks.")
        return existing_tracks  # No updates

    detection_positions = np.array([[d['x'], d['y']] for d in filtered_detections])
    matched_detection_indices = set()
    matched_track_indices = set()

    for track_idx, track in enumerate(active_tracks):
        pred_pos = np.array(track.predict_next_position())
        dists = np.linalg.norm(detection_positions - pred_pos, axis=1)
        candidates = np.where(dists < max_distance)[0]

        # Print likelihoods for focus track
        if id(track) == focus_tid and len(candidates) > 0:
            print(f"\nFrame {frame_idx}: Focus track {focus_id} (tid={focus_tid}) candidates:")
            for det_idx in candidates:
                detection = filtered_detections[det_idx]
                det_pos = np.array([detection['x'], detection['y']])
                dist = np.linalg.norm(det_pos - pred_pos)
                print(f"  Candidate det_idx={det_idx} at {det_pos}, dist={dist:.2f}")

        if len(candidates) == 1:
            det_idx = candidates[0]
            if det_idx not in matched_detection_indices:
                detection = filtered_detections[det_idx]
                track.positions.append((detection['x'], detection['y']))
                track.areas.append(detection['area'])
                track.frame_indices.append(frame_idx)
                matched_detection_indices.add(det_idx)
                matched_track_indices.add(track_idx)
        elif len(candidates) > 1:
            pred_pos = np.array(track.predict_next_position())
            best_dist = float('inf')
            best_det_idx = None
            for det_idx in candidates:
                if det_idx in matched_detection_indices:
                    continue
                detection = filtered_detections[det_idx]
                det_pos = np.array([detection['x'], detection['y']])
                dist = np.linalg.norm(det_pos - pred_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_det_idx = det_idx
            if best_det_idx is not None:
                detection = filtered_detections[best_det_idx]
                track.positions.append((detection['x'], detection['y']))
                track.areas.append(detection['area'])
                track.frame_indices.append(frame_idx)
                matched_detection_indices.add(best_det_idx)
                matched_track_indices.add(track_idx)
        # else: no candidates, do nothing

    # Create new tracks for unmatched detections (in vicinity only)
    for i, detection in enumerate(filtered_detections):
        if i not in matched_detection_indices:
            new_track = SingleTrack(
                positions=[(detection['x'], detection['y'])],
                areas=[detection['area']],
                frame_indices=[frame_idx]
            )
            existing_tracks.append(new_track)

    return existing_tracks

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

    rect_data = np.load(rectification_file, allow_pickle=True)
        
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
    # Add before your main loop
    short_track_ids = {}
    next_short_id = 0

    for idx, (u_file, l_file) in tqdm(enumerate(zip(upper_images, lower_images))):
        print(f"Processing frame {idx + 1}/{len(upper_images)}")
        
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

        u_crop = u_thresh[upper_bound_top+90:lower_bound_top, left_bound+120:right_bound-70]
        l_crop = l_thresh[upper_bound_bottom:lower_bound_bottom-20, left_bound+120:right_bound-70] 

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
        #upper_tracks = track_objects_in_image(upper_objects, upper_tracks, idx)
        # lower_tracks = track_objects_in_image(lower_objects, lower_tracks, idx)
        
        # Match tracks between images
        # matched_tracks = match_tracks_between_images(upper_tracks, lower_tracks, window_size, trajectory_similarity)
        
        # Visualize results
        u_vis = cv2.cvtColor(u_crop, cv2.COLOR_GRAY2BGR)
        
        # Assign colors to all upper tracks (no need for matched pairs)
        for track in upper_tracks:
            tid = id(track)
            if tid not in track_colors:
                track_colors[tid] = tuple(np.random.randint(0, 255, 3).tolist())
            if tid not in short_track_ids:
                short_track_ids[tid] = next_short_id
                next_short_id += 1
        
        # Set the ID you want to focus on
        focus_id = 14
        focus_tid = None
        for tid, short_id in short_track_ids.items():
            if short_id == focus_id:
                focus_tid = tid
                break

        # Find the focused track
        focus_track = None
        for track in upper_tracks:
            if id(track) == focus_tid:
                focus_track = track
                break

        # Only proceed if the focus track exists
        if focus_track is not None:
            focus_pos = np.array(focus_track.last_position)
            vicinity_radius = 50  # pixels, adjust as needed

            # Find tracks in the vicinity (including the focus track itself)
            tracks_to_show = []
            for track in upper_tracks:
                pos = np.array(track.last_position)
                if np.linalg.norm(pos - focus_pos) < vicinity_radius or id(track) == focus_tid:
                    tracks_to_show.append(track)

            # Visualize only the upper image
            u_vis = cv2.cvtColor(u_crop, cv2.COLOR_GRAY2BGR)

            # --- Annotate likelihoods for focus track candidates ---
            # Get detections in the vicinity
            candidate_detections = []
            for obj in upper_objects:
                pos = np.array([obj['x'], obj['y']])
                if np.linalg.norm(pos - focus_pos) < vicinity_radius:
                    candidate_detections.append(obj)
            # Always include the closest detection if none in vicinity
            if not candidate_detections and upper_objects:
                dists = [np.linalg.norm(np.array([obj['x'], obj['y']]) - focus_pos) for obj in upper_objects]
                min_idx = int(np.argmin(dists))
                candidate_detections.append(upper_objects[min_idx])

            # Draw likelihoods (distance to predicted position) for each candidate
            pred_pos = np.array(focus_track.predict_next_position())
            for obj in candidate_detections:
                det_pos = np.array([obj['x'], obj['y']])
                dist = np.linalg.norm(det_pos - pred_pos)
                cv2.putText(
                    u_vis,
                    f"{dist:.1f}",
                    (int(det_pos[0]) + 10, int(det_pos[1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2,
                )
                # cv2.circle(u_vis, last_pos, 3, color, -1)

            # Save or show only the upper image
            output_path = os.path.join(output_dir, f"tracked_focus_{idx:04d}.jpg")
            cv2.imwrite(output_path, u_vis)

        # # Then assign colors to unmatched tracks
        # for track in upper_tracks + lower_tracks:
        #     if id(track) not in track_colors:
        #         track_colors[id(track)] = tuple(np.random.randint(0, 255, 3).tolist())
        
        # # Draw all tracks (filtered by visible_track_indices if you want)
        # for track_idx, track in enumerate(upper_tracks + lower_tracks):
        #     color = track_colors[id(track)]
        #     # Draw track history
        #     for i in range(len(track.positions) - 1):
        #         pt1 = tuple(map(int, track.positions[i]))
        #         pt2 = tuple(map(int, track.positions[i + 1]))
        #         if track in upper_tracks: 
        #             cv2.line(u_vis, pt1, pt2, color, 2)
        #         else:
        #             cv2.line(l_vis, pt1, pt2, color, 2)
        #     # Draw track ID at last position
        #     last_pos = tuple(map(int, track.last_position))
        #     label = f"ID:{short_track_ids[id(track)]}"
        #     if track in upper_tracks:
        #         cv2.putText(u_vis, label, last_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        #     else:
        #         cv2.putText(l_vis, label, last_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # # Stack images
        # stacked = np.concatenate([u_vis, l_vis], axis=0)

        # # Draw green separator line
        # cv2.line(stacked, 
        #          (0, split_line_y), 
        #          (stacked.shape[1], split_line_y), 
        #          (0, 255, 0), 
        #          2)
        
        # # Draw detection circles and persistent track IDs at the last position of each track
        # for track_idx, track in enumerate(upper_tracks):
        #     last_pos = tuple(map(int, track.last_position))
        #     cv2.circle(stacked, last_pos, 3, (0, 0, 255), -1)
        #     cv2.putText(stacked, f"T{track_idx}", (last_pos[0]+5, last_pos[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        # for track_idx, track in enumerate(lower_tracks):
        #     last_pos = tuple(map(int, track.last_position))
        #     y_shift = last_pos[1] + split_line_y
        #     cv2.circle(stacked, (last_pos[0], y_shift), 3, (0, 0, 255), -1)
        #     cv2.putText(stacked, f"T{track_idx}", (last_pos[0]+5, y_shift-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        # # Draw matches
        # # split_line_y = u_vis.shape[0]
        # # for upper_track, lower_track in matched_tracks:
        # #     u_pos = upper_track.last_position
        # #     l_pos = lower_track.last_position
        # #     cv2.line(stacked,
        # #             u_pos,
        # #             (l_pos[0], l_pos[1] + split_line_y),
        # #             (0, 255, 255),  # Yellow
        # #             1)
        
        # # Save output
        # output_path = os.path.join(output_dir, f"tracked_{idx:04d}.jpg")
        # cv2.imwrite(output_path, stacked)
    
    # Save tracking results to CSV
    # save_tracks_to_csv(output_dir, upper_tracks, lower_tracks, matched_tracks)

if __name__ == "__main__":
    main_processing_loop(
        input_dir="/Users/vdausmann/oyster_project/images/20250312_21.2_01",
        output_dir="/Users/vdausmann/swimming_in_salt/result_images/20250312_21.2_01",
        #stereo_calib_file="/Users/vdausmann/oyster_project/planktracker3D/calibration/stereo_calibration.npz",
        stereo_calib_file="/Users/vdausmann/swimming_in_salt/calibration/stereo_calibration.npz",
        rectification_file="/Users/vdausmann/swimming_in_salt/calibration/stereo_rectification.npz",
        threshold=10,
        roi_file="/Users/vdausmann/swimming_in_salt/calibration/roi_coordinates.npz",
        window_size=15, #window in x direction for matching objects in stereo image pairs
        trajectory_similarity=0.2, #minimum trajectory similarity for matching objects in stereo image pairs
        min_area=1
    )