import os
import glob
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class SingleTrack:
    positions: List[Tuple[int, int]]
    areas: List[float]
    frame_indices: List[int]
    timestamps: List[str]  # NEW: Store timestamp for each frame
    motion_pattern: str = "unknown"  # "linear", "sinusoidal", etc.

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
    def last_timestamp(self) -> str:
        return self.timestamps[-1] if self.timestamps else ""

    @property
    def velocity(self) -> Tuple[float, float]:
        if len(self.positions) < 2:
            return (0, 0)
        return (
            self.positions[-1][0] - self.positions[-2][0],
            self.positions[-1][1] - self.positions[-2][1]
        )

    def predict_next_position(self) -> Tuple[int, int]:
        # For sinusoidal motion, look at more history
        if len(self.positions) < 3:
            return self.last_position
            
        # Calculate acceleration (change in velocity)
        last_vel = (self.positions[-1][0] - self.positions[-2][0],
                self.positions[-1][1] - self.positions[-2][1])
        prev_vel = (self.positions[-2][0] - self.positions[-3][0],
                self.positions[-2][1] - self.positions[-3][1])
        accel = (last_vel[0] - prev_vel[0], last_vel[1] - prev_vel[1])
        
        # Apply both velocity and acceleration
        return (
            int(self.last_position[0] + last_vel[0] + 0.5 * accel[0]),
            int(self.last_position[1] + last_vel[1] + 0.5 * accel[1])
        )
    
    def classify_motion_pattern(self, window=10):
        """Analyze motion pattern of this track."""
        if len(self.positions) < window:
            return "unknown"
            
        # Calculate directional changes
        directions = []
        for i in range(2, len(self.positions)):
            v1 = (self.positions[i-1][0] - self.positions[i-2][0], 
                 self.positions[i-1][1] - self.positions[i-2][1])
            v2 = (self.positions[i][0] - self.positions[i-1][0],
                 self.positions[i][1] - self.positions[i-1][1])
            
            # Use dot product to detect direction change
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            if dot < 0:
                directions.append("change")
            else:
                directions.append("same")
                
        # Count direction changes
        changes = directions.count("change")
        
        # Classify based on direction changes
        if changes > window * 0.3:  # More than 30% direction changes
            return "sinusoidal"
        else:
            return "linear"

import re
from datetime import datetime

def extract_timestamp_from_csv_filename(filename: str) -> Optional[str]:
    """Extract timestamp from CSV filename like 'lower_20250404_114558_108161_objects.csv'"""
    try:
        # Pattern: camera_YYYYMMDD_HHMMSS_milliseconds_objects.csv
        pattern = r'(\w+)_(\d{8})_(\d{6})_(\d+)_objects\.csv'
        match = re.search(pattern, filename)
        
        if match:
            camera, date_str, time_str, milliseconds = match.groups()
            
            # Parse date and time
            date_part = datetime.strptime(date_str, '%Y%m%d').date()
            time_part = datetime.strptime(time_str, '%H%M%S').time()
            
            # Combine date and time
            dt = datetime.combine(date_part, time_part)
            
            # Add milliseconds
            dt = dt.replace(microsecond=int(milliseconds[:6].ljust(6, '0')))
            
            return dt.isoformat()
        
        # Fallback: try to extract any timestamp-like pattern
        timestamp_patterns = [
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?)',  # ISO format
            r'(\d{4}\d{2}\d{2}_\d{6})',  # YYYYMMDD_HHMMSS format
            r'(\d{10,13})',  # Unix timestamp
        ]
        
        for pattern in timestamp_patterns:
            match = re.search(pattern, filename)
            if match:
                timestamp_str = match.group(1)
                
                if '_' in timestamp_str and len(timestamp_str) == 15:  # YYYYMMDD_HHMMSS
                    date_part, time_part = timestamp_str.split('_')
                    dt = datetime.strptime(f"{date_part}_{time_part}", '%Y%m%d_%H%M%S')
                    return dt.isoformat()
                elif timestamp_str.isdigit():  # Unix timestamp
                    timestamp_num = int(timestamp_str)
                    if timestamp_num > 1e12:  # Milliseconds
                        dt = datetime.fromtimestamp(timestamp_num / 1000.0)
                    else:  # Seconds
                        dt = datetime.fromtimestamp(timestamp_num)
                    return dt.isoformat()
                else:
                    return timestamp_str
        
        print(f"No timestamp pattern found in filename: {filename}")
        return None
        
    except Exception as e:
        print(f"Error extracting timestamp from filename {filename}: {e}")
        return None

def load_csv_with_timestamps(csv_file: str) -> pd.DataFrame:
    """Load CSV and ensure it has timestamp information"""
    df = pd.read_csv(csv_file)
    
    # Check if timestamp column already exists
    if 'timestamp' not in df.columns:
        # Extract timestamp from filename
        filename = os.path.basename(csv_file)
        file_timestamp = extract_timestamp_from_csv_filename(filename)
        
        if file_timestamp:
            # Add timestamp column - same timestamp for all detections in this frame
            df['timestamp'] = file_timestamp
            print(f"Added timestamp {file_timestamp} to {len(df)} detections from {filename}")
        else:
            # Fallback: use frame number as relative timestamp
            frame_num = df['frame'].iloc[0] if 'frame' in df.columns and len(df) > 0 else 0
            fallback_timestamp = f"frame_{frame_num:06d}_time_{frame_num/30.0:.3f}s"
            df['timestamp'] = fallback_timestamp
            print(f"Using fallback timestamp for {filename}: {fallback_timestamp}")
    
    return df

def predict_sinusoidal(track: SingleTrack) -> Tuple[int, int]:
    """
    Predict next position for a track with sinusoidal motion pattern.
    Uses frequency analysis on recent positions to make a better prediction.
    """
    # Need at least 4 points for meaningful sinusoidal prediction
    if len(track.positions) < 4:
        return track.predict_next_position()
    
    # Look at the last several positions to detect oscillation pattern
    # Use more points for better frequency estimation
    window_size = min(10, len(track.positions))
    recent_positions = track.positions[-window_size:]
    
    # Extract x and y components separately
    x_vals = [pos[0] for pos in recent_positions]
    y_vals = [pos[1] for pos in recent_positions]
    
    # Calculate average direction (overall trend)
    avg_x_vel = (recent_positions[-1][0] - recent_positions[0][0]) / (window_size - 1)
    avg_y_vel = (recent_positions[-1][1] - recent_positions[0][1]) / (window_size - 1)
    
    # Detect oscillation in each direction
    x_changes = sum(1 for i in range(1, len(x_vals)-1) if 
                    (x_vals[i+1] - x_vals[i]) * (x_vals[i] - x_vals[i-1]) < 0)
    y_changes = sum(1 for i in range(1, len(y_vals)-1) if 
                    (y_vals[i+1] - y_vals[i]) * (y_vals[i] - y_vals[i-1]) < 0)
    
    # Calculate last and previous velocity vectors
    last_vel = (track.positions[-1][0] - track.positions[-2][0],
                track.positions[-1][1] - track.positions[-2][1])
    prev_vel = (track.positions[-2][0] - track.positions[-3][0],
                track.positions[-2][1] - track.positions[-3][1])
    
    # Check if we're near an inflection point in the sinusoidal movement
    # This is where direction changes most rapidly
    inflection_point_x = (last_vel[0] * prev_vel[0] < 0)
    inflection_point_y = (last_vel[1] * prev_vel[1] < 0)
    
    # Make prediction
    pred_x = track.positions[-1][0]
    pred_y = track.positions[-1][1]
    
    # For directions with sinusoidal behavior
    if x_changes > window_size * 0.2:  # Significant x oscillation
        if inflection_point_x:
            # Near turning point - dampened velocity
            pred_x += int(last_vel[0] * 0.5)
        else:
            # Use velocity + slight acceleration for prediction
            pred_x += int(last_vel[0] + 0.3 * (last_vel[0] - prev_vel[0]))
    else:
        # More linear movement - use average velocity
        pred_x += int(avg_x_vel)
    
    if y_changes > window_size * 0.2:  # Significant y oscillation
        if inflection_point_y:
            # Near turning point - dampened velocity
            pred_y += int(last_vel[1] * 0.5)
        else:
            # Use velocity + slight acceleration for prediction
            pred_y += int(last_vel[1] + 0.3 * (last_vel[1] - prev_vel[1]))
    else:
        # More linear movement - use average velocity
        pred_y += int(avg_y_vel)
    
    return (pred_x, pred_y)

def track_objects(
    csv_files: List[str],
    max_distance: float = 20,
    max_frames_gap: int = 5,
    min_area: int = 1
) -> List[SingleTrack]:
    """Track objects across frames using detection CSVs with timestamp preservation."""
    tracks: List[SingleTrack] = []
    print(f"Tracking objects across {len(csv_files)} frames with timestamp preservation...")
    
    for frame_idx, csv_file in tqdm(enumerate(csv_files)):
        # Load CSV with timestamp information
        df = load_csv_with_timestamps(csv_file)
        
        detections = []
        for _, row in df.iterrows():
            if row['area'] >= min_area:
                detection = {
                    'x': row['x'],
                    'y': row['y'],
                    'area': row['area'],
                    'timestamp': row['timestamp']
                }
                detections.append(detection)
        
        # Remove old tracks
        active_tracks = [t for t in tracks if frame_idx - t.last_frame <= max_frames_gap]
        matched_detection_indices = set()
        
        if active_tracks and detections:
            # Calculate distances differently based on track pattern
            for i, track in enumerate(active_tracks):
                # Classify track motion if enough history
                if len(track.positions) > 5:
                    track.motion_pattern = track.classify_motion_pattern()
                    
            # Apply different prediction methods based on pattern
            pred_positions = []
            for track in active_tracks:
                if track.motion_pattern == "sinusoidal" and len(track.positions) >= 4:
                    # Use a more sophisticated prediction for sinusoidal
                    pred_pos = predict_sinusoidal(track)
                    pred_positions.append(pred_pos)
                else:
                    # Use standard prediction for other patterns
                    pred_positions.append(track.predict_next_position())
            
            pred_positions = np.array(pred_positions)
            det_positions = np.array([[d['x'], d['y']] for d in detections])
            dists = np.linalg.norm(det_positions[:, None, :] - pred_positions[None, :, :], axis=2)
            
            while True:
                min_idx = np.unravel_index(np.argmin(dists), dists.shape)
                det_idx, track_idx = min_idx
                
                # Add motion pattern continuity bonus
                if active_tracks[track_idx].motion_pattern == "sinusoidal":
                    # Give advantage to sinusoidal tracks (reduce distance)
                    dists[det_idx, track_idx] *= 0.8  # 20% bonus
                    
                if dists[det_idx, track_idx] > max_distance:
                    break
                if det_idx in matched_detection_indices:
                    dists[det_idx, :] = np.inf
                    continue
                    
                detection = detections[det_idx]
                track = active_tracks[track_idx]
                
                # Add detection to track WITH timestamp
                track.positions.append((detection['x'], detection['y']))
                track.areas.append(detection['area'])
                track.frame_indices.append(frame_idx)
                track.timestamps.append(detection['timestamp'])  # NEW: Add timestamp
                
                matched_detection_indices.add(det_idx)
                dists[det_idx, :] = np.inf
                dists[:, track_idx] = np.inf
                
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                new_track = SingleTrack(
                    positions=[(detection['x'], detection['y'])],
                    areas=[detection['area']],
                    frame_indices=[frame_idx],
                    timestamps=[detection['timestamp']]  # NEW: Include timestamp
                )
                tracks.append(new_track)
    
    return tracks

def save_tracks(tracks: List[SingleTrack], output_csv: str):
    """Save tracks to a CSV file with timestamps."""
    rows = []
    for tid, track in enumerate(tracks):
        for pos, area, frame, timestamp in zip(track.positions, track.areas, track.frame_indices, track.timestamps):
            rows.append({
                'track_id': tid,
                'frame': frame,
                'x': pos[0],
                'y': pos[1],
                'area': area,
                'motion_pattern': track.motion_pattern,
                'timestamp': timestamp  # NEW: Include timestamp
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved {len(tracks)} tracks with timestamps to {output_csv}")

def save_tracking_report(
    tracks: List[SingleTrack], 
    output_file: str, 
    parameters: Dict,
    csv_files: List[str]
):
    """Save a detailed tracking report with parameters and statistics."""
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("OBJECT TRACKING REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Parameters section
        f.write("TRACKING PARAMETERS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Max distance threshold: {parameters['max_distance']} pixels\n")
        f.write(f"Max frames gap: {parameters['max_frames_gap']} frames\n")
        f.write(f"Min area threshold: {parameters['min_area']} pixels²\n")
        f.write(f"Total frames processed: {len(csv_files)}\n")
        f.write(f"Input directory: {parameters.get('input_dir', 'N/A')}\n")
        f.write(f"Prefix: {parameters.get('prefix', 'N/A')}\n\n")
        
        # Overall statistics
        f.write("TRACKING STATISTICS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"★ Total tracks found: {len(tracks)}\n")
        
        # Track length statistics
        track_lengths = [len(track.positions) for track in tracks]
        if track_lengths:
            f.write(f"★ Average track length: {np.mean(track_lengths):.1f} frames\n")
            f.write(f"★ Median track length: {np.median(track_lengths):.1f} frames\n")
            f.write(f"★ Longest track: {max(track_lengths)} frames\n")
            f.write(f"★ Shortest track: {min(track_lengths)} frames\n")
        
        # Track quality statistics
        long_tracks = [t for t in tracks if len(t.positions) >= 10]
        medium_tracks = [t for t in tracks if 5 <= len(t.positions) < 10]
        short_tracks = [t for t in tracks if len(t.positions) < 5]
        
        f.write(f"★ Long tracks (≥10 frames): {len(long_tracks)} ({len(long_tracks)/len(tracks)*100:.1f}%)\n")
        f.write(f"★ Medium tracks (5-9 frames): {len(medium_tracks)} ({len(medium_tracks)/len(tracks)*100:.1f}%)\n")
        f.write(f"★ Short tracks (<5 frames): {len(short_tracks)} ({len(short_tracks)/len(tracks)*100:.1f}%)\n\n")
        
        # Motion pattern analysis
        motion_patterns = {}
        for track in tracks:
            pattern = track.motion_pattern
            motion_patterns[pattern] = motion_patterns.get(pattern, 0) + 1
        
        f.write("MOTION PATTERN ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        for pattern, count in motion_patterns.items():
            f.write(f"★ {pattern.capitalize()} motion: {count} tracks ({count/len(tracks)*100:.1f}%)\n")
        f.write("\n")
        
        # Detailed track information
        f.write("DETAILED TRACK INFORMATION:\n")
        f.write("-" * 30 + "\n")
        f.write("Track ID | Length | Motion Pattern | Start Frame | End Frame | Avg Area\n")
        f.write("-" * 70 + "\n")
        
        for tid, track in enumerate(tracks):
            avg_area = np.mean(track.areas) if track.areas else 0
            f.write(f"{tid:8d} | {len(track.positions):6d} | {track.motion_pattern:13s} | "
                   f"{track.frame_indices[0]:10d} | {track.frame_indices[-1]:9d} | {avg_area:8.1f}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("Report generated successfully!\n")

# def visualize_tracks(
#     tracks: List[SingleTrack],
#     image_dir: str,
#     prefix: str,
#     output_dir: str,
#     debug_frames: List[int] = None,
#     target_tracks: List[int] = None,
#     track_fade_frames: int = 10,
#     show_trajectory_length: int = None,
#     show_legend: bool = True,
#     show_epipolar_lines: bool = False,
#     # epipolar_params: dict = None,
#     # stereo_rectifier = None,
#     show_area: bool = False,
#     fixed_canvas_size: Tuple[int, int] = None
# ):
#     """Visualize and save annotated images with tracks on black background."""
#     print(f"Visualizing {len(tracks)} tracks on black background...")
#     print(f"Track fade-out: {track_fade_frames} frames")
#     print(f"Trajectory length: {show_trajectory_length if show_trajectory_length else 'unlimited'}")
#     print(f"Show legend: {show_legend}")
#     print(f"Show epipolar lines: {show_epipolar_lines}")
    
#     # Validate inputs
#     if not tracks:
#         print("⚠️ No tracks provided to visualize")
#         return
    
#     if debug_frames is not None:
#         # Filter out None values from debug_frames and ensure they're integers
#         debug_frames = [f for f in debug_frames if f is not None and isinstance(f, (int, float))]
#         if not debug_frames:
#             print("⚠️ No valid frames in debug_frames after filtering None values")
#             return
#         print(f"Debug frames: {debug_frames}")
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Set default epipolar parameters
#     # if epipolar_params is None:
#     #     epipolar_params = {
#     #         'line_spacing': 50,
#     #         'line_color': (128, 128, 128),
#     #         'line_thickness': 1
#     #     }
    
#     # Determine canvas dimensions
#     if fixed_canvas_size is not None:
#         canvas_width, canvas_height = fixed_canvas_size
#         offset_x = 0  # No offset needed for fixed canvas
#         offset_y = 0
#         print(f"Using fixed canvas size: {canvas_width} x {canvas_height}")
#     else:
#         # Fallback to dynamic sizing
#         all_x = []
#         all_y = []
#         for track in tracks:
#             for pos in track.positions:
#                 if pos is not None and len(pos) >= 2 and all(isinstance(coord, (int, float)) for coord in pos):
#                     all_x.append(pos[0])
#                     all_y.append(pos[1])
        
#         if not all_x or not all_y:
#             print("⚠️ No valid track positions found!")
#             return
        
#         # Create canvas size with some padding
#         min_x, max_x = int(min(all_x)), int(max(all_x))
#         min_y, max_y = int(min(all_y)), int(max(all_y))
#         padding = 50
        
#         canvas_width = max_x - min_x + 2 * padding
#         canvas_height = max_y - min_y + 2 * padding
        
#         # Offset to center the tracks
#         offset_x = padding - min_x
#         offset_y = padding - min_y
#         print(f"Using dynamic canvas size: {canvas_width} x {canvas_height}")
#         print(f"Track bounds: x({min_x}, {max_x}), y({min_y}, {max_y})")
    
#     # Generate colors for tracks
#     colors = [tuple(np.random.randint(50, 255, 3).tolist()) for _ in range(len(tracks))]
    
#     # Get all frame indices that we want to visualize
#     if debug_frames is not None:
#         frames_to_process = debug_frames
#     else:
#         all_frame_numbers = set()
#         for track in tracks:
#             # Filter out None values and non-numeric values from frame_indices
#             valid_frames = [f for f in track.frame_indices 
#                            if f is not None and isinstance(f, (int, float)) and not np.isnan(f)]
#             all_frame_numbers.update(valid_frames)
#         frames_to_process = sorted(list(all_frame_numbers))
    
#     if not frames_to_process:
#         print("⚠️ No valid frames to process")
#         return
    
#     print(f"Processing {len(frames_to_process)} frames: {frames_to_process}")
    
#     for frame_idx in tqdm(frames_to_process):
#         # Validate frame_idx more thoroughly
#         if frame_idx is None or not isinstance(frame_idx, (int, float)) or np.isnan(frame_idx):
#             print(f"⚠️ Skipping invalid frame_idx: {frame_idx}")
#             continue
            
#         frame_idx = int(frame_idx)  # Ensure it's an integer
        
#         # Create black canvas with fixed dimensions
#         img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
#         # Draw epipolar lines if requested
#         if show_epipolar_lines:
#             draw_epipolar_lines(img, canvas_width, canvas_height)
        
#         # Track visibility and activity counters
#         active_tracks_count = 0
#         visible_tracks_count = 0
        
#         # Draw track trajectories and current positions
#         for tid, track in enumerate(tracks):
#             # Skip if not a target track when target_tracks is specified
#             if target_tracks and tid not in target_tracks:
#                 continue
            
#             # Validate track data more thoroughly
#             if not track.frame_indices or not track.positions:
#                 continue
                
#             # Filter out None values and invalid values from track data
#             valid_frame_data = []
#             for i, frame in enumerate(track.frame_indices):
#                 if (frame is not None and 
#                     isinstance(frame, (int, float)) and 
#                     not np.isnan(frame) and 
#                     i < len(track.positions) and 
#                     track.positions[i] is not None and
#                     len(track.positions[i]) >= 2 and
#                     all(isinstance(coord, (int, float)) and not np.isnan(coord) for coord in track.positions[i])):
#                     valid_frame_data.append((int(frame), track.positions[i]))
            
#             if not valid_frame_data:
#                 continue
            
#             valid_frames = [frame for frame, pos in valid_frame_data]
            
#             # Check if track is currently active
#             is_active = frame_idx in valid_frames
            
#             # Find the last frame this track was active
#             last_active_frame = -1
#             for frame in valid_frames:
#                 try:
#                     if isinstance(frame, (int, float)) and frame <= frame_idx:
#                         last_active_frame = max(last_active_frame, int(frame))
#                 except (TypeError, ValueError) as e:
#                     print(f"⚠️ Error comparing frame {frame} with {frame_idx}: {e}")
#                     continue
            
#             # Calculate frames since last activity
#             if last_active_frame >= 0:
#                 frames_since_active = frame_idx - last_active_frame
#             else:
#                 frames_since_active = float('inf')
            
#             # Determine if track should be visible
#             should_show_track = (
#                 is_active or  # Currently active
#                 (track_fade_frames == 0) or  # No fade-out (show forever)
#                 (frames_since_active <= track_fade_frames)  # Within fade-out window
#             )
            
#             if not should_show_track:
#                 continue
                
#             visible_tracks_count += 1
#             if is_active:
#                 active_tracks_count += 1
            
#             # Calculate alpha/opacity for fading effect
#             if is_active:
#                 alpha = 1.0
#             elif track_fade_frames > 0:
#                 alpha = max(0.1, 1.0 - (frames_since_active / track_fade_frames))
#             else:
#                 alpha = 0.8  # Slightly faded for inactive but permanent tracks
            
#             # Apply alpha to color
#             faded_color = tuple(int(c * alpha) for c in colors[tid])
            
#             # Draw current position if present in this frame
#             if is_active:
#                 try:
#                     # Find the position for this exact frame
#                     pos = None
#                     for frame, position in valid_frame_data:
#                         if frame == frame_idx:
#                             pos = position
#                             break
                    
#                     if pos is not None and len(pos) >= 2:
#                         # Use coordinates with offset
#                         canvas_x = int(pos[0] + offset_x)
#                         canvas_y = int(pos[1] + offset_y)
                        
#                         # Ensure coordinates are within canvas bounds
#                         if 0 <= canvas_x < canvas_width and 0 <= canvas_y < canvas_height:
#                             # Draw circle for current position (brighter for active)
#                             cv2.circle(img, (canvas_x, canvas_y), 8, colors[tid], -1)
                            
#                             # Draw track ID with background for better visibility
#                             text_pos = (canvas_x + 12, canvas_y - 12)
#                             if text_pos[0] + 30 < canvas_width and text_pos[1] - 15 >= 0:
#                                 cv2.rectangle(img, (text_pos[0]-2, text_pos[1]-15), (text_pos[0]+30, text_pos[1]+5), (0, 0, 0), -1)
#                                 cv2.putText(img, f"T{tid}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[tid], 2)
                            
#                             if show_area is True:
#                                 # Find corresponding area
#                                 try:
#                                     original_idx = track.frame_indices.index(frame_idx)
#                                     if original_idx < len(track.areas):
#                                         area = track.areas[original_idx]
#                                         area_pos = (canvas_x + 12, canvas_y + 15)
#                                         if area_pos[0] < canvas_width and area_pos[1] < canvas_height:
#                                             cv2.putText(img, f"{area:.0f}px", area_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[tid], 1)
#                                 except (ValueError, IndexError):
#                                     pass  # Skip area display if not found
                            
#                 except (ValueError, IndexError, TypeError) as e:
#                     print(f"⚠️ Error drawing track {tid} at frame {frame_idx}: {e}")
#                     continue
            
#             # Draw trajectory up to current frame
#             track_points = []
#             track_frames = []
#             for frame, pos in valid_frame_data:
#                 if frame <= frame_idx:
#                     canvas_pos = (int(pos[0] + offset_x), int(pos[1] + offset_y))
#                     # Only add points that are within canvas bounds
#                     if 0 <= canvas_pos[0] < canvas_width and 0 <= canvas_pos[1] < canvas_height:
#                         track_points.append(canvas_pos)
#                         track_frames.append(frame)
            
#             # # Limit trajectory length if specified
#             if show_trajectory_length and len(track_points) > show_trajectory_length:
#                 track_points = track_points[-show_trajectory_length:]
#                 track_frames = track_frames[-show_trajectory_length:]
            
#             # Draw trajectory lines with fading
#             if len(track_points) > 1:
#                 for j in range(1, len(track_points)):
#                     pt1 = track_points[j-1]
#                     pt2 = track_points[j]
                    
#                     # Calculate line alpha based on age
#                     line_frame = track_frames[j]
#                     line_age = frame_idx - line_frame
                    
#                     if track_fade_frames > 0 and not is_active:
#                         line_alpha = max(0.1, 1.0 - (line_age / (track_fade_frames + 10)))
#                     else:
#                         line_alpha = max(0.3, 1.0 - (line_age / 30))  # Gradual fade for trajectory
                    
#                     line_color = tuple(int(c * 1) for c in colors[tid])
#                     cv2.line(img, pt1, pt2, line_color, 2)
        
#         # Add frame information to the image (only if show_legend is True)
#         if show_legend:
#             cv2.rectangle(img, (10, 10), (400, 150), (40, 40, 40), -1)  # Dark gray background
#             cv2.putText(img, f"Frame: {frame_idx}", (20, 40), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
#             cv2.putText(img, f"Active tracks: {active_tracks_count}", (20, 70), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
#             cv2.putText(img, f"Visible tracks: {visible_tracks_count}", (20, 100), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
#             cv2.putText(img, f"Total tracks: {len(tracks)}", (20, 130), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
#         # Save the annotated image
#         output_filename = f"{prefix}_tracking_frame_{frame_idx:03d}.png"
#         out_path = os.path.join(output_dir, output_filename)
#         cv2.imwrite(out_path, img)
    
#     # Create overview image showing all tracks
#     # create_tracks_overview_black(tracks, canvas_width, canvas_height, offset_x, offset_y, prefix, output_dir, colors, show_epipolar_lines, epipolar_params, stereo_rectifier)
    
#     print(f"✓ Visualization complete! Saved {len(frames_to_process)} frames to {output_dir}")
def visualize_tracks(
    tracks: List[SingleTrack],
    image_dir: str,
    prefix: str,
    output_dir: str,
    debug_frames: List[int] = None,
    target_tracks: List[int] = None,
    track_fade_frames: int = 10,
    show_trajectory_length: int = None,
    show_legend: bool = True,
    show_epipolar_lines: bool = False,
    show_area: bool = False,
    fixed_canvas_size: Tuple[int, int] = None
):
    """Visualize and save annotated images with tracks on black background."""
    print(f"Visualizing {len(tracks)} tracks on black background...")
    print(f"Track fade-out: {track_fade_frames} frames")
    print(f"Trajectory length: {show_trajectory_length if show_trajectory_length else 'unlimited'}")
    print(f"Show legend: {show_legend}")
    print(f"Show epipolar lines: {show_epipolar_lines}")
    
    # Validate inputs
    if not tracks:
        print("⚠️ No tracks provided to visualize")
        return
    
    if debug_frames is not None:
        debug_frames = [f for f in debug_frames if f is not None and isinstance(f, (int, float))]
        if not debug_frames:
            print("⚠️ No valid frames in debug_frames after filtering None values")
            return
        print(f"Debug frames: {debug_frames}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine canvas dimensions (same as before)
    if fixed_canvas_size is not None:
        canvas_width, canvas_height = fixed_canvas_size
        offset_x = 0
        offset_y = 0
        print(f"Using fixed canvas size: {canvas_width} x {canvas_height}")
    else:
        all_x = []
        all_y = []
        for track in tracks:
            for pos in track.positions:
                if pos is not None and len(pos) >= 2 and all(isinstance(coord, (int, float)) for coord in pos):
                    all_x.append(pos[0])
                    all_y.append(pos[1])
        
        if not all_x or not all_y:
            print("⚠️ No valid track positions found!")
            return
        
        min_x, max_x = int(min(all_x)), int(max(all_x))
        min_y, max_y = int(min(all_y)), int(max(all_y))
        padding = 50
        
        canvas_width = max_x - min_x + 2 * padding
        canvas_height = max_y - min_y + 2 * padding
        
        offset_x = padding - min_x
        offset_y = padding - min_y
        print(f"Using dynamic canvas size: {canvas_width} x {canvas_height}")
        print(f"Track bounds: x({min_x}, {max_x}), y({min_y}, {max_y})")
    
    # Generate colors for tracks
    colors = [tuple(np.random.randint(50, 255, 3).tolist()) for _ in range(len(tracks))]
    
    # Get all frame indices that we want to visualize
    if debug_frames is not None:
        frames_to_process = debug_frames
    else:
        all_frame_numbers = set()
        for track in tracks:
            valid_frames = [f for f in track.frame_indices 
                           if f is not None and isinstance(f, (int, float)) and not np.isnan(f)]
            all_frame_numbers.update(valid_frames)
        frames_to_process = sorted(list(all_frame_numbers))
    
    if not frames_to_process:
        print("⚠️ No valid frames to process")
        return
    
    print(f"Processing {len(frames_to_process)} frames: {frames_to_process}")
    
    # Build a frame-to-timestamp mapping from all tracks
    frame_timestamp_map = {}
    for track in tracks:
        for frame_idx, timestamp in zip(track.frame_indices, track.timestamps):
            if frame_idx is not None and timestamp is not None:
                # Use the first timestamp found for each frame (should be consistent across tracks)
                if frame_idx not in frame_timestamp_map:
                    frame_timestamp_map[frame_idx] = timestamp
    
    print(f"Built timestamp mapping for {len(frame_timestamp_map)} frames")
    
    for frame_idx in tqdm(frames_to_process):
        # Validate frame_idx
        if frame_idx is None or not isinstance(frame_idx, (int, float)) or np.isnan(frame_idx):
            print(f"⚠️ Skipping invalid frame_idx: {frame_idx}")
            continue
            
        frame_idx = int(frame_idx)
        
        # Get timestamp for this frame
        frame_timestamp = frame_timestamp_map.get(frame_idx, None)
        
        # Create black canvas with fixed dimensions
        img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Draw epipolar lines if requested
        if show_epipolar_lines:
            draw_epipolar_lines(img, canvas_width, canvas_height)
        
        # Track visibility and activity counters
        active_tracks_count = 0
        visible_tracks_count = 0
        
        # Draw track trajectories and current positions (same logic as before)
        for tid, track in enumerate(tracks):
            if target_tracks and tid not in target_tracks:
                continue
            
            if not track.frame_indices or not track.positions:
                continue
                
            # Filter out None values and invalid values from track data
            valid_frame_data = []
            for i, frame in enumerate(track.frame_indices):
                if (frame is not None and 
                    isinstance(frame, (int, float)) and 
                    not np.isnan(frame) and 
                    i < len(track.positions) and 
                    track.positions[i] is not None and
                    len(track.positions[i]) >= 2 and
                    all(isinstance(coord, (int, float)) and not np.isnan(coord) for coord in track.positions[i])):
                    valid_frame_data.append((int(frame), track.positions[i]))
            
            if not valid_frame_data:
                continue
            
            valid_frames = [frame for frame, pos in valid_frame_data]
            is_active = frame_idx in valid_frames
            
            # Find the last frame this track was active
            last_active_frame = -1
            for frame in valid_frames:
                try:
                    if isinstance(frame, (int, float)) and frame <= frame_idx:
                        last_active_frame = max(last_active_frame, int(frame))
                except (TypeError, ValueError) as e:
                    print(f"⚠️ Error comparing frame {frame} with {frame_idx}: {e}")
                    continue
            
            # Calculate frames since last activity
            if last_active_frame >= 0:
                frames_since_active = frame_idx - last_active_frame
            else:
                frames_since_active = float('inf')
            
            # Determine if track should be visible
            should_show_track = (
                is_active or
                (track_fade_frames == 0) or
                (frames_since_active <= track_fade_frames)
            )
            
            if not should_show_track:
                continue
                
            visible_tracks_count += 1
            if is_active:
                active_tracks_count += 1
            
            # Calculate alpha/opacity for fading effect
            if is_active:
                alpha = 1.0
            elif track_fade_frames > 0:
                alpha = max(0.1, 1.0 - (frames_since_active / track_fade_frames))
            else:
                alpha = 0.8
            
            faded_color = tuple(int(c * alpha) for c in colors[tid])
            
            # Draw current position if present in this frame
            if is_active:
                try:
                    pos = None
                    for frame, position in valid_frame_data:
                        if frame == frame_idx:
                            pos = position
                            break
                    
                    if pos is not None and len(pos) >= 2:
                        canvas_x = int(pos[0] + offset_x)
                        canvas_y = int(pos[1] + offset_y)
                        
                        if 0 <= canvas_x < canvas_width and 0 <= canvas_y < canvas_height:
                            cv2.circle(img, (canvas_x, canvas_y), 8, colors[tid], -1)
                            
                            text_pos = (canvas_x + 12, canvas_y - 12)
                            if text_pos[0] + 30 < canvas_width and text_pos[1] - 15 >= 0:
                                cv2.rectangle(img, (text_pos[0]-2, text_pos[1]-15), (text_pos[0]+30, text_pos[1]+5), (0, 0, 0), -1)
                                cv2.putText(img, f"T{tid}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[tid], 2)
                            
                            if show_area is True:
                                try:
                                    original_idx = track.frame_indices.index(frame_idx)
                                    if original_idx < len(track.areas):
                                        area = track.areas[original_idx]
                                        area_pos = (canvas_x + 12, canvas_y + 15)
                                        if area_pos[0] < canvas_width and area_pos[1] < canvas_height:
                                            cv2.putText(img, f"{area:.0f}px", area_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[tid], 1)
                                except (ValueError, IndexError):
                                    pass
                            
                except (ValueError, IndexError, TypeError) as e:
                    print(f"⚠️ Error drawing track {tid} at frame {frame_idx}: {e}")
                    continue
            
            # Draw trajectory up to current frame
            track_points = []
            track_frames = []
            for frame, pos in valid_frame_data:
                if frame <= frame_idx:
                    canvas_pos = (int(pos[0] + offset_x), int(pos[1] + offset_y))
                    if 0 <= canvas_pos[0] < canvas_width and 0 <= canvas_pos[1] < canvas_height:
                        track_points.append(canvas_pos)
                        track_frames.append(frame)
            
            if show_trajectory_length and len(track_points) > show_trajectory_length:
                track_points = track_points[-show_trajectory_length:]
                track_frames = track_frames[-show_trajectory_length:]
            
            # Draw trajectory lines with fading
            if len(track_points) > 1:
                for j in range(1, len(track_points)):
                    pt1 = track_points[j-1]
                    pt2 = track_points[j]
                    
                    line_frame = track_frames[j]
                    line_age = frame_idx - line_frame
                    
                    if track_fade_frames > 0 and not is_active:
                        line_alpha = max(0.1, 1.0 - (line_age / (track_fade_frames + 10)))
                    else:
                        line_alpha = max(0.3, 1.0 - (line_age / 30))
                    
                    line_color = tuple(int(c * 1) for c in colors[tid])
                    cv2.line(img, pt1, pt2, line_color, 2)
        
        # Add frame information to the image (enhanced to show timestamp)
        if show_legend:
            cv2.rectangle(img, (10, 10), (450, 180), (40, 40, 40), -1)  # Larger box for timestamp
            cv2.putText(img, f"Frame: {frame_idx}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Display timestamp if available
            if frame_timestamp:
                # Format timestamp for display
                timestamp_display = _format_timestamp_for_display(frame_timestamp)
                cv2.putText(img, f"Time: {timestamp_display}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
            
            cv2.putText(img, f"Active tracks: {active_tracks_count}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.putText(img, f"Visible tracks: {visible_tracks_count}", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            
            cv2.putText(img, f"Total tracks: {len(tracks)}", (20, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Generate filename with timestamp
        output_filename = _generate_timestamped_filename(prefix, frame_idx, frame_timestamp)
        out_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(out_path, img)
    
    print(f"✓ Visualization complete! Saved {len(frames_to_process)} frames to {output_dir}")

def _format_timestamp_for_display(timestamp: str) -> str:
    """Format timestamp for display in the visualization"""
    try:
        if 'T' in timestamp:
            # ISO format: 2025-04-04T11:45:58.108161
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%H:%M:%S.%f')[:-3]  # Show milliseconds
        elif 'time_' in timestamp and 's' in timestamp:
            # Frame-based format: frame_000123_time_4.100s
            time_part = timestamp.split('time_')[1].replace('s', '')
            return f"{float(time_part):.3f}s"
        else:
            # Return as-is if format is unknown
            return timestamp[:20]  # Truncate if too long
    except Exception as e:
        print(f"Error formatting timestamp {timestamp}: {e}")
        return timestamp[:20]

def _generate_timestamped_filename(prefix: str, frame_idx: int, frame_timestamp: Optional[str]) -> str:
    """Generate filename with timestamp information"""
    try:
        if frame_timestamp is None:
            # Fallback to frame index only
            return f"{prefix}_tracking_frame_{frame_idx:03d}.png"
        
        if 'T' in frame_timestamp:
            # ISO format: 2025-04-04T11:45:58.108161
            dt = datetime.fromisoformat(frame_timestamp.replace('Z', '+00:00'))
            # Format as: prefix_tracking_20250404_114558_108_frame_123.png
            timestamp_str = dt.strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Remove last 3 digits of microseconds
            return f"{prefix}_tracking_{timestamp_str}_frame_{frame_idx:03d}.png"
        
        elif 'time_' in frame_timestamp and 's' in frame_timestamp:
            # Frame-based format: frame_000123_time_4.100s
            time_part = frame_timestamp.split('time_')[1].replace('s', '')
            time_formatted = f"{float(time_part):08.3f}s".replace('.', '_')
            return f"{prefix}_tracking_time_{time_formatted}_frame_{frame_idx:03d}.png"
        
        else:
            # Unknown format - sanitize and use
            safe_timestamp = re.sub(r'[^\w\-_.]', '_', frame_timestamp)[:20]
            return f"{prefix}_tracking_{safe_timestamp}_frame_{frame_idx:03d}.png"
    
    except Exception as e:
        print(f"Error generating timestamped filename for frame {frame_idx}: {e}")
        # Fallback to frame index only
        return f"{prefix}_tracking_frame_{frame_idx:03d}.png"

# Add the visualize_tracks_with_colors function for the annotation app:
def visualize_tracks_with_colors(
    tracks: List[SingleTrack],
    colors: List[Tuple[int, int, int]],
    image_dir: str,
    prefix: str,
    output_dir: str,
    debug_frames: List[int] = None,
    target_tracks: List[int] = None,
    track_fade_frames: int = 10,
    show_trajectory_length: int = None,
    show_legend: bool = True,
    show_epipolar_lines: bool = False,
    show_area: bool = False,
    fixed_canvas_size: Tuple[int, int] = None
):
    """Visualize tracks with custom colors - same as visualize_tracks but uses provided colors"""
    print(f"Visualizing {len(tracks)} tracks with custom colors...")
    
    # Validate inputs
    if not tracks:
        print("⚠️ No tracks provided to visualize")
        return
    
    if len(colors) != len(tracks):
        print(f"⚠️ Color count ({len(colors)}) doesn't match track count ({len(tracks)})")
        # Extend colors if needed
        while len(colors) < len(tracks):
            colors.append(tuple(np.random.randint(50, 255, 3).tolist()))
    
    if debug_frames is not None:
        debug_frames = [f for f in debug_frames if f is not None and isinstance(f, (int, float))]
        if not debug_frames:
            print("⚠️ No valid frames in debug_frames after filtering None values")
            return
        print(f"Debug frames: {debug_frames}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine canvas dimensions (same logic as original function)
    if fixed_canvas_size is not None:
        canvas_width, canvas_height = fixed_canvas_size
        offset_x = 0
        offset_y = 0
        print(f"Using fixed canvas size: {canvas_width} x {canvas_height}")
    else:
        all_x = []
        all_y = []
        for track in tracks:
            for pos in track.positions:
                if pos is not None and len(pos) >= 2 and all(isinstance(coord, (int, float)) for coord in pos):
                    all_x.append(pos[0])
                    all_y.append(pos[1])
        
        if not all_x or not all_y:
            print("⚠️ No valid track positions found!")
            return
        
        min_x, max_x = int(min(all_x)), int(max(all_x))
        min_y, max_y = int(min(all_y)), int(max(all_y))
        padding = 50
        
        canvas_width = max_x - min_x + 2 * padding
        canvas_height = max_y - min_y + 2 * padding
        
        offset_x = padding - min_x
        offset_y = padding - min_y
        print(f"Using dynamic canvas size: {canvas_width} x {canvas_height}")
    
    # Get frames to process (same logic as original)
    if debug_frames is not None:
        frames_to_process = debug_frames
    else:
        all_frame_numbers = set()
        for track in tracks:
            valid_frames = [f for f in track.frame_indices 
                           if f is not None and isinstance(f, (int, float)) and not np.isnan(f)]
            all_frame_numbers.update(valid_frames)
        frames_to_process = sorted(list(all_frame_numbers))
    
    if not frames_to_process:
        print("⚠️ No valid frames to process")
        return
    
    print(f"Processing {len(frames_to_process)} frames: {frames_to_process}")
    
    # Build frame-to-timestamp mapping
    frame_timestamp_map = {}
    for track in tracks:
        for frame_idx, timestamp in zip(track.frame_indices, track.timestamps):
            if frame_idx is not None and timestamp is not None:
                if frame_idx not in frame_timestamp_map:
                    frame_timestamp_map[frame_idx] = timestamp
    
    # Process each frame (same logic as original, but use provided colors)
    for frame_idx in tqdm(frames_to_process):
        if frame_idx is None or not isinstance(frame_idx, (int, float)) or np.isnan(frame_idx):
            print(f"⚠️ Skipping invalid frame_idx: {frame_idx}")
            continue
            
        frame_idx = int(frame_idx)
        
        # Get timestamp for this frame
        frame_timestamp = frame_timestamp_map.get(frame_idx, None)
        
        # Create black canvas
        img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        
        # Draw epipolar lines if requested
        if show_epipolar_lines:
            draw_epipolar_lines(img, canvas_width, canvas_height)
        
        active_tracks_count = 0
        visible_tracks_count = 0
        
        # Draw tracks using provided colors (rest of the logic is the same as the original function)
        for tid, track in enumerate(tracks):
            if target_tracks and tid not in target_tracks:
                continue
            
            if not track.frame_indices or not track.positions:
                continue
                
            valid_frame_data = []
            for i, frame in enumerate(track.frame_indices):
                if (frame is not None and 
                    isinstance(frame, (int, float)) and 
                    not np.isnan(frame) and 
                    i < len(track.positions) and 
                    track.positions[i] is not None and
                    len(track.positions[i]) >= 2 and
                    all(isinstance(coord, (int, float)) and not np.isnan(coord) for coord in track.positions[i])):
                    valid_frame_data.append((int(frame), track.positions[i]))
            
            if not valid_frame_data:
                continue
            
            valid_frames = [frame for frame, pos in valid_frame_data]
            is_active = frame_idx in valid_frames
            
            last_active_frame = -1
            for frame in valid_frames:
                try:
                    if isinstance(frame, (int, float)) and frame <= frame_idx:
                        last_active_frame = max(last_active_frame, int(frame))
                except (TypeError, ValueError):
                    continue
            
            if last_active_frame >= 0:
                frames_since_active = frame_idx - last_active_frame
            else:
                frames_since_active = float('inf')
            
            should_show_track = (
                is_active or
                (track_fade_frames == 0) or
                (frames_since_active <= track_fade_frames)
            )
            
            if not should_show_track:
                continue
                
            visible_tracks_count += 1
            if is_active:
                active_tracks_count += 1
            
            # Calculate alpha for fading
            if is_active:
                alpha = 1.0
            elif track_fade_frames > 0:
                alpha = max(0.1, 1.0 - (frames_since_active / track_fade_frames))
            else:
                alpha = 0.8
            
            # Use the provided color for this track
            track_color = colors[tid]
            faded_color = tuple(int(c * alpha) for c in track_color)
            
            # Draw current position
            if is_active:
                try:
                    pos = None
                    for frame, position in valid_frame_data:
                        if frame == frame_idx:
                            pos = position
                            break
                    
                    if pos is not None and len(pos) >= 2:
                        canvas_x = int(pos[0] + offset_x)
                        canvas_y = int(pos[1] + offset_y)
                        
                        if 0 <= canvas_x < canvas_width and 0 <= canvas_y < canvas_height:
                            cv2.circle(img, (canvas_x, canvas_y), 8, track_color, -1)
                            
                            text_pos = (canvas_x + 12, canvas_y - 12)
                            if text_pos[0] + 30 < canvas_width and text_pos[1] - 15 >= 0:
                                cv2.rectangle(img, (text_pos[0]-2, text_pos[1]-15), (text_pos[0]+30, text_pos[1]+5), (0, 0, 0), -1)
                                cv2.putText(img, f"T{tid}", text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 2)
                            
                            if show_area:
                                try:
                                    original_idx = track.frame_indices.index(frame_idx)
                                    if original_idx < len(track.areas):
                                        area = track.areas[original_idx]
                                        area_pos = (canvas_x + 12, canvas_y + 15)
                                        if area_pos[0] < canvas_width and area_pos[1] < canvas_height:
                                            cv2.putText(img, f"{area:.0f}px", area_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, track_color, 1)
                                except (ValueError, IndexError):
                                    pass
                            
                except (ValueError, IndexError, TypeError) as e:
                    print(f"⚠️ Error drawing track {tid} at frame {frame_idx}: {e}")
                    continue
            
            # Draw trajectory
            track_points = []
            track_frames = []
            for frame, pos in valid_frame_data:
                if frame <= frame_idx:
                    canvas_pos = (int(pos[0] + offset_x), int(pos[1] + offset_y))
                    if 0 <= canvas_pos[0] < canvas_width and 0 <= canvas_pos[1] < canvas_height:
                        track_points.append(canvas_pos)
                        track_frames.append(frame)
            
            if show_trajectory_length and len(track_points) > show_trajectory_length:
                track_points = track_points[-show_trajectory_length:]
                track_frames = track_frames[-show_trajectory_length:]
            
            # Draw trajectory lines
            if len(track_points) > 1:
                for j in range(1, len(track_points)):
                    pt1 = track_points[j-1]
                    pt2 = track_points[j]
                    
                    line_frame = track_frames[j]
                    line_age = frame_idx - line_frame
                    
                    if track_fade_frames > 0 and not is_active:
                        line_alpha = max(0.1, 1.0 - (line_age / (track_fade_frames + 10)))
                    else:
                        line_alpha = max(0.3, 1.0 - (line_age / 30))
                    
                    line_color = tuple(int(c * line_alpha) for c in track_color)
                    cv2.line(img, pt1, pt2, line_color, 2)
        
        # Add frame information with timestamp
        if show_legend:
            cv2.rectangle(img, (10, 10), (450, 180), (40, 40, 40), -1)
            cv2.putText(img, f"Frame: {frame_idx}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Display timestamp
            if frame_timestamp:
                timestamp_display = _format_timestamp_for_display(frame_timestamp)
                cv2.putText(img, f"Time: {timestamp_display}", (20, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)
            
            cv2.putText(img, f"Active tracks: {active_tracks_count}", (20, 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(img, f"Visible tracks: {visible_tracks_count}", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(img, f"Total tracks: {len(tracks)}", (20, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        
        # Save the image with timestamped filename
        output_filename = _generate_timestamped_filename(prefix, frame_idx, frame_timestamp)
        out_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(out_path, img)
    
    print(f"✓ Visualization complete! Saved {len(frames_to_process)} frames to {output_dir}")

def draw_epipolar_lines(img, canvas_width, canvas_height):
    """Draw epipolar lines on the image."""
    # if stereo_rectifier is None:
    #     print("⚠ Warning: No stereo rectifier provided for epipolar lines")
    #     return
    
    line_spacing =  50
    line_color = 128, 128, 128
    line_thickness = 1
    
    # # Check if this is a rectified image (horizontal epipolar lines)
    # if 'rectified' in prefix.lower():
        # For rectified images, epipolar lines are horizontal
    for y in range(0, canvas_height, line_spacing):
        cv2.line(img, (0, y), (canvas_width, y), line_color, line_thickness)
    # else:
    #     # For original images, use the fundamental matrix to compute epipolar lines
    #     try:
    #         # Get fundamental matrix from stereo rectifier
    #         if hasattr(stereo_rectifier, 'F') and stereo_rectifier.F is not None:
    #             F = stereo_rectifier.F
                
    #             # Sample points along the image to generate epipolar lines
    #             # Determine which camera this is for
    #             camera_side = 'left' if 'left' in prefix.lower() else 'right'
                
    #             # Generate sample points and compute corresponding epipolar lines
    #             for x in range(0, canvas_width, line_spacing):
    #                 for y in range(0, canvas_height, line_spacing):
    #                     point = np.array([x, y, 1.0])
                        
    #                     if camera_side == 'left':
    #                         # Point in left image, compute epipolar line in right image
    #                         # But since we're drawing on the current image, we need the reverse
    #                         epiline = F.T @ point
    #                     else:
    #                         # Point in right image, compute epipolar line in left image
    #                         epiline = F @ point
                        
    #                     # Draw the epipolar line if it intersects the image
    #                     draw_epiline(img, epiline, canvas_width, canvas_height, line_color, line_thickness)
    #         else:
    #             print("⚠ Warning: Fundamental matrix not available for epipolar lines")
        # except Exception as e:
        #     print(f"⚠ Warning: Failed to draw epipolar lines: {e}")

def draw_epiline(img, line, width, height, color, thickness):
    """Draw an epipolar line on the image."""
    a, b, c = line
    
    if abs(b) > 1e-6:  # Not a vertical line
        # Calculate intersection points with image borders
        x1, y1 = 0, int(-c / b)
        x2, y2 = width, int(-(c + a * width) / b)
        
        # Clip to image boundaries
        if y1 < 0:
            y1 = 0
            x1 = int(-c / a) if abs(a) > 1e-6 else 0
        if y1 >= height:
            y1 = height - 1
            x1 = int(-(c + b * y1) / a) if abs(a) > 1e-6 else 0
            
        if y2 < 0:
            y2 = 0
            x2 = int(-c / a) if abs(a) > 1e-6 else width
        if y2 >= height:
            y2 = height - 1
            x2 = int(-(c + b * y2) / a) if abs(a) > 1e-6 else width
        
        # Draw line if it's within image bounds
        if 0 <= x1 < width and 0 <= x2 < width:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def create_tracks_overview_black(tracks, canvas_width, canvas_height, offset_x, offset_y, prefix, output_dir, colors, show_epipolar_lines=False, epipolar_params=None, stereo_rectifier=None):
    """Create an overview image showing all track trajectories on black background"""
    
    # Create black canvas
    img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    
    # Draw epipolar lines if requested
    # if show_epipolar_lines:
    #     draw_epipolar_lines(img, canvas_width, canvas_height, epipolar_params or {}, stereo_rectifier, prefix)
    
    # Draw all track trajectories
    for tid, track in enumerate(tracks):
        if len(track.positions) > 1:
            # Convert all positions to canvas coordinates
            canvas_points = []
            for pos in track.positions:
                canvas_x = int(pos[0] + offset_x)
                canvas_y = int(pos[1] + offset_y)
                canvas_points.append((canvas_x, canvas_y))
            
            # Draw full trajectory
            for j in range(1, len(canvas_points)):
                pt1 = canvas_points[j-1]
                pt2 = canvas_points[j]
                cv2.line(img, pt1, pt2, colors[tid], 2)
            
            # Mark start and end points
            start_pos = canvas_points[0]
            end_pos = canvas_points[-1]
            
            cv2.circle(img, start_pos, 10, (0, 255, 0), -1)  # Green start
            cv2.circle(img, end_pos, 10, (0, 0, 255), -1)    # Red end
            
            # Add track ID at start
            cv2.putText(img, f"T{tid}", (start_pos[0]+15, start_pos[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[tid], 2)
    
    # Add title with dark background
    cv2.rectangle(img, (10, 10), (450, 120), (40, 40, 40), -1)
    cv2.putText(img, f"{prefix.upper()} - All Tracks Overview", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Total tracks: {len(tracks)}", (20, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Green: Start, Red: End", (20, 100), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save overview
    overview_path = os.path.join(output_dir, f"{prefix}_tracks_overview.png")
    cv2.imwrite(overview_path, img)
    print(f"✓ Track overview saved: {overview_path}")
    
if __name__ == "__main__":
    import sys
    import glob
    prefixes = ['upper','lower']
    input_dir = "/Users/vdausmann/swimming_in_salt/detection_results/20250404_24.4_24_01"
    
    # Define tracking parameters
    tracking_params = {
        'max_distance': 20,
        'max_frames_gap': 5,
        'min_area': 1,
        'input_dir': input_dir
    }
    
    for prefix in prefixes:
        print(f"Processing prefix: {prefix}")
        tracking_params['prefix'] = prefix
        
        # Detect and save objects
        csv_files = sorted(glob.glob(os.path.join(input_dir, f"{prefix}_*_objects.csv")))
        tracks = track_objects(
            csv_files, 
            max_distance=tracking_params['max_distance'],
            max_frames_gap=tracking_params['max_frames_gap'],
            min_area=tracking_params['min_area']
        )
        
        # Save tracks
        save_tracks(tracks, os.path.join(input_dir, f"{prefix}_tracks.csv"))
        
        # Save tracking report
        save_tracking_report(
            tracks, 
            os.path.join(input_dir, f"{prefix}_tracking_report.txt"),
            tracking_params,
            csv_files
        )
        
        # Visualize tracks
        visualize_tracks(
            tracks,
            image_dir=input_dir,
            prefix=prefix,
            output_dir=os.path.join(input_dir, f"{prefix}_tracks_vis"),
            #debug_frames= [66, 67, 68],  # Frame where the switch happens and neighbors
            #target_tracks= [12]     # The track IDs we're investigating
        )
        
        print(f"✓ Completed {prefix}: {len(tracks)} tracks found")
        print(f"✓ Report saved to: {prefix}_tracking_report.txt")