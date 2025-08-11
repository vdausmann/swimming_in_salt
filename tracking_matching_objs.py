import os
import glob
import cv2
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class SingleTrack:
    positions: List[Tuple[int, int]]
    areas: List[float]
    frame_indices: List[int]
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
    """Track objects across frames using detection CSVs."""
    tracks: List[SingleTrack] = []
    print(f"Tracking objects across {len(csv_files)} frames...")
    for frame_idx, csv_file in tqdm(enumerate(csv_files)):
        df = pd.read_csv(csv_file)
        detections = [
            {'x': row['x'], 'y': row['y'], 'area': row['area']}
            for _, row in df.iterrows() if row['area'] >= min_area
        ]
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
                track.positions.append((detection['x'], detection['y']))
                track.areas.append(detection['area'])
                track.frame_indices.append(frame_idx)
                matched_detection_indices.add(det_idx)
                dists[det_idx, :] = np.inf
                dists[:, track_idx] = np.inf
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                new_track = SingleTrack(
                    positions=[(detection['x'], detection['y'])],
                    areas=[detection['area']],
                    frame_indices=[frame_idx]
                )
                tracks.append(new_track)
    return tracks

def save_tracks(tracks: List[SingleTrack], output_csv: str):
    """Save tracks to a CSV file."""
    rows = []
    for tid, track in enumerate(tracks):
        for pos, area, frame in zip(track.positions, track.areas, track.frame_indices):
            rows.append({
                'track_id': tid,
                'frame': frame,
                'x': pos[0],
                'y': pos[1],
                'area': area,
                'motion_pattern': track.motion_pattern
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

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

def visualize_tracks(
    tracks: List[SingleTrack],
    image_dir: str,
    prefix: str,
    output_dir: str,
    debug_frames: List[int] = None,  # Add frames to debug (e.g., [36])
    target_tracks: List[int] = None  # Track IDs to focus on (e.g., [14, 360])
):
    """Visualize and save annotated images with tracks."""
    print(f"Visualizing {len(tracks)} tracks...")
    os.makedirs(output_dir, exist_ok=True)
    # Assume images are named as in the detection CSVs
    image_files = sorted(glob.glob(os.path.join(image_dir, f"{prefix}_*.png")))
    # Build a mapping from frame index to image file
    frame_to_img = {i: f for i, f in enumerate(image_files)}
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(tracks))]
    
    # Get all frame indices
    all_frames = sorted(list(frame_to_img.keys()))
    
    for frame_idx, img_file in tqdm(frame_to_img.items()):
        img = cv2.imread(img_file)
        if img is None:
            continue
            
        # Draw current positions and tracks
        for tid, track in enumerate(tracks):
            # Skip if not a target track when target_tracks is specified
            if target_tracks and tid not in target_tracks:
                continue
                
            # Draw current position if present in this frame
            if frame_idx in track.frame_indices:
                idx = track.frame_indices.index(frame_idx)
                pos = track.positions[idx]
                cv2.circle(img, (int(pos[0]), int(pos[1])), 4, colors[tid], -1)
                cv2.putText(img, f"T{tid}", (int(pos[0])+5, int(pos[1])-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[tid], 1)
                
                # Draw trajectory
                for j in range(1, len(track.positions)):
                    if track.frame_indices[j] > frame_idx:
                        break
                    if track.frame_indices[j-1] <= frame_idx:
                        pt1 = track.positions[j-1]
                        pt2 = track.positions[j]
                        cv2.line(img, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), colors[tid], 2)
            
            # Draw prediction for next frame (if track exists in previous frame)
            if frame_idx > 0 and frame_idx-1 in track.frame_indices:
                # Find the state of the track in the previous frame
                prev_idx = track.frame_indices.index(frame_idx-1)
                
                # Create a temporary track with data up to previous frame
                temp_track = SingleTrack(
                    positions=track.positions[:prev_idx+1],
                    areas=track.areas[:prev_idx+1],
                    frame_indices=track.frame_indices[:prev_idx+1]
                )
                
                #  
        
        # Add frame number to the image
        cv2.putText(img, f"Frame: {frame_idx}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add extra debugging for critical frames
        if debug_frames and frame_idx in debug_frames:
            cv2.putText(img, f"DEBUG FRAME", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        
        out_path = os.path.join(output_dir, os.path.basename(img_file))
        cv2.imwrite(out_path, img)

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