#!/Users/vdausmann/miniforge3/envs/cv/bin/python
"""
Track Annotation and Validation Tool

A Plotly Dash application for annotating and validating object tracks
across stereo camera image sequences. Uses the cv conda environment.
"""

import os
import sys
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import colorsys
import tempfile
import re

import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2

import dash
from dash import dcc, html, Input, Output, State, callback_context, ALL
import dash_bootstrap_components as dbc

# Import your tracking visualization functions
from tracking_matching_objs import SingleTrack, visualize_tracks

# Configuration - set your base directory here
BASE_DATA_DIR = "/Users/vdausmann/swimming_in_salt/detection_results"  # Change this to your base directory

def get_available_data_directories(base_dir: str) -> List[Dict]:
    """
    Scan the base directory for subdirectories that contain track data files.
    Returns a list of dictionaries with 'label' and 'value' for dropdown options.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    data_dirs = []
    
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            # Check if directory contains required track files
            upper_tracks = subdir / "upper_tracks.csv"
            lower_tracks = subdir / "lower_tracks.csv"
            
            if upper_tracks.exists() and lower_tracks.exists():
                # Count images to show in dropdown
                upper_images = len(list(subdir.glob("upper_*.png")))
                lower_images = len(list(subdir.glob("lower_*.png")))
                
                data_dirs.append({
                    'label': f"{subdir.name} ({upper_images} upper, {lower_images} lower images)",
                    'value': str(subdir)
                })
    
    # Sort by directory name
    data_dirs.sort(key=lambda x: x['label'])
    return data_dirs

class StereoCalibration:
    """Handle stereo calibration and 3D reconstruction"""
    
    def __init__(self, calibration_path: str, baseline_mm: float = 38.0):
        self.baseline_mm = baseline_mm
        self.load_calibration(calibration_path)
        
    def load_calibration(self, calibration_path: str):
        """Load stereo calibration data from npz file"""
        try:
            calib_data = np.load(calibration_path)
            
            self.camera_matrix_upper = calib_data['cameraMatrixUpper.npy']
            self.camera_matrix_lower = calib_data['cameraMatrixLower.npy']
            self.dist_coeffs_upper = calib_data['distCoeffsUpper.npy']
            self.dist_coeffs_lower = calib_data['distCoeffsLower.npy']
            self.R = calib_data['R.npy']  # Rotation matrix
            self.T = calib_data['T.npy']  # Translation vector
            self.E = calib_data['E.npy']  # Essential matrix
            
            # Compute projection matrices for triangulation
            self.P1 = np.hstack([self.camera_matrix_upper, np.zeros((3, 1))])
            self.P2 = np.hstack([self.camera_matrix_lower @ self.R, self.camera_matrix_lower @ self.T])
            
            print("✅ Loaded stereo calibration data")
            print(f"   Upper camera matrix: {self.camera_matrix_upper[0,0]:.1f} focal length")
            print(f"   Lower camera matrix: {self.camera_matrix_lower[0,0]:.1f} focal length")
            print(f"   Translation: {self.T.flatten()}")
            
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            raise
    
    def triangulate_points(self, upper_points: np.ndarray, lower_points: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from stereo correspondences
        
        Args:
            upper_points: Nx2 array of points in upper camera (rectified coordinates)
            lower_points: Nx2 array of points in lower camera (rectified coordinates)
            
        Returns:
            Nx3 array of 3D points in mm
        """
        if len(upper_points) != len(lower_points):
            raise ValueError("Number of upper and lower points must match")
        
        if len(upper_points) == 0:
            return np.array([]).reshape(0, 3)
        
        # Ensure points are in correct format
        upper_points = np.array(upper_points, dtype=np.float32).reshape(-1, 2)
        lower_points = np.array(lower_points, dtype=np.float32).reshape(-1, 2)
        
        # Load rectification data to get the proper projection matrices
        try:
            rect_data = np.load("/Users/vdausmann/swimming_in_salt/calibration/stereo_rectification.npz")
            P1 = rect_data['P1']  # Rectified projection matrix for upper camera
            P2 = rect_data['P2']  # Rectified projection matrix for lower camera
        except:
            # Fallback: use the projection matrices from calibration
            P1 = self.P1
            P2 = self.P2
        
        # Account for cropping and rotation from your detection pipeline
        try:
            roi_data = np.load("/Users/vdausmann/swimming_in_salt/calibration/roi_coordinates.npz")
            horizontal_lines_upper = roi_data['horizontal_lines_upper']
            horizontal_lines_lower = roi_data['horizontal_lines_lower']
            vertical_lines = roi_data['vertical_lines']
            
            left_bound = int(vertical_lines[0])
            upper_bound_top = int(horizontal_lines_upper[0])
            upper_bound_bottom = int(horizontal_lines_lower[0])
            right_bound = int(vertical_lines[1])
            
            # Adjust points back to full rectified image coordinates
            # Reverse the cropping
            upper_points_full = upper_points.copy()
            upper_points_full[:, 0] += left_bound + 120  # Add back left crop offset
            upper_points_full[:, 1] += upper_bound_top + 90  # Add back top crop offset
            
            lower_points_full = lower_points.copy()
            lower_points_full[:, 0] += left_bound + 120  # Add back left crop offset
            lower_points_full[:, 1] += upper_bound_bottom  # Add back top crop offset
            
            # Reverse the 90-degree counterclockwise rotation
            # Original rotation: (x,y) -> (y, height-x)
            # Reverse: (x,y) -> (height-y, x)
            original_height = 2028  # Adjust based on your camera resolution
            
            upper_points_unrotated = np.column_stack([
                upper_points_full[:, 1],  # Use y coordinate as new x
                original_height - upper_points_full[:, 0]  # Use (height - x) as new y
            ])

            lower_points_unrotated = np.column_stack([
                lower_points_full[:, 1],  # Use y coordinate as new x  
                original_height - lower_points_full[:, 0]  # Use (height - x) as new y
            ])
            
        except Exception as e:
            print(f"Warning: Could not load ROI data for coordinate adjustment: {e}")
            # Use points as-is if we can't load ROI data
            upper_points_unrotated = upper_points
            lower_points_unrotated = lower_points
        
        # Triangulate using the original camera coordinates
        points_4d_hom = cv2.triangulatePoints(
            P1, P2,
            upper_points_unrotated.T, lower_points_unrotated.T
        )
        
        # Convert from homogeneous coordinates
        points_3d = points_4d_hom[:3] / points_4d_hom[3]
        points_3d = points_3d.T  # Shape: (N, 3)
        
        # Now transform the 3D coordinates to match your desired track orientation
        # Since we rotated the images 90° counterclockwise, we need to rotate the 3D coordinates
        # 90° clockwise to get back to the original coordinate system
        
        # 90-degree clockwise rotation matrix around Z-axis:
        # [cos(-90)  -sin(-90)  0]   [0   1  0]
        # [sin(-90)   cos(-90)  0] = [-1  0  0] 
        # [   0         0       1]   [0   0  1]
        
        rotation_matrix = np.array([
            [0,  1, 0],
            [-1, 0, 0],
            [0,  0, 1]
        ])
        
        # Apply rotation to get coordinates in track orientation
        points_3d_rotated = points_3d @ rotation_matrix.T
        
        return points_3d_rotated

class TrackAnnotationTool:
    """Main class for the track annotation tool"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.load_data()
        self.current_frame = 0
        self.temp_dir = Path(tempfile.mkdtemp())  # For generated images
        
        # Load stereo calibration
        self.load_stereo_calibration()
        
        # Load or initialize stereo matches
        self.load_stereo_matches()
    
    def load_stereo_calibration(self):
        """Load stereo calibration data"""
        calibration_path = "/Users/vdausmann/oyster_project/planktracker3D/calibration/stereo_calibration.npz"
        try:
            self.stereo_calib = StereoCalibration(calibration_path, baseline_mm=38.0)
        except Exception as e:
            print(f"❌ Could not load stereo calibration: {e}")
            self.stereo_calib = None
    
    def load_stereo_matches(self):
        """Load or initialize stereo track matches"""
        stereo_csv_path = self.data_dir / "stereo_track_matches.csv"
        
        if stereo_csv_path.exists():
            try:
                self.stereo_matches = pd.read_csv(stereo_csv_path)
                print(f"✅ Loaded {len(self.stereo_matches)} stereo matches from {stereo_csv_path}")
            except Exception as e:
                print(f"❌ Error loading stereo matches: {e}")
                # Initialize empty DataFrame
                self.stereo_matches = pd.DataFrame(columns=[
                    'match_id', 'upper_track_id', 'lower_track_id', 
                    'created_at', 'created_frame'
                ])
        else:
            # Initialize empty DataFrame
            self.stereo_matches = pd.DataFrame(columns=[
                'match_id', 'upper_track_id', 'lower_track_id', 
                'created_at', 'created_frame'
            ])
            print("✅ Initialized empty stereo matches DataFrame")
    
    def update_3d_trajectories_csv(self) -> str:
        """Update the 3D trajectories CSV with all current matches"""
        if self.stereo_calib is None:
            print("❌ No stereo calibration available")
            return None
        
        all_3d_data = []
        
        for _, match in self.stereo_matches.iterrows():
            match_id = match['match_id']
            data_3d = self.get_3d_track_data(match_id)
            
            if "error" not in data_3d:
                points_3d = data_3d["points_3d"]
                frames = data_3d["frames"]
                timestamps = data_3d["timestamps"]
                
                for i, (frame, point) in enumerate(zip(frames, points_3d)):
                    all_3d_data.append({
                        'match_id': match_id,
                        'upper_track_id': data_3d['upper_track_id'],
                        'lower_track_id': data_3d['lower_track_id'],
                        'frame': frame,
                        'timestamp': timestamps[i] if i < len(timestamps) else None,
                        'x_mm': point[0],
                        'y_mm': point[1], 
                        'z_mm': point[2],
                        'point_index': i
                    })
        
        if all_3d_data:
            df_3d = pd.DataFrame(all_3d_data)
            csv_path = self.data_dir / "3d_trajectories.csv"
            df_3d.to_csv(csv_path, index=False)
            print(f"✅ Updated 3D trajectories CSV: {csv_path} ({len(all_3d_data)} points)")
            return str(csv_path)
        else:
            # Create empty CSV if no 3D data
            empty_df = pd.DataFrame(columns=[
                'match_id', 'upper_track_id', 'lower_track_id', 'frame', 
                'timestamp', 'x_mm', 'y_mm', 'z_mm', 'point_index'
            ])
            csv_path = self.data_dir / "3d_trajectories.csv"
            empty_df.to_csv(csv_path, index=False)
            print("✅ Created empty 3D trajectories CSV")
            return str(csv_path)

    # Keep the old method for manual export if needed
    def save_3d_trajectories_to_csv(self) -> str:
        """Manual export of 3D trajectories (same as update_3d_trajectories_csv)"""
        return self.update_3d_trajectories_csv()

    def load_data(self):
        """Load tracking data and images"""
        print(f"Loading data from {self.data_dir}")
        
        # Load track CSVs
        self.upper_tracks = pd.read_csv(self.data_dir / "upper_tracks.csv")
        self.lower_tracks = pd.read_csv(self.data_dir / "lower_tracks.csv")
        
        # Add display_track_id column if it doesn't exist
        self.upper_tracks = self._add_display_track_id_column(self.upper_tracks, "upper")
        self.lower_tracks = self._add_display_track_id_column(self.lower_tracks, "lower")
        
        # Save updated CSVs if new column was added
        self.upper_tracks.to_csv(self.data_dir / "upper_tracks.csv", index=False)
        self.lower_tracks.to_csv(self.data_dir / "lower_tracks.csv", index=False)
        
        # Find original images (not visualization ones)
        self.upper_images = sorted(list(self.data_dir.glob("upper_*.png")))
        self.lower_images = sorted(list(self.data_dir.glob("lower_*.png")))
        
        print(f"✅ Found {len(self.upper_images)} upper images, {len(self.lower_images)} lower images")
        
        # Convert tracks to SingleTrack objects for visualization
        self.upper_track_objects = self._convert_to_track_objects(self.upper_tracks)
        self.lower_track_objects = self._convert_to_track_objects(self.lower_tracks)
        
        # Determine frame range
        self.max_frame = max(
            self.upper_tracks['frame'].max() if len(self.upper_tracks) > 0 else 0,
            self.lower_tracks['frame'].max() if len(self.lower_tracks) > 0 else 0,
            len(self.upper_images) - 1,
            len(self.lower_images) - 1
        )
        
        # Generate colors for tracks
        self.track_colors = self._generate_track_colors()
        
        print(f"✅ Loaded {len(self.upper_tracks)} upper track points, {len(self.lower_tracks)} lower track points")
        print(f"✅ Frame range: 0 to {self.max_frame}")

    def _add_display_track_id_column(self, tracks_df: pd.DataFrame, camera: str) -> pd.DataFrame:
            """Add display_track_id column if it doesn't exist"""
            if 'display_track_id' not in tracks_df.columns:
                # Initialize display_track_id with original track_id
                tracks_df['display_track_id'] = tracks_df['track_id'].copy()
                print(f"✅ Added display_track_id column to {camera} tracks")
            else:
                print(f"✅ display_track_id column already exists in {camera} tracks")
            
            return tracks_df

    def get_next_display_track_id(self, camera: str) -> int:
        """Get the next available display track ID"""
        if camera == "upper":
            existing_ids = set(self.upper_tracks['display_track_id'].dropna())
        else:
            existing_ids = set(self.lower_tracks['display_track_id'].dropna())
        
        # Find the next available ID starting from 1
        next_id = 1
        while next_id in existing_ids:
            next_id += 1
        return next_id

    def merge_tracks(self, camera: str, track_id1: int, track_id2: int, current_frame: int) -> bool:
        """
        Merge two tracks: Replace track_id2 with track_id1 for all frames >= current_frame
        """
        try:
            if camera == "upper":
                tracks_df = self.upper_tracks
            else:
                tracks_df = self.lower_tracks
            
            # Find tracks to merge
            track1_mask = tracks_df['display_track_id'] == track_id1
            track2_mask = (tracks_df['display_track_id'] == track_id2) & (tracks_df['frame'] >= current_frame)
            
            if not track1_mask.any():
                print(f"❌ Track {track_id1} not found in {camera} camera")
                return False
            
            if not track2_mask.any():
                print(f"❌ Track {track_id2} not found in {camera} camera for frames >= {current_frame}")
                return False
            
            # Merge: assign track_id1 to all track_id2 points from current_frame onwards
            tracks_df.loc[track2_mask, 'display_track_id'] = track_id1
            
            # Update the dataframe
            if camera == "upper":
                self.upper_tracks = tracks_df
            else:
                self.lower_tracks = tracks_df
            
            # Save to CSV
            self._save_tracks_csv(camera)
            
            # Regenerate track objects and colors
            self._regenerate_track_data()
            
            print(f"✅ Merged tracks in {camera} camera: {track_id2} → {track_id1} (from frame {current_frame})")
            return True
            
        except Exception as e:
            print(f"❌ Error merging tracks: {e}")
            return False

    def split_track(self, camera: str, track_id: int, current_frame: int) -> bool:
        """
        Split a track: Create new track for all frames >= current_frame
        """
        try:
            if camera == "upper":
                tracks_df = self.upper_tracks
            else:
                tracks_df = self.lower_tracks
            
            # Find track to split
            track_mask = tracks_df['display_track_id'] == track_id
            split_mask = track_mask & (tracks_df['frame'] >= current_frame)
            
            if not track_mask.any():
                print(f"❌ Track {track_id} not found in {camera} camera")
                return False
            
            if not split_mask.any():
                print(f"❌ No points found for track {track_id} in frames >= {current_frame}")
                return False
            
            # Get new track ID
            new_track_id = self.get_next_display_track_id(camera)
            
            # Assign new ID to points from current_frame onwards
            tracks_df.loc[split_mask, 'display_track_id'] = new_track_id
            
            # Update the dataframe
            if camera == "upper":
                self.upper_tracks = tracks_df
            else:
                self.lower_tracks = tracks_df
            
            # Save to CSV
            self._save_tracks_csv(camera)
            
            # Regenerate track objects and colors
            self._regenerate_track_data()
            
            print(f"✅ Split track in {camera} camera: {track_id} → {track_id} + {new_track_id} (split at frame {current_frame})")
            return True
            
        except Exception as e:
            print(f"❌ Error splitting track: {e}")
            return False

    def hide_matched_tracks(self, upper_track_id: int, lower_track_id: int) -> bool:
        """
        Hide matched tracks by setting their display_track_id to NaN
        """
        try:
            # Hide upper track
            upper_mask = self.upper_tracks['display_track_id'] == upper_track_id
            self.upper_tracks.loc[upper_mask, 'display_track_id'] = np.nan
            
            # Hide lower track  
            lower_mask = self.lower_tracks['display_track_id'] == lower_track_id
            self.lower_tracks.loc[lower_mask, 'display_track_id'] = np.nan
            
            # Save both CSVs
            self._save_tracks_csv("upper")
            self._save_tracks_csv("lower")
            
            # Regenerate track objects and colors
            self._regenerate_track_data()
            
            print(f"✅ Hidden matched tracks: Upper {upper_track_id}, Lower {lower_track_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error hiding matched tracks: {e}")
            return False

    def unhide_tracks(self, upper_track_id: int, lower_track_id: int) -> bool:
        """
        Unhide tracks by restoring their display_track_id from track_id
        """
        try:
            # Unhide upper track
            upper_mask = self.upper_tracks['track_id'] == upper_track_id
            self.upper_tracks.loc[upper_mask, 'display_track_id'] = upper_track_id
            
            # Unhide lower track
            lower_mask = self.lower_tracks['track_id'] == lower_track_id
            self.lower_tracks.loc[lower_mask, 'display_track_id'] = lower_track_id
            
            # Save both CSVs
            self._save_tracks_csv("upper")
            self._save_tracks_csv("lower")
            
            # Regenerate track objects and colors
            self._regenerate_track_data()
            
            print(f"✅ Unhidden tracks: Upper {upper_track_id}, Lower {lower_track_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error unhiding tracks: {e}")
            return False

    def _save_tracks_csv(self, camera: str):
        """Save tracks CSV file"""
        if camera == "upper":
            csv_path = self.data_dir / "upper_tracks.csv"
            self.upper_tracks.to_csv(csv_path, index=False)
        else:
            csv_path = self.data_dir / "lower_tracks.csv"
            self.lower_tracks.to_csv(csv_path, index=False)
        
        print(f"💾 Saved {camera} tracks to {csv_path}")

    def _regenerate_track_data(self):
        """Regenerate track objects and colors after track editing"""
        self.upper_track_objects = self._convert_to_track_objects(self.upper_tracks)
        self.lower_track_objects = self._convert_to_track_objects(self.lower_tracks)
        self.track_colors = self._generate_track_colors()

    def _convert_to_track_objects(self, tracks_df: pd.DataFrame) -> List[SingleTrack]:
        """Convert DataFrame tracks to SingleTrack objects using display_track_id"""
        track_objects = []
        
        # Only consider tracks that are visible (display_track_id is not NaN)
        visible_tracks = tracks_df[tracks_df['display_track_id'].notna()]
        
        for display_track_id in visible_tracks['display_track_id'].unique():
            track_data = visible_tracks[visible_tracks['display_track_id'] == display_track_id].sort_values('frame')
            
            track = SingleTrack(
                positions=[(int(row['x']), int(row['y'])) for _, row in track_data.iterrows()],
                areas=[row['area'] for _, row in track_data.iterrows()],
                frame_indices=[row['frame'] for _, row in track_data.iterrows()],
                motion_pattern=track_data['motion_pattern'].iloc[0] if 'motion_pattern' in track_data.columns else 'unknown'
            )
            track_objects.append(track)
        
        return track_objects
    
    def generate_frame_image(self, frame_idx: int, camera: str, selected_track: Optional[int] = None) -> Optional[str]:
        """Generate a single frame image with track visualization"""
        try:
            # Get the appropriate data
            if camera == "upper":
                image_files = self.upper_images
                track_objects = self.upper_track_objects
                tracks_df = self.upper_tracks
            else:
                image_files = self.lower_images
                track_objects = self.lower_track_objects
                tracks_df = self.lower_tracks
            
            if frame_idx >= len(image_files):
                return None
                
            # Load the base image
            img_path = image_files[frame_idx]
            img = cv2.imread(str(img_path))
            if img is None:
                return None
            
            # Generate colors for tracks
            colors = self.track_colors
            
            # Filter tracks if a specific track is selected
            tracks_to_show = track_objects
            if selected_track is not None:
                # Find the track object that corresponds to the selected track ID
                selected_track_obj = None
                for tid, track in enumerate(track_objects):
                    # Check if this track object contains the selected track ID
                    track_data = tracks_df[tracks_df['display_track_id'] == selected_track]
                    if len(track_data) > 0:
                        # Match by comparing some track data
                        track_frames = set(track.frame_indices)
                        data_frames = set(track_data['frame'].tolist())
                        if track_frames == data_frames:
                            selected_track_obj = track
                            colors = [colors[tid]]  # Use the original color
                            break
                
                if selected_track_obj:
                    tracks_to_show = [selected_track_obj]
                else:
                    tracks_to_show = []
            
            # Draw tracks on the image with persistence
            for tid, track in enumerate(tracks_to_show):
                color = colors[tid if len(colors) > tid else 0]
                # Ensure color is a proper BGR tuple
                if isinstance(color, str):
                    # Convert from RGB string to BGR tuple if needed
                    color = (0, 255, 0)  # Default green
                elif len(color) > 3:
                    color = color[:3]  # Take only first 3 values (BGR)
                
                # Find the last position before or at current frame
                last_position = None
                last_frame = -1
                track_id_for_label = selected_track if selected_track is not None else tid
                
                # Find the most recent position up to current frame
                for i, frame in enumerate(track.frame_indices):
                    if frame <= frame_idx:
                        last_position = track.positions[i]
                        last_frame = frame
                    else:
                        break  # frames are sorted, so we can stop here
                
                # Draw track if we have a recent position (within 5 frames)
                if last_position is not None and (frame_idx - last_frame) <= 5:
                    # Calculate alpha based on how old the position is
                    frames_since_last = frame_idx - last_frame
                    if frames_since_last == 0:
                        # Current frame - full opacity
                        alpha = 1.0
                        marker_size = 6
                    else:
                        # Fade out over 5 frames
                        alpha = max(0.2, 1.0 - (frames_since_last / 5.0))
                        marker_size = max(3, 6 - frames_since_last)
                    
                    # Apply alpha to color
                    faded_color = tuple(int(c * alpha) for c in color)
                    
                    # Draw current/last position
                    cv2.circle(img, last_position, marker_size, faded_color, -1)
                    
                    # Add track ID label (only if not too faded)
                    if alpha > 0.3:
                        label_alpha = min(1.0, alpha * 1.5)
                        label_color = tuple(int(c * label_alpha) for c in color)
                        cv2.putText(img, f"T{track_id_for_label}", 
                                (last_position[0]+8, last_position[1]-8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
                    
                    # Draw trajectory up to current frame (with full opacity for the path)
                    for j in range(1, len(track.positions)):
                        if track.frame_indices[j] > frame_idx:
                            break
                        if track.frame_indices[j-1] <= frame_idx:
                            pt1 = track.positions[j-1]
                            pt2 = track.positions[j]
                            cv2.line(img, pt1, pt2, color, 2)
            
            # Add frame number
            cv2.putText(img, f"Frame: {frame_idx}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if selected_track is not None:
                cv2.putText(img, f"Selected Track: {selected_track}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Save to temp file
            temp_path = self.temp_dir / f"{camera}_frame_{frame_idx}_track_{selected_track or 'all'}.png"
            cv2.imwrite(str(temp_path), img)
            
            return str(temp_path)
            
        except Exception as e:
            print(f"Error generating frame image: {e}")
            return None

    def _generate_track_colors(self) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for each track as BGR tuples for OpenCV"""
        all_track_ids = set(self.upper_tracks['track_id'].unique()) | set(self.lower_tracks['track_id'].unique())
        colors = []
        
        for i, track_id in enumerate(sorted(all_track_ids)):
            # Use HSV color space for better distribution
            hue = (i * 137.508) % 360  # Golden angle for better distribution
            saturation = 0.8
            value = 0.9
            
            rgb = colorsys.hsv_to_rgb(hue/360, saturation, value)
            # Convert to BGR for OpenCV (and ensure integers)
            bgr = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
            colors.append(bgr)
    
        return colors
    
    def get_frame_data(self, frame_idx: int) -> Tuple[Dict, Dict]:
        """Get track data for a specific frame"""
        upper_frame = self.upper_tracks[self.upper_tracks['frame'] == frame_idx]
        lower_frame = self.lower_tracks[self.lower_tracks['frame'] == frame_idx]
        
        upper_data = {}
        for _, row in upper_frame.iterrows():
            track_id = row['track_id']
            upper_data[track_id] = {
                'x': row['x'], 'y': row['y'], 'area': row['area'],
                'motion_pattern': row.get('motion_pattern', 'unknown')
            }
        
        lower_data = {}
        for _, row in lower_frame.iterrows():
            track_id = row['track_id']
            lower_data[track_id] = {
                'x': row['x'], 'y': row['y'], 'area': row['area'],
                'motion_pattern': row.get('motion_pattern', 'unknown')
            }
        
        return upper_data, lower_data
    
    def get_frame_data(self, frame_idx: int) -> Tuple[Dict, Dict]:
        """Get track data for a specific frame using display_track_id"""
        upper_frame = self.upper_tracks[
            (self.upper_tracks['frame'] == frame_idx) & 
            (self.upper_tracks['display_track_id'].notna())
        ]
        lower_frame = self.lower_tracks[
            (self.lower_tracks['frame'] == frame_idx) & 
            (self.lower_tracks['display_track_id'].notna())
        ]
        
        upper_data = {}
        for _, row in upper_frame.iterrows():
            display_track_id = int(row['display_track_id'])
            upper_data[display_track_id] = {
                'x': row['x'], 'y': row['y'], 'area': row['area'],
                'motion_pattern': row.get('motion_pattern', 'unknown'),
                'original_track_id': row['track_id']
            }
        
        lower_data = {}
        for _, row in lower_frame.iterrows():
            display_track_id = int(row['display_track_id'])
            lower_data[display_track_id] = {
                'x': row['x'], 'y': row['y'], 'area': row['area'],
                'motion_pattern': row.get('motion_pattern', 'unknown'),
                'original_track_id': row['track_id']
            }
        
        return upper_data, lower_data

    def get_track_info(self, display_track_id: int, camera: str = None) -> Dict:
        """Get detailed information about a specific track using display_track_id"""
        if display_track_id is None:
            return {}
        
        if camera == "upper":
            track_data = self.upper_tracks[
                (self.upper_tracks['display_track_id'] == display_track_id) &
                (self.upper_tracks['display_track_id'].notna())
            ]
        elif camera == "lower":
            track_data = self.lower_tracks[
                (self.lower_tracks['display_track_id'] == display_track_id) &
                (self.lower_tracks['display_track_id'].notna())
            ]
        else:
            # Get from both cameras
            upper_track = self.upper_tracks[
                (self.upper_tracks['display_track_id'] == display_track_id) &
                (self.upper_tracks['display_track_id'].notna())
            ]
            lower_track = self.lower_tracks[
                (self.lower_tracks['display_track_id'] == display_track_id) &
                (self.lower_tracks['display_track_id'].notna())
            ]
            track_data = pd.concat([upper_track, lower_track])
        
        if len(track_data) == 0:
            return {}
        
        # Sort by frame
        track_data = track_data.sort_values('frame')
        
        # Calculate velocities
        velocities = []
        if len(track_data) > 1:
            for i in range(1, len(track_data)):
                prev_row = track_data.iloc[i-1]
                curr_row = track_data.iloc[i]
                
                # Calculate distance in pixels
                dx = curr_row['x'] - prev_row['x']
                dy = curr_row['y'] - prev_row['y']
                distance = np.sqrt(dx**2 + dy**2)
                
                # Calculate time difference (assuming ~30 fps, so 1 frame = 1/30 second)
                frame_diff = curr_row['frame'] - prev_row['frame']
                time_diff = frame_diff / 30.0  # Convert frames to seconds
                
                if time_diff > 0:
                    velocity = distance / time_diff  # pixels per second
                    velocities.append(velocity)
        
        # Get last position
        last_row = track_data.iloc[-1]
        last_position = (last_row['x'], last_row['y'])
        
        # Calculate average and last velocity
        avg_velocity = np.mean(velocities) if velocities else 0
        last_velocity = velocities[-1] if velocities else 0
        
        # Get original track IDs that make up this display track
        original_track_ids = list(track_data['track_id'].unique())
        
        info = {
            'display_track_id': display_track_id,
            'original_track_ids': original_track_ids,
            'total_points': len(track_data),
            'frame_range': f"{track_data['frame'].min()} - {track_data['frame'].max()}",
            'last_position': last_position,
            'avg_velocity': avg_velocity,
            'last_velocity': last_velocity,
            'camera': camera
        }
        
        # Add camera-specific info if needed
        if camera == "upper" or camera is None:
            upper_track = self.upper_tracks[
                (self.upper_tracks['display_track_id'] == display_track_id) &
                (self.upper_tracks['display_track_id'].notna())
            ]
            if len(upper_track) > 0:
                info['upper_points'] = len(upper_track)
                info['upper_frame_range'] = f"{upper_track['frame'].min()} - {upper_track['frame'].max()}"
                info['upper_avg_area'] = upper_track['area'].mean()
                info['upper_motion_pattern'] = upper_track['motion_pattern'].iloc[0] if 'motion_pattern' in upper_track.columns else 'unknown'
        
        if camera == "lower" or camera is None:
            lower_track = self.lower_tracks[
                (self.lower_tracks['display_track_id'] == display_track_id) &
                (self.lower_tracks['display_track_id'].notna())
            ]
            if len(lower_track) > 0:
                info['lower_points'] = len(lower_track)
                info['lower_frame_range'] = f"{lower_track['frame'].min()} - {lower_track['frame'].max()}"
                info['lower_avg_area'] = lower_track['area'].mean()
                info['lower_motion_pattern'] = lower_track['motion_pattern'].iloc[0] if 'motion_pattern' in lower_track.columns else 'unknown'
        
        return info

    
    def create_stereo_match(self, upper_track_id: int, lower_track_id: int, current_frame: int) -> bool:
        """Create a new stereo track match and automatically save 3D data, then hide the tracks"""
        try:
            # Check if either track is already matched
            existing_upper = self.stereo_matches[self.stereo_matches['upper_track_id'] == upper_track_id]
            existing_lower = self.stereo_matches[self.stereo_matches['lower_track_id'] == lower_track_id]
            
            if len(existing_upper) > 0:
                print(f"Upper track {upper_track_id} is already matched to lower track {existing_upper.iloc[0]['lower_track_id']}")
                return False
            
            if len(existing_lower) > 0:
                print(f"Lower track {lower_track_id} is already matched to upper track {existing_lower.iloc[0]['upper_track_id']}")
                return False
            
            # Generate new match ID
            if len(self.stereo_matches) == 0:
                match_id = 1
            else:
                match_id = self.stereo_matches['match_id'].max() + 1
            
            # Create new match
            new_match = {
                'match_id': match_id,
                'upper_track_id': upper_track_id,
                'lower_track_id': lower_track_id,
                'created_at': datetime.now().isoformat(),
                'created_frame': current_frame
            }
            
            # Add to DataFrame
            self.stereo_matches = pd.concat([
                self.stereo_matches, 
                pd.DataFrame([new_match])
            ], ignore_index=True)
            
            # Save to CSV
            stereo_csv_path = self.data_dir / "stereo_track_matches.csv"
            self.stereo_matches.to_csv(stereo_csv_path, index=False)
            
            # Hide the matched tracks
            self.hide_matched_tracks(upper_track_id, lower_track_id)
            
            print(f"✅ Created stereo match {match_id}: Upper {upper_track_id} ↔ Lower {lower_track_id}")
            print(f"✅ Hidden matched tracks from display")
            
            # Automatically update 3D trajectories CSV
            self.update_3d_trajectories_csv()
            
            return True
            
        except Exception as e:
            print(f"Error creating stereo match: {e}")
            return False
    
    def get_stereo_match_info(self, upper_track_id: int = None, lower_track_id: int = None) -> Dict:
        """Get information about stereo matches for given track(s)"""
        info = {}
        
        if upper_track_id is not None:
            match = self.stereo_matches[self.stereo_matches['upper_track_id'] == upper_track_id]
            if len(match) > 0:
                info['upper_matched'] = True
                info['upper_match_id'] = match.iloc[0]['match_id']
                info['upper_paired_with'] = match.iloc[0]['lower_track_id']
            else:
                info['upper_matched'] = False
        
        if lower_track_id is not None:
            match = self.stereo_matches[self.stereo_matches['lower_track_id'] == lower_track_id]
            if len(match) > 0:
                info['lower_matched'] = True
                info['lower_match_id'] = match.iloc[0]['match_id']
                info['lower_paired_with'] = match.iloc[0]['upper_track_id']
            else:
                info['lower_matched'] = False
                
        return info
    
    def extract_timestamp_from_filename(self, filename: str) -> Optional[str]:
        """Extract timestamp from filename in format upper/lower_YYYYMMDD_HHMMSS_microseconds.png"""
        match = re.search(r'(?:upper|lower)_(\d{8})_(\d{6})_(\d+)\.png', filename)
        if match:
            date, time, microseconds = match.groups()
            return f"{date}_{time}_{microseconds}"
        return None

    def calculate_frame_timestamps(self, frames: List[int], camera: str = "upper") -> List[Optional[str]]:
        """Calculate timestamps for given frames based on image filenames"""
        timestamps = []
        image_files = self.upper_images if camera == "upper" else self.lower_images
        
        for frame in frames:
            if frame < len(image_files):
                timestamp = self.extract_timestamp_from_filename(os.path.basename(image_files[frame]))
                timestamps.append(timestamp)
            else:
                timestamps.append(None)
        
        return timestamps

    def calculate_time_differences(self, timestamps: List[str]) -> List[float]:
        """Calculate time differences between consecutive timestamps in seconds"""
        time_diffs = []
        
        for i in range(1, len(timestamps)):
            if timestamps[i] is None or timestamps[i-1] is None:
                time_diffs.append(None)
                continue
                
            try:
                # Parse timestamps: YYYYMMDD_HHMMSS_microseconds
                prev_parts = timestamps[i-1].split('_')
                curr_parts = timestamps[i].split('_')
                
                if len(prev_parts) != 3 or len(curr_parts) != 3:
                    time_diffs.append(None)
                    continue
                
                # Convert to datetime
                prev_dt = datetime.strptime(f"{prev_parts[0]}_{prev_parts[1]}", "%Y%m%d_%H%M%S")
                curr_dt = datetime.strptime(f"{curr_parts[0]}_{curr_parts[1]}", "%Y%m%d_%H%M%S")
                
                # Add microseconds
                prev_dt = prev_dt.replace(microsecond=int(prev_parts[2]))
                curr_dt = curr_dt.replace(microsecond=int(curr_parts[2]))
                
                # Calculate difference in seconds
                time_diff = (curr_dt - prev_dt).total_seconds()
                time_diffs.append(time_diff)
                
            except Exception as e:
                print(f"Error calculating time difference: {e}")
                time_diffs.append(None)
        
        return time_diffs

    def get_3d_track_data(self, match_id: int) -> Dict:
        """Get 3D trajectory data for a stereo matched track pair with enhanced statistics"""
        if self.stereo_calib is None:
            return {"error": "No stereo calibration available"}
        
        # Find the match
        match = self.stereo_matches[self.stereo_matches['match_id'] == match_id]
        if len(match) == 0:
            return {"error": "Match not found"}
        
        upper_track_id = match.iloc[0]['upper_track_id']
        lower_track_id = match.iloc[0]['lower_track_id']
        
        # Get track data
        upper_track = self.upper_tracks[self.upper_tracks['track_id'] == upper_track_id].sort_values('frame')
        lower_track = self.lower_tracks[self.lower_tracks['track_id'] == lower_track_id].sort_values('frame')
        
        if len(upper_track) == 0 or len(lower_track) == 0:
            return {"error": "Track data not found"}
        
        # Find common frames
        upper_frames = set(upper_track['frame'])
        lower_frames = set(lower_track['frame'])
        common_frames = sorted(upper_frames & lower_frames)
        
        if len(common_frames) == 0:
            return {"error": "No common frames between tracks"}
        
        # Extract corresponding points
        upper_points = []
        lower_points = []
        frames = []
        
        for frame in common_frames:
            upper_point = upper_track[upper_track['frame'] == frame]
            lower_point = lower_track[lower_track['frame'] == frame]
            
            if len(upper_point) == 1 and len(lower_point) == 1:
                upper_points.append([upper_point.iloc[0]['x'], upper_point.iloc[0]['y']])
                lower_points.append([lower_point.iloc[0]['x'], lower_point.iloc[0]['y']])
                frames.append(frame)
        
        if len(upper_points) == 0:
            return {"error": "No valid point correspondences"}
        
        # Triangulate 3D points
        try:
            points_3d = self.stereo_calib.triangulate_points(
                np.array(upper_points), 
                np.array(lower_points)
            )
            
            # Calculate timestamps for frames
            timestamps = self.calculate_frame_timestamps(frames, "upper")
            time_diffs = self.calculate_time_differences(timestamps)
            
            # Calculate 3D statistics
            stats = self.calculate_3d_statistics(points_3d, timestamps, time_diffs)
            
            return {
                "match_id": match_id,
                "upper_track_id": upper_track_id,
                "lower_track_id": lower_track_id,
                "frames": frames,
                "timestamps": timestamps,
                "points_3d": points_3d,
                "upper_points": upper_points,
                "lower_points": lower_points,
                "num_points": len(points_3d),
                "statistics": stats
            }
            
        except Exception as e:
            return {"error": f"Triangulation failed: {e}"}

    def calculate_3d_statistics(self, points_3d: np.ndarray, timestamps: List[str], time_diffs: List[float]) -> Dict:
        """Calculate comprehensive 3D trajectory statistics"""
        if len(points_3d) < 2:
            return {
                "total_distance_mm": 0.0,
                "avg_speed_xyz": {"x": 0.0, "y": 0.0, "z": 0.0},
                "instantaneous_speeds": [],
                "frame_distances": [],
                "valid_time_intervals": 0
            }
        
        # Calculate distances between consecutive points
        frame_distances = []
        instantaneous_speeds = []
        velocity_components = {"x": [], "y": [], "z": []}
        
        for i in range(1, len(points_3d)):
            # 3D distance between consecutive points
            dx = points_3d[i, 0] - points_3d[i-1, 0]
            dy = points_3d[i, 1] - points_3d[i-1, 1]
            dz = points_3d[i, 2] - points_3d[i-1, 2]
            
            distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
            frame_distances.append(distance_3d)
            
            # Calculate velocity if we have valid time difference
            if i-1 < len(time_diffs) and time_diffs[i-1] is not None and time_diffs[i-1] > 0:
                dt = time_diffs[i-1]
                
                # Velocity components (mm/s)
                vx = dx / dt
                vy = dy / dt
                vz = dz / dt
                
                velocity_components["x"].append(vx)
                velocity_components["y"].append(vy)
                velocity_components["z"].append(vz)
                
                # Total instantaneous speed
                speed = distance_3d / dt
                instantaneous_speeds.append(speed)
            else:
                instantaneous_speeds.append(None)
        
        # Calculate averages
        total_distance = np.sum(frame_distances)
        
        avg_speed_xyz = {
            "x": np.mean(velocity_components["x"]) if velocity_components["x"] else 0.0,
            "y": np.mean(velocity_components["y"]) if velocity_components["y"] else 0.0,
            "z": np.mean(velocity_components["z"]) if velocity_components["z"] else 0.0
        }
        
        # Calculate overall statistics
        valid_speeds = [s for s in instantaneous_speeds if s is not None]
        valid_time_intervals = len(valid_speeds)
        
        return {
            "total_distance_mm": total_distance,
            "avg_speed_xyz": avg_speed_xyz,
            "avg_speed_total": np.mean(valid_speeds) if valid_speeds else 0.0,
            "max_speed": np.max(valid_speeds) if valid_speeds else 0.0,
            "min_speed": np.min(valid_speeds) if valid_speeds else 0.0,
            "instantaneous_speeds": instantaneous_speeds,
            "frame_distances": frame_distances,
            "velocity_components": velocity_components,
            "valid_time_intervals": valid_time_intervals,
            "total_time_seconds": np.sum([t for t in time_diffs if t is not None]) if time_diffs else 0.0
        }
    
    def get_stereo_matches_table_data(self) -> List[Dict]:
        """Get data for the stereo matches table"""
        table_data = []
        
        for _, match in self.stereo_matches.iterrows():
            match_id = match['match_id']
            upper_track_id = match['upper_track_id']
            lower_track_id = match['lower_track_id']
            created_frame = match['created_frame']
            
            # Get basic track statistics
            upper_track = self.upper_tracks[self.upper_tracks['track_id'] == upper_track_id]
            lower_track = self.lower_tracks[self.lower_tracks['track_id'] == lower_track_id]
            
            # Calculate common frames
            upper_frames = set(upper_track['frame']) if len(upper_track) > 0 else set()
            lower_frames = set(lower_track['frame']) if len(lower_track) > 0 else set()
            common_frames = upper_frames & lower_frames
            
            # Get 3D data if stereo calibration is available
            three_d_status = "❌ No calibration"
            total_distance = 0.0
            avg_speed = 0.0
            
            if self.stereo_calib is not None:
                try:
                    data_3d = self.get_3d_track_data(match_id)
                    if "error" not in data_3d and "statistics" in data_3d:
                        stats = data_3d["statistics"]
                        three_d_status = "✅ Available"
                        total_distance = stats['total_distance_mm']
                        avg_speed = stats['avg_speed_total']
                    else:
                        three_d_status = "⚠️ Error"
                except:
                    three_d_status = "⚠️ Error"
            
            table_data.append({
                'match_id': match_id,
                'upper_track': upper_track_id,
                'lower_track': lower_track_id,
                'common_frames': len(common_frames),
                'frame_range': f"{min(common_frames)} - {max(common_frames)}" if common_frames else "No common frames",
                'created_frame': created_frame,
                '3d_status': three_d_status,
                'distance_mm': f"{total_distance:.1f}",
                'avg_speed_mm_s': f"{avg_speed:.1f}",
                'created_at': match['created_at'][:19] if 'created_at' in match else 'Unknown'
            })
        
        return table_data

def create_app() -> dash.Dash:
    """Create and configure the Dash application"""
    
    # Create Dash app
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
    app.title = "Track Annotation Tool"
    
    # Get available data directories
    available_dirs = get_available_data_directories(BASE_DATA_DIR)
    
    # Cache for tool instances to prevent recreating them constantly
    _tool_cache = {}
    
    # Helper function to get tool instance with caching
    def get_tool_instance(tool_data):
        if not tool_data or not tool_data.get("initialized"):
            return None
        
        data_dir = tool_data["data_dir"]
        
        # Check if we already have this tool instance cached
        if data_dir in _tool_cache:
            return _tool_cache[data_dir]
        
        # Create new tool instance and cache it
        try:
            tool = TrackAnnotationTool(data_dir)
            _tool_cache[data_dir] = tool
            return tool
        except Exception as e:
            print(f"Error creating tool instance: {e}")
            return None
    
    # Function to clear tool cache when needed
    def clear_tool_cache():
        _tool_cache.clear()

    # Define the layout
    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("🎯 Track Annotation & Validation Tool", className="text-center mb-4"),
                html.Hr()
            ])
        ]),
        
        # Data Directory Selection
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📁 Select Data Directory"),
                    dbc.CardBody([
                        dcc.Dropdown(
                            id="data-dir-dropdown",
                            options=available_dirs,
                            placeholder="Select a data directory...",
                            className="mb-2"
                        ),
                        dcc.Loading(
                            id="loading-data-dir",
                            type="default",
                            children=[
                                html.Div(id="data-dir-status", className="text-muted")
                            ]
                        )
                    ])
                ])
            ])
        ], className="mb-3"),
        
        # Main content area (initially hidden)
        html.Div(id="main-content", style={"display": "none"}, children=[
            # Control panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📹 Frame Control"),
                        dbc.CardBody([
                            html.Div([
                                dbc.Button("⏮️", id="btn-first-frame", size="sm", className="me-1"),
                                dbc.Button("⏪", id="btn-prev-frame", size="sm", className="me-1"),
                                dbc.Button("⏩", id="btn-next-frame", size="sm", className="me-1"),
                                dbc.Button("⏭️", id="btn-last-frame", size="sm", className="me-1"),
                            ], className="mb-2"),
                            dcc.Slider(
                                id="frame-slider",
                                min=0,
                                max=100,  # Will be updated when data is loaded
                                value=0,
                                marks={},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Div(id="current-frame-display", className="text-center mt-2")
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔧 Track Editing"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Upper Track ID:", className="form-label"),
                                    dbc.Input(id="merge-upper-track1", type="number", placeholder="Track 1", size="sm"),
                                    dbc.Input(id="merge-upper-track2", type="number", placeholder="Track 2", size="sm", className="mt-1"),
                                    dbc.Button("🔗 Merge Upper", id="btn-merge-upper", color="warning", size="sm", className="w-100 mt-1"),
                                    dbc.Button("✂️ Split Upper", id="btn-split-upper", color="info", size="sm", className="w-100 mt-1"),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Lower Track ID:", className="form-label"),
                                    dbc.Input(id="merge-lower-track1", type="number", placeholder="Track 1", size="sm"),
                                    dbc.Input(id="merge-lower-track2", type="number", placeholder="Track 2", size="sm", className="mt-1"),
                                    dbc.Button("🔗 Merge Lower", id="btn-merge-lower", color="warning", size="sm", className="w-100 mt-1"),
                                    dbc.Button("✂️ Split Lower", id="btn-split-lower", color="info", size="sm", className="w-100 mt-1"),
                                ], width=6)
                            ])
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎛️ Actions"),
                        dbc.CardBody([
                            dbc.Button("Clear Upper Selection", id="btn-clear-upper", color="secondary", size="sm", className="mb-1 w-100"),
                            dbc.Button("Clear Lower Selection", id="btn-clear-lower", color="secondary", size="sm", className="mb-1 w-100"),
                            dbc.Button("Clear All Selections", id="btn-clear-all", color="secondary", size="sm", className="mb-1 w-100"),
                            html.Hr(),
                            dbc.Button(
                                "🔗 Create Stereo Match", 
                                id="btn-create-stereo-match",
                                color="success", 
                                size="sm", 
                                className="w-100",
                                disabled=True
                            ),
                            dbc.Button(
                                "🎯 View 3D Trajectory", 
                                id="btn-view-3d",
                                color="primary", 
                                size="sm", 
                                className="w-100 mt-1",
                                disabled=True
                            ),
                        ])
                    ])
                ], width=4)
            ], className="mb-3"),
            
            # Image display
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📷 Upper Camera"),
                        dbc.CardBody([
                            dcc.Graph(
                                id="upper-image", 
                                style={"height": "938px"},
                                config={'displayModeBar': False, "scrollZoom": True, "doubleClick": "reset"}
                            )
                        ], style={"padding": "0"})
                    ])
                ], width=6),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📷 Lower Camera"),
                        dbc.CardBody([
                            dcc.Graph(
                                id="lower-image", 
                                style={"height": "938px"},
                                config={'displayModeBar': False, "scrollZoom": True, "doubleClick": "reset"}
                            )
                        ], style={"padding": "0"})
                    ])
                ], width=6)
            ], className="mb-3"),
            
            # Track information panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Track Information"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Upper Camera Selection"),
                                    html.Div(id="upper-track-info-display", style={"minHeight": "120px"})
                                ], width=6),
                                dbc.Col([
                                    html.H6("Lower Camera Selection"),
                                    html.Div(id="lower-track-info-display", style={"minHeight": "120px"})
                                ], width=6)
                            ])
                        ])
                    ])
                ])
            ], className="mb-3"),
            
            # Stereo matching panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🎯 Stereo Matching"),
                        dbc.CardBody([
                            html.Div(id="stereo-match-info", style={"minHeight": "60px"})
                        ])
                    ])
                ])
            ], className="mb-3"),
            
            # Stereo Matches Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.H5("🔗 Stereo Track Matches", className="mb-0"),
                            dbc.Button("🔄 Refresh", id="btn-refresh-matches", size="sm", color="outline-secondary", className="float-end")
                        ]),
                        dbc.CardBody([
                            html.Div(id="stereo-matches-table-container", style={"maxHeight": "400px", "overflowY": "auto"})
                        ])
                    ])
                ])
            ], className="mb-3"),

            # 3D Viewer panel (initially hidden)
            dbc.Row([
                dbc.Col([
                    dbc.Collapse([
                        dbc.Card([
                            dbc.CardHeader("🎯 3D Trajectory Viewer"),
                            dbc.CardBody([
                                dcc.Graph(
                                    id="3d-trajectory-plot",
                                    style={"height": "600px"},
                                    config={'displayModeBar': True}
                                ),
                                html.Div(id="3d-info-display", className="mt-2")
                            ])
                        ])
                    ], id="3d-viewer-collapse", is_open=False)
                ])
            ], className="mb-3"),
        ]),
        
        # Hidden components for state
        dcc.Store(id="tool-store"),  # Store the tool instance data
        dcc.Store(id="upper-selected-track-store"),
        dcc.Store(id="lower-selected-track-store"),
        dcc.Store(id="current-3d-match-store"),
        dcc.Store(id="selected-stereo-match-store"),
        
    ], fluid=True)
    
    # Callback to initialize the tool when a directory is selected
    @app.callback(
        [Output("tool-store", "data"),
         Output("main-content", "style"),
         Output("data-dir-status", "children"),
         Output("frame-slider", "max"),
         Output("frame-slider", "marks"),
         Output("frame-slider", "value")],
        [Input("data-dir-dropdown", "value")]
    )
    def initialize_tool(selected_dir):
        if not selected_dir:
            return None, {"display": "none"}, "Please select a data directory.", 100, {}, 0
        
        try:
            # Clear cache when switching directories
            clear_tool_cache()
            
            # Initialize the tool
            tool = TrackAnnotationTool(selected_dir)
            
            # Cache the tool immediately
            _tool_cache[selected_dir] = tool
            
            # Create marks for slider
            marks = {i: str(i) for i in range(0, tool.max_frame + 1, max(1, tool.max_frame // 10))}
            
            # Store tool data (we'll store minimal info and recreate as needed)
            tool_data = {
                "data_dir": selected_dir,
                "max_frame": tool.max_frame,
                "initialized": True
            }
            
            status_msg = dbc.Alert(
                f"✅ Loaded data from {Path(selected_dir).name} - {tool.max_frame + 1} frames",
                color="success"
            )
            
            return tool_data, {"display": "block"}, status_msg, tool.max_frame, marks, 0
            
        except Exception as e:
            error_msg = dbc.Alert(
                f"❌ Error loading data: {e}",
                color="danger"
            )
            return None, {"display": "none"}, error_msg, 100, {}, 0

    # Update all existing callbacks to use the tool from store
    @app.callback(
        Output("frame-slider", "value", allow_duplicate=True),  # Add allow_duplicate=True here
        [Input("btn-first-frame", "n_clicks"),
         Input("btn-prev-frame", "n_clicks"),
         Input("btn-next-frame", "n_clicks"),
         Input("btn-last-frame", "n_clicks")],
        [State("frame-slider", "value"),
         State("tool-store", "data")],
        prevent_initial_call=True  # Add this to prevent conflicts
    )
    def navigate_frames(first, prev, next_btn, last, current_frame, tool_data):
        if not tool_data:
            return 0
            
        if not callback_context.triggered:
            return current_frame
        
        button_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        max_frame = tool_data.get("max_frame", 0)
        
        if button_id == "btn-first-frame":
            return 0
        elif button_id == "btn-prev-frame":
            return max(0, current_frame - 1)
        elif button_id == "btn-next-frame":
            return min(max_frame, current_frame + 1)
        elif button_id == "btn-last-frame":
            return max_frame
        
        return current_frame

    # Callback for merging tracks
    @app.callback(
        [Output("upper-image", "figure", allow_duplicate=True),
         Output("lower-image", "figure", allow_duplicate=True)],
        [Input("btn-merge-upper", "n_clicks"),
         Input("btn-merge-lower", "n_clicks")],
        [State("merge-upper-track1", "value"),
         State("merge-upper-track2", "value"),
         State("merge-lower-track1", "value"),
         State("merge-lower-track2", "value"),
         State("frame-slider", "value"),
         State("tool-store", "data"),
         State("upper-selected-track-store", "data"),
         State("lower-selected-track-store", "data")],
        prevent_initial_call=True
    )
    def handle_merge_tracks(merge_upper_clicks, merge_lower_clicks,
                           upper_track1, upper_track2, lower_track1, lower_track2,
                           current_frame, tool_data, upper_selected, lower_selected):
        if not callback_context.triggered or not tool_data:
            return dash.no_update, dash.no_update
        
        tool = get_tool_instance(tool_data)
        if not tool:
            return dash.no_update, dash.no_update
        
        trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        success = False
        if trigger_id == "btn-merge-upper" and upper_track1 and upper_track2:
            success = tool.merge_tracks("upper", upper_track1, upper_track2, current_frame)
        elif trigger_id == "btn-merge-lower" and lower_track1 and lower_track2:
            success = tool.merge_tracks("lower", lower_track1, lower_track2, current_frame)
        
        if success:
            # Refresh both images
            upper_data, lower_data = tool.get_frame_data(current_frame)
            upper_fig = create_image_figure(tool, current_frame, "upper", upper_data, upper_selected)
            lower_fig = create_image_figure(tool, current_frame, "lower", lower_data, lower_selected)
            return upper_fig, lower_fig
        
        return dash.no_update, dash.no_update

    # Callback for splitting tracks
    @app.callback(
        [Output("upper-image", "figure", allow_duplicate=True),
         Output("lower-image", "figure", allow_duplicate=True)],
        [Input("btn-split-upper", "n_clicks"),
         Input("btn-split-lower", "n_clicks")],
        [State("merge-upper-track1", "value"),
         State("merge-lower-track1", "value"),
         State("frame-slider", "value"),
         State("tool-store", "data"),
         State("upper-selected-track-store", "data"),
         State("lower-selected-track-store", "data")],
        prevent_initial_call=True
    )
    def handle_split_tracks(split_upper_clicks, split_lower_clicks,
                           upper_track_id, lower_track_id,
                           current_frame, tool_data, upper_selected, lower_selected):
        if not callback_context.triggered or not tool_data:
            return dash.no_update, dash.no_update
        
        tool = get_tool_instance(tool_data)
        if not tool:
            return dash.no_update, dash.no_update
        
        trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        success = False
        if trigger_id == "btn-split-upper" and upper_track_id:
            success = tool.split_track("upper", upper_track_id, current_frame)
        elif trigger_id == "btn-split-lower" and lower_track_id:
            success = tool.split_track("lower", lower_track_id, current_frame)
        
        if success:
            # Refresh both images
            upper_data, lower_data = tool.get_frame_data(current_frame)
            upper_fig = create_image_figure(tool, current_frame, "upper", upper_data, upper_selected)
            lower_fig = create_image_figure(tool, current_frame, "lower", lower_data, lower_selected)
            return upper_fig, lower_fig
        
        return dash.no_update, dash.no_update

    # Auto-populate track IDs when tracks are selected
    @app.callback(
        [Output("merge-upper-track1", "value"),
         Output("merge-lower-track1", "value")],
        [Input("upper-selected-track-store", "data"),
         Input("lower-selected-track-store", "data")]
    )
    def auto_populate_track_ids(upper_selected, lower_selected):
        return upper_selected, lower_selected

    # Update image callbacks to use tool from store
    @app.callback(
        Output("upper-image", "figure"),
        [Input("frame-slider", "value"),
         Input("upper-selected-track-store", "data")],
        [State("tool-store", "data")]
    )
    def update_upper_display(frame_idx, selected_track, tool_data):
        if not tool_data:
            return go.Figure()
            
        try:
            tool = get_tool_instance(tool_data)
            if not tool:
                return go.Figure()
                
            upper_data, _ = tool.get_frame_data(frame_idx)
            return create_image_figure(tool, frame_idx, "upper", upper_data, selected_track)
        except Exception as e:
            print(f"Error updating upper display: {e}")
            return go.Figure()
    
    @app.callback(
        Output("lower-image", "figure"),
        [Input("frame-slider", "value"),
         Input("lower-selected-track-store", "data")],
        [State("tool-store", "data")]
    )
    def update_lower_display(frame_idx, selected_track, tool_data):
        if not tool_data:
            return go.Figure()
            
        try:
            tool = get_tool_instance(tool_data)
            if not tool:
                return go.Figure()
                
            _, lower_data = tool.get_frame_data(frame_idx)
            return create_image_figure(tool, frame_idx, "lower", lower_data, selected_track)
        except Exception as e:
            print(f"Error updating lower display: {e}")
            return go.Figure()
    
    # Callback for current frame display
    @app.callback(
        Output("current-frame-display", "children"),
        [Input("frame-slider", "value")],
        [State("tool-store", "data")]
    )
    def update_frame_display(frame_idx, tool_data):
        if not tool_data:
            return "Frame 0 / 0"
        max_frame = tool_data.get("max_frame", 0)
        return f"Frame {frame_idx} / {max_frame}"
    
    # Callback for upper camera track selection
    @app.callback(
        Output("upper-selected-track-store", "data"),
        [Input("upper-image", "clickData"),
         Input("btn-clear-upper", "n_clicks"),
         Input("btn-clear-all", "n_clicks")]
    )
    def select_upper_track(upper_click, clear_upper, clear_all):
        if not callback_context.triggered:
            return None
        
        trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        # Clear selection
        if trigger_id in ["btn-clear-upper", "btn-clear-all"]:
            return None
        
        # Handle image clicks
        if trigger_id == "upper-image" and upper_click and 'points' in upper_click:
            point = upper_click['points'][0]
            if 'customdata' in point:
                track_id = point['customdata']
                print(f"Selected upper track: {track_id}")
                return track_id
        
        return None
    
    # Callback for lower camera track selection
    @app.callback(
        Output("lower-selected-track-store", "data"),
        [Input("lower-image", "clickData"),
         Input("btn-clear-lower", "n_clicks"),
         Input("btn-clear-all", "n_clicks")]
    )
    def select_lower_track(lower_click, clear_lower, clear_all):
        if not callback_context.triggered:
            return None
        
        trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        # Clear selection
        if trigger_id in ["btn-clear-lower", "btn-clear-all"]:
            return None
        
        # Handle image clicks
        if trigger_id == "lower-image" and lower_click and 'points' in lower_click:
            point = lower_click['points'][0]
            if 'customdata' in point:
                track_id = point['customdata']
                print(f"Selected lower track: {track_id}")
                return track_id
        
        return None
    
    # Callback to display upper track information
    @app.callback(
        Output("upper-track-info-display", "children"),
        [Input("upper-selected-track-store", "data")],
        [State("tool-store", "data")]
    )
    def update_upper_track_info(selected_track, tool_data):
        if not tool_data or selected_track is None:
            return html.P("No track selected", className="text-muted")
        
        try:
            tool = get_tool_instance(tool_data)
            if not tool:
                return html.P("Tool not available", className="text-muted")
            
            track_info = tool.get_track_info(selected_track, "upper")
            
            if not track_info:
                return html.P("Track not found", className="text-muted")
            
            return html.Div([
                html.P(f"🎯 Display Track ID: {track_info['display_track_id']}", className="fw-bold"),
                html.P(f"📋 Original Track IDs: {', '.join(map(str, track_info['original_track_ids']))}", className="text-muted small"),
                html.P(f"📊 Points: {track_info['total_points']}"),
                html.P(f"🎬 Frames: {track_info['frame_range']}"),
                html.P(f"📍 Last Position: {track_info['last_position']}"),
                html.P(f"🏃 Avg Velocity: {track_info['avg_velocity']:.2f} px/s"),
                html.P(f"📏 Avg Area: {track_info.get('upper_avg_area', 'N/A'):.1f}" if 'upper_avg_area' in track_info else ""),
                html.P(f"🔄 Motion: {track_info.get('upper_motion_pattern', 'unknown')}")
            ])
        except Exception as e:
            return html.P(f"Error: {e}", className="text-danger")
    
    # Callback to display lower track information
    @app.callback(
        Output("lower-track-info-display", "children"),
        [Input("lower-selected-track-store", "data")],
        [State("tool-store", "data")]
    )
    def update_lower_track_info(selected_track, tool_data):
        if not tool_data or selected_track is None:
            return html.P("No track selected", className="text-muted")
        
        try:
            tool = get_tool_instance(tool_data)
            if not tool:
                return html.P("Tool not available", className="text-muted")
            
            track_info = tool.get_track_info(selected_track, "lower")
            
            if not track_info:
                return html.P("Track not found", className="text-muted")
            
            return html.Div([
                html.P(f"🎯 Display Track ID: {track_info['display_track_id']}", className="fw-bold"),
                html.P(f"📋 Original Track IDs: {', '.join(map(str, track_info['original_track_ids']))}", className="text-muted small"),
                html.P(f"📊 Points: {track_info['total_points']}"),
                html.P(f"🎬 Frames: {track_info['frame_range']}"),
                html.P(f"📍 Last Position: {track_info['last_position']}"),
                html.P(f"🏃 Avg Velocity: {track_info['avg_velocity']:.2f} px/s"),
                html.P(f"📏 Avg Area: {track_info.get('lower_avg_area', 'N/A'):.1f}" if 'lower_avg_area' in track_info else ""),
                html.P(f"🔄 Motion: {track_info.get('lower_motion_pattern', 'unknown')}")
            ])
        except Exception as e:
            return html.P(f"Error: {e}", className="text-danger")
    
    # Callback to enable/disable stereo match button and show current selection status
    @app.callback(
        [Output("btn-create-stereo-match", "disabled"),
         Output("stereo-match-info", "children")],
        [Input("upper-selected-track-store", "data"),
         Input("lower-selected-track-store", "data")],
        [State("tool-store", "data")]
    )
    def update_stereo_match_status(upper_track, lower_track, tool_data):
        if not tool_data:
            return True, html.P("Tool not available", className="text-muted")
        
        try:
            tool = get_tool_instance(tool_data)
            if not tool:
                return True, html.P("Tool not available", className="text-muted")
            
            # Check if both tracks are selected
            both_selected = upper_track is not None and lower_track is not None
            
            # Get stereo match information
            stereo_info = tool.get_stereo_match_info(upper_track, lower_track)
            
            # Create status display
            status_elements = []
            
            if upper_track is not None:
                if stereo_info.get('upper_matched', False):
                    status_elements.append(
                        dbc.Alert(
                            f"Upper Track {upper_track} is already matched to Lower Track {stereo_info['upper_paired_with']} (Match #{stereo_info['upper_match_id']})",
                            color="info"
                        )
                    )
                else:
                    status_elements.append(
                        html.P(f"✓ Upper Track {upper_track} selected", className="text-success")
                    )
            else:
                status_elements.append(
                    html.P("⭕ No upper track selected", className="text-muted")
                )
            
            if lower_track is not None:
                if stereo_info.get('lower_matched', False):
                    status_elements.append(
                        dbc.Alert(
                            f"Lower Track {lower_track} is already matched to Upper Track {stereo_info['lower_paired_with']} (Match #{stereo_info['lower_match_id']})",
                            color="info"
                        )
                    )
                else:
                    status_elements.append(
                        html.P(f"✓ Lower Track {lower_track} selected", className="text-success")
                    )
            else:
                status_elements.append(
                    html.P("⭕ No lower track selected", className="text-muted")
                )
            
            if both_selected and not stereo_info.get('upper_matched', False) and not stereo_info.get('lower_matched', False):
                status_elements.append(
                    dbc.Alert(
                        f"Ready to create stereo match: Upper {upper_track} ↔ Lower {lower_track}",
                        color="success"
                    )
                )
            
            # Disable button if not both selected or if either track is already matched
            button_disabled = not both_selected or stereo_info.get('upper_matched', False) or stereo_info.get('lower_matched', False)
            
            return button_disabled, html.Div(status_elements)
        except Exception as e:
            return True, html.P(f"Error: {e}", className="text-danger")
    
    # Update the stereo match creation callback to use tool from store
    @app.callback(
        [Output("3d-viewer-collapse", "is_open", allow_duplicate=True),
         Output("3d-trajectory-plot", "figure", allow_duplicate=True),
         Output("3d-info-display", "children", allow_duplicate=True),
         Output("current-3d-match-store", "data", allow_duplicate=True)],
        [Input("btn-create-stereo-match", "n_clicks")],
        [State("upper-selected-track-store", "data"),
         State("lower-selected-track-store", "data"),
         State("frame-slider", "value"),
         State("tool-store", "data")],
        prevent_initial_call=True
    )
    def create_stereo_match_and_show_3d(n_clicks, upper_track, lower_track, current_frame, tool_data):
        if not n_clicks or upper_track is None or lower_track is None or not tool_data:
            return False, go.Figure(), "", None
        
        try:
            tool = get_tool_instance(tool_data)
            if not tool:
                return False, go.Figure(), "Tool not available", None
            
            success = tool.create_stereo_match(upper_track, lower_track, current_frame)
            
            if success:
                # Create success notification
                notification = dbc.Alert(
                    f"✅ Created stereo match: Upper Track {upper_track} ↔ Lower Track {lower_track}",
                    color="success",
                    dismissable=True
                )
                
                # Now automatically generate and show 3D trajectory
                try:
                    # Find the newly created match
                    match = tool.stereo_matches[
                        (tool.stereo_matches['upper_track_id'] == upper_track) & 
                        (tool.stereo_matches['lower_track_id'] == lower_track)
                    ]
                    
                    if len(match) > 0:
                        match_id = match.iloc[0]['match_id']
                        
                        # Get 3D data
                        data_3d = tool.get_3d_track_data(match_id)
                        
                        if "error" not in data_3d:
                            # Create 3D plot (same as in your existing code)
                            fig = go.Figure()
                            
                            points_3d = data_3d["points_3d"]
                            frames = data_3d["frames"]
                            
                            # Add 3D trajectory
                            fig.add_trace(go.Scatter3d(
                                x=points_3d[:, 0],
                                y=points_3d[:, 1], 
                                z=points_3d[:, 2],
                                mode='markers+lines',
                                marker=dict(
                                    size=6,
                                    color=frames,
                                    colorscale='Viridis',
                                    colorbar=dict(title="Frame"),
                                    line=dict(width=1, color='black')
                                ),
                                line=dict(color='blue', width=4),
                                name=f"Match {match_id}",
                                hovertemplate="<b>Frame %{marker.color}</b><br>" +
                                             "X: %{x:.2f} mm<br>" +
                                             "Y: %{y:.2f} mm<br>" +
                                             "Z: %{z:.2f} mm<br>" +
                                             "<extra></extra>"
                            ))
                            
                            # Add start and end markers
                            if len(points_3d) > 0:
                                fig.add_trace(go.Scatter3d(
                                    x=[points_3d[0, 0]],
                                    y=[points_3d[0, 1]],
                                    z=[points_3d[0, 2]],
                                    mode='markers',
                                    marker=dict(size=12, color='green', symbol='diamond'),
                                    name='Start',
                                    hovertemplate="<b>Start</b><br>Frame: %{text}<br>" +
                                                 "X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>Z: %{z:.2f} mm<extra></extra>",
                                    text=[frames[0]]
                                ))
                                
                                fig.add_trace(go.Scatter3d(
                                    x=[points_3d[-1, 0]],
                                    y=[points_3d[-1, 1]],
                                    z=[points_3d[-1, 2]], 
                                    mode='markers',
                                    marker=dict(size=12, color='red', symbol='diamond'),
                                    name='End',
                                    hovertemplate="<b>End</b><br>Frame: %{text}<br>" +
                                                 "X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>Z: %{z:.2f} mm<extra></extra>",
                                    text=[frames[-1]]
                                ))
                            
                            # Set layout
                            fig.update_layout(
                                title=f"3D Trajectory - Match {match_id} (Upper {upper_track} ↔ Lower {lower_track})",
                                scene=dict(
                                    xaxis_title="X (mm)",
                                    yaxis_title="Y (mm)", 
                                    zaxis_title="Z (mm)",
                                    aspectmode='data'
                                ),
                                legend=dict(x=0, y=1),
                                margin=dict(l=0, r=0, t=40, b=0)
                            )
                            
                            # Create info display with enhanced statistics
                            if "statistics" in data_3d:
                                stats = data_3d["statistics"]
                                
                                info_text = html.Div([
                                    html.H6("📊 3D Track Statistics:", className="mb-2"),
                                    dbc.Row([
                                        dbc.Col([
                                            html.P(f"• Match ID: {match_id}", className="mb-1"),
                                            html.P(f"• Upper Track: {data_3d['upper_track_id']}, Lower Track: {data_3d['lower_track_id']}", className="mb-1"),
                                            html.P(f"• Points: {data_3d['num_points']}", className="mb-1"),
                                            html.P(f"• Frame range: {min(frames)} - {max(frames)}", className="mb-1"),
                                        ], width=6),
                                        dbc.Col([
                                            html.P(f"• Total distance: {stats['total_distance_mm']:.2f} mm", className="mb-1"),
                                            html.P(f"• Total time: {stats['total_time_seconds']:.3f} s", className="mb-1"),
                                            html.P(f"• Valid intervals: {stats['valid_time_intervals']}", className="mb-1"),
                                        ], width=6)
                                    ]),
                                    html.Hr(),
                                    html.H6("🏃 Velocity Analysis:", className="mb-2"),
                                    dbc.Row([
                                        dbc.Col([
                                            html.P("Average Velocities:", className="fw-bold mb-1"),
                                            html.P(f"• X: {stats['avg_speed_xyz']['x']:.2f} mm/s", className="mb-1"),
                                            html.P(f"• Y: {stats['avg_speed_xyz']['y']:.2f} mm/s", className="mb-1"),
                                            html.P(f"• Z: {stats['avg_speed_xyz']['z']:.2f} mm/s", className="mb-1"),
                                        ], width=6),
                                        dbc.Col([
                                            html.P("Speed Statistics:", className="fw-bold mb-1"),
                                            html.P(f"• Average: {stats['avg_speed_total']:.2f} mm/s", className="mb-1"),
                                            html.P(f"• Maximum: {stats['max_speed']:.2f} mm/s", className="mb-1"),
                                            html.P(f"• Minimum: {stats['min_speed']:.2f} mm/s", className="mb-1"),
                                        ], width=6)
                                    ])
                                ])
                            else:
                                info_text = html.Div([
                                    html.P(f"📊 3D Track Statistics:"),
                                    html.P(f"• Match ID: {match_id}"),
                                    html.P(f"• Upper Track: {data_3d['upper_track_id']}, Lower Track: {data_3d['lower_track_id']}"),
                                    html.P(f"• Points: {data_3d['num_points']}"),
                                    html.P(f"• Frame range: {min(frames)} - {max(frames)}"),
                                    html.P(f"• Total distance: {np.sum(np.linalg.norm(np.diff(points_3d, axis=0), axis=1)):.2f} mm") if len(points_3d) > 1 else html.P("• Total distance: 0 mm"),
                                ])
                            
                            return True, fig, info_text, match_id
                        
                        else:
                            # Show error in 3D display
                            error_notification = dbc.Alert(
                                f"✅ Created stereo match but failed to generate 3D trajectory: {data_3d['error']}",
                                color="warning",
                                dismissable=True
                            )
                            return True, go.Figure(), f"Error: {data_3d['error']}", match_id
                    
                except Exception as e:
                    print(f"Error generating 3D visualization: {e}")
                    error_notification = dbc.Alert(
                        f"✅ Created stereo match but failed to generate 3D visualization: {e}",
                        color="warning",
                        dismissable=True
                    )
                    return False, go.Figure(), "", None
                
                return False, go.Figure(), "", None
        
            else:
                failure_notification = dbc.Alert(
                    f"❌ Failed to create stereo match - one or both tracks may already be matched",
                    color="danger",
                    dismissable=True
                )
                return False, go.Figure(), failure_notification, None
        except Exception as e:
            return False, go.Figure(), f"Error: {e}", None
    
    # Update the stereo matches table callback to use tool from store
    @app.callback(
        Output("stereo-matches-table-container", "children"),
        [Input("btn-refresh-matches", "n_clicks"),
         Input("btn-create-stereo-match", "n_clicks"),
         Input("tool-store", "data")],  # Add tool-store as an Input instead of State
        prevent_initial_call=True  # Change this to True
    )
    def update_stereo_matches_table(refresh_clicks, create_clicks, tool_data):
        if not tool_data:
            return html.P("Please select a data directory first.", className="text-muted")
        
        try:
            tool = get_tool_instance(tool_data)
            if not tool:
                return html.P("Tool not available", className="text-muted")
            
            table_data = tool.get_stereo_matches_table_data()
            
            if len(table_data) == 0:
                return html.Div([
                    html.P("No stereo matches created yet.", className="text-muted text-center p-3")
                ])
            
            # Create table with clickable rows using pattern-matching IDs
            table_rows = []
            for i, row_data in enumerate(table_data):
                table_rows.append(
                    html.Tr([
                        html.Td(str(row_data['match_id'])),
                        html.Td(str(row_data['upper_track'])),
                        html.Td(str(row_data['lower_track'])),
                        html.Td(str(row_data['common_frames'])),
                        html.Td(row_data['frame_range']),
                        html.Td(row_data['3d_status']),
                        html.Td(row_data['distance_mm']),
                        html.Td(row_data['avg_speed_mm_s']),
                        html.Td(row_data['created_at'].split('T')[0] if 'T' in row_data['created_at'] else row_data['created_at'])
                    ], 
                    id={"type": "stereo-match-row", "index": row_data['match_id']},
                    style={"cursor": "pointer"},
                    className="table-row-hover"
                    )
                )
            
            # Create table
            table_header = html.Thead([
                html.Tr([
                    html.Th("Match ID", style={"width": "8%"}),
                    html.Th("Upper Track", style={"width": "10%"}),
                    html.Th("Lower Track", style={"width": "10%"}),
                    html.Th("Common Frames", style={"width": "12%"}),
                    html.Th("Frame Range", style={"width": "15%"}),
                    html.Th("3D Status", style={"width": "10%"}),
                    html.Th("Distance (mm)", style={"width": "12%"}),
                    html.Th("Avg Speed (mm/s)", style={"width": "13%"}),
                    html.Th("Created", style={"width": "10%"})
                ])
            ])
            
            table = dbc.Table([table_header, html.Tbody(table_rows)], 
                            striped=True, 
                            hover=True, 
                            responsive=True, 
                            size="sm")
            
            return table
        except Exception as e:
            return html.P(f"Error: {e}", className="text-danger")
    
    # Update the table row click callback to use tool from store
    @app.callback(
        [Output("3d-viewer-collapse", "is_open", allow_duplicate=True),
         Output("3d-trajectory-plot", "figure", allow_duplicate=True),
         Output("3d-info-display", "children", allow_duplicate=True),
         Output("selected-stereo-match-store", "data"),
         Output("upper-selected-track-store", "data", allow_duplicate=True),
         Output("lower-selected-track-store", "data", allow_duplicate=True)],
        [Input({"type": "stereo-match-row", "index": ALL}, "n_clicks")],
        [State("tool-store", "data")],
        prevent_initial_call=True
    )
    def handle_table_row_click(n_clicks_list, tool_data):
        if not any(n_clicks_list) or not callback_context.triggered or not tool_data:
            return False, go.Figure(), "", None, None, None
        
        try:
            tool = get_tool_instance(tool_data)
            if not tool:
                return False, go.Figure(), "Tool not available", None, None, None
            
            # Find which row was clicked
            triggered_id = callback_context.triggered[0]['prop_id']
            
            # Extract match_id from the triggered component
            import json
            try:
                component_id = json.loads(triggered_id.split('.')[0])
                match_id = component_id['index']
            except:
                return False, go.Figure(), "", None, None, None
            
            # Get match data from the dataframe
            match_row = tool.stereo_matches[tool.stereo_matches['match_id'] == match_id]
            
            if len(match_row) == 0:
                return False, go.Figure(), "", None, None, None
            
            upper_track_id = match_row.iloc[0]['upper_track_id']
            lower_track_id = match_row.iloc[0]['lower_track_id']
            
            # Get 3D data
            data_3d = tool.get_3d_track_data(match_id)
            
            if "error" in data_3d:
                return True, go.Figure(), f"Error: {data_3d['error']}", match_id, upper_track_id, lower_track_id
            
            # Create 3D plot (same as above - you can copy from your existing code)
            fig = go.Figure()
            
            points_3d = data_3d["points_3d"]
            frames = data_3d["frames"]
            
            # Add 3D trajectory
            fig.add_trace(go.Scatter3d(
                x=points_3d[:, 0],
                y=points_3d[:, 1], 
 
                z=points_3d[:, 2],
                mode='markers+lines',
                marker=dict(
                    size=6,
                    color=frames,
                    colorscale='Viridis',
                    colorbar=dict(title="Frame"),
                    line=dict(width=1, color='black')
                ),
                line=dict(color='blue', width=4),
                name=f"Match {match_id}",
                hovertemplate="<b>Frame %{marker.color}</b><br>" +
                            "X: %{x:.2f} mm<br>" +
                            "Y: %{y:.2f} mm<br>" +
                            "Z: %{z:.2f} mm<br>" +
                            "<extra></extra>"
            ))
            
            # Add start and end markers
            if len(points_3d) > 0:
                fig.add_trace(go.Scatter3d(
                    x=[points_3d[0, 0]],
                    y=[points_3d[0, 1]],
                    z=[points_3d[0, 2]],
                    mode='markers',
                    marker=dict(size=12, color='green', symbol='diamond'),
                    name='Start',
                    hovertemplate="<b>Start</b><br>Frame: %{text}<br>" +
                                "X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>Z: %{z:.2f} mm<extra></extra>",
                    text=[frames[0]]
                ))
                
                fig.add_trace(go.Scatter3d(
                    x=[points_3d[-1, 0]],
                    y=[points_3d[-1, 1]],
                    z=[points_3d[-1, 2]], 
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='diamond'),
                    name='End',
                    hovertemplate="<b>End</b><br>Frame: %{text}<br>" +
                                "X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>Z: %{z:.2f} mm<extra></extra>",
                    text=[frames[-1]]
                ))
            
            # Set layout
            fig.update_layout(
                title=f"3D Trajectory - Match {match_id} (Upper {upper_track_id} ↔ Lower {lower_track_id})",
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)", 
                    zaxis_title="Z (mm)",
                    aspectmode='data'
                ),
                legend=dict(x=0, y=1),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            # Create info display
            info_text = html.Div([
                html.P(f"📊 3D Track Statistics:"),
                html.P(f"• Match ID: {match_id}"),
                html.P(f"• Upper Track: {data_3d['upper_track_id']}, Lower Track: {data_3d['lower_track_id']}"),
                html.P(f"• Points: {data_3d['num_points']}"),
                html.P(f"• Frame range: {min(frames)} - {max(frames)}"),
                html.P(f"• Total distance: {np.sum(np.linalg.norm(np.diff(points_3d, axis=0), axis=1)):.2f} mm") if len(points_3d) > 1 else html.P("• Total distance: 0 mm"),
            ])
            
            return True, fig, info_text, match_id, upper_track_id, lower_track_id
        except Exception as e:
            return False, go.Figure(), f"Error: {e}", None, None, None
    
    # Update the 3D viewer toggle callback to use tool from store
    @app.callback(
        [Output("3d-viewer-collapse", "is_open", allow_duplicate=True),
         Output("3d-trajectory-plot", "figure", allow_duplicate=True),
         Output("3d-info-display", "children", allow_duplicate=True),
         Output("current-3d-match-store", "data", allow_duplicate=True)],
        [Input("btn-view-3d", "n_clicks")],
        [State("upper-selected-track-store", "data"),
         State("lower-selected-track-store", "data"),
         State("3d-viewer-collapse", "is_open"),
         State("tool-store", "data")],
        prevent_initial_call=True
    )
    def toggle_3d_viewer(n_clicks, upper_track, lower_track, is_open, tool_data):
        if not n_clicks or not tool_data:
            return is_open, go.Figure(), "", None
        
        try:
            tool = get_tool_instance(tool_data)
            if not tool:
                return is_open, go.Figure(), "Tool not available", None
            
            # If already open, just close it
            if is_open:
                return False, go.Figure(), "", None
            
            if upper_track is None or lower_track is None:
                return is_open, go.Figure(), "", None
            
            # Find the stereo match
            upper_match = tool.stereo_matches[tool.stereo_matches['upper_track_id'] == upper_track]
            if len(upper_match) == 0:
                return is_open, go.Figure(), "No stereo match found", None
            
            match_id = upper_match.iloc[0]['match_id']
            
            # Get 3D data
            data_3d = tool.get_3d_track_data(match_id)
            
            if "error" in data_3d:
                return True, go.Figure(), f"Error: {data_3d['error']}", None
            
            # Create 3D plot (same as above)
            fig = go.Figure()
            
            points_3d = data_3d["points_3d"]
            frames = data_3d["frames"]
            
            # Add 3D trajectory
            fig.add_trace(go.Scatter3d(
                x=points_3d[:, 0],
                y=points_3d[:, 1], 
                z=points_3d[:, 2],
                mode='markers+lines',
                marker=dict(
                    size=6,
                    color=frames,
                    colorscale='Viridis',
                    colorbar=dict(title="Frame"),
                    line=dict(width=1, color='black')
                ),
                line=dict(color='blue', width=4),
                name=f"Match {match_id}",
                hovertemplate="<b>Frame %{marker.color}</b><br>" +
                             "X: %{x:.2f} mm<br>" +
                             "Y: %{y:.2f} mm<br>" +
                             "Z: %{z:.2f} mm<br>" +
                             "<extra></extra>"
            ))
            
            # Add start and end markers
            if len(points_3d) > 0:
                fig.add_trace(go.Scatter3d(
                    x=[points_3d[0, 0]],
                    y=[points_3d[0, 1]],
                    z=[points_3d[0, 2]],
                    mode='markers',
                    marker=dict(size=12, color='green', symbol='diamond'),
                    name='Start',
                    hovertemplate="<b>Start</b><br>Frame: %{text}<br>" +
                                 "X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>Z: %{z:.2f} mm<extra></extra>",
                    text=[frames[0]]
                ))
                
                fig.add_trace(go.Scatter3d(
                    x=[points_3d[-1, 0]],
                    y=[points_3d[-1, 1]],
                    z=[points_3d[-1, 2]], 
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='diamond'),
                    name='End',
                    hovertemplate="<b>End</b><br>Frame: %{text}<br>" +
                                 "X: %{x:.2f} mm<br>Y: %{y:.2f} mm<br>Z: %{z:.2f} mm<extra></extra>",
                    text=[frames[-1]]
                ))
            
            # Set layout
            fig.update_layout(
                title=f"3D Trajectory - Match {match_id} (Upper {upper_track} ↔ Lower {lower_track})",
                scene=dict(
                    xaxis_title="X (mm)",
                    yaxis_title="Y (mm)", 
                    zaxis_title="Z (mm)",
                    aspectmode='data'
                ),
                legend=dict(x=0, y=1),
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            # Create info display
            info_text = html.Div([
                html.P(f"📊 3D Track Statistics:"),
                html.P(f"• Match ID: {match_id}"),
                html.P(f"• Upper Track: {data_3d['upper_track_id']}, Lower Track: {data_3d['lower_track_id']}"),
                html.P(f"• Points: {data_3d['num_points']}"),
                html.P(f"• Frame range: {min(frames)} - {max(frames)}"),
                html.P(f"• Total distance: {np.sum(np.linalg.norm(np.diff(points_3d, axis=0), axis=1)):.2f} mm") if len(points_3d) > 1 else html.P("• Total distance: 0 mm"),
            ])
            
            return True, fig, info_text, match_id
        except Exception as e:
            return is_open, go.Figure(), f"Error: {e}", None

    return app

def create_image_figure(tool: TrackAnnotationTool, frame_idx: int, camera: str, 
                       track_data: Dict, selected_track: Optional[int]) -> go.Figure:
    """Create a plotly figure for displaying images with tracks"""
    
    # Generate the visualization image on-the-fly
    img_path = tool.generate_frame_image(frame_idx, camera, selected_track)
    
    if img_path is None:
        # Return empty figure if frame doesn't exist
        fig = go.Figure()
        fig.add_annotation(text="Frame not available", 
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font_size=20)
        return fig
    
    try:
        # Read image and convert to base64 for display
        with open(img_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
        
        # Get image dimensions
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        # Create figure
        fig = go.Figure()
        
        # Add the actual image
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{encoded_image}",
                xref="x", yref="y",
                x=0, y=0,
                sizex=img_width, sizey=img_height,
                sizing="stretch",
                opacity=1.0,
                layer="below"
            )
        )
        
        # Add vertical grid lines every 10 pixels
        for x in range(0, img_width, 50):
            fig.add_shape(
                type="line",
                x0=x, y0=0,
                x1=x, y1=img_height,
                line=dict(
                    color="yellow",
                    width=0.5,
                    dash="solid"
                ),
                layer="above"
            )

        # Add invisible clickable points for track selection (only if no track is selected)
        if selected_track is None:
            for track_id, data in track_data.items():
                # Add invisible clickable overlay
                fig.add_trace(go.Scatter(
                    x=[data['x']],
                    y=[data['y']],
                    mode='markers',
                    marker=dict(
                        size=15,
                        color="rgba(0,0,0,0)",  # Invisible
                        line=dict(width=0)
                    ),
                    customdata=[track_id],  # For click detection
                    name=f"Track {track_id}",
                    showlegend=False,
                    hovertemplate=f"<b>Track {track_id}</b><br>" +
                                 f"Position: ({data['x']:.1f}, {data['y']:.1f})<br>" +
                                 f"Area: {data['area']:.1f}<br>" +
                                 f"Click to select<br>" +
                                 "<extra></extra>"
                ))
        
        # Configure layout to show image at real size
        fig.update_layout(
            xaxis=dict(
                range=[0, img_width], 
                showgrid=False, 
                zeroline=False, 
                showticklabels=False,
                scaleanchor="y",
                scaleratio=1
            ),
            yaxis=dict(
                range=[img_height, 0],  # Flip Y axis for image display
                showgrid=False, 
                zeroline=False, 
                showticklabels=False
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            showlegend=False,
            plot_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error loading generated image {img_path}: {e}")
        # Return empty figure on error
        fig = go.Figure()
        fig.add_annotation(text=f"Error loading image: {e}", 
                          xref="paper", yref="paper", x=0.5, y=0.5,
                          showarrow=False, font_size=16)
        return fig

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True, host='localhost', port=8050)
