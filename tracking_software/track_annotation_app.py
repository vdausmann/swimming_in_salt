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
BASE_DATA_DIR = "../swimming_in_salt_data/results/tracking_results"  # Updated from detection_results

def get_available_data_directories(base_dir: str) -> List[Dict]:
    """
    Scan the base directory for subdirectories that contain track data files.
    Updated to work with new tracking pipeline structure and prioritize rectified tracks.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    data_dirs = []
    
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            # Look for rectified track files first (preferred for stereo analysis)
            upper_tracks_rectified = subdir / "upper_tracks_rectified.csv"
            lower_tracks_rectified = subdir / "lower_tracks_rectified.csv"
            
            # Fallback to original tracks if rectified not available
            upper_tracks_original = subdir / "upper_tracks.csv"
            lower_tracks_original = subdir / "lower_tracks.csv"
            
            has_rectified = upper_tracks_rectified.exists() and lower_tracks_rectified.exists()
            has_original = upper_tracks_original.exists() and lower_tracks_original.exists()
            
            if has_rectified or has_original:
                # Count visualization images
                viz_dir = subdir / "visualizations"
                original_images = 0
                rectified_images = 0
                
                if viz_dir.exists():
                    original_images = len(list(viz_dir.glob("*_original_*.png")))
                    rectified_viz_dir = viz_dir / "rectified"
                    if rectified_viz_dir.exists():
                        rectified_images = len(list(rectified_viz_dir.glob("*_rectified_*.png")))
                
                # Create descriptive label indicating coordinate system
                if has_rectified:
                    coord_system = "rectified"
                    label = f"{subdir.name} [RECTIFIED]"
                else:
                    coord_system = "original"
                    label = f"{subdir.name} [original only]"
                
                if original_images > 0 or rectified_images > 0:
                    label += f" ({original_images} orig, {rectified_images} rect viz)"
                else:
                    label += " (no visualizations)"
                
                # Only include label and value for dropdown - remove coordinate_system
                data_dirs.append({
                    'label': label,
                    'value': str(subdir)
                    # Remove coordinate_system from here since dropdown doesn't accept extra keys
                })
    
    # Sort by directory name, prioritizing rectified (based on label content)
    data_dirs.sort(key=lambda x: (not '[RECTIFIED]' in x['label'], x['label']))
    return data_dirs

class StereoCalibration:
    """Handle stereo calibration and 3D reconstruction - Updated for rectified coordinates"""
    
    def __init__(self, calibration_path: str, baseline_mm: float = 38.0):
        self.baseline_mm = baseline_mm
        self.load_calibration(calibration_path)
        
    def load_calibration(self, calibration_path: str):
        """Load stereo calibration data from npz file - Updated for new structure"""
        try:
            calib_data = np.load(calibration_path)
            
            print(f"Available calibration keys: {list(calib_data.keys())}")
            
            # Updated to match your new calibration file structure
            self.camera_matrix_upper = calib_data['left_camera_matrix']  # Assuming upper = left
            self.camera_matrix_lower = calib_data['right_camera_matrix']  # Assuming lower = right
            self.dist_coeffs_upper = calib_data['left_distortion']
            self.dist_coeffs_lower = calib_data['right_distortion']
            self.R = calib_data['R']  # Rotation matrix
            self.T = calib_data['T']  # Translation vector
            self.E = calib_data['E']  # Essential matrix
            
            # Try to load rectified projection matrices (preferred for rectified coordinates)
            if 'P1' in calib_data and 'P2' in calib_data:
                self.P1 = calib_data['P1']  # Rectified projection matrix for upper camera
                self.P2 = calib_data['P2']  # Rectified projection matrix for lower camera
                print("✅ Using rectified projection matrices from calibration")
            else:
                # Fallback: compute projection matrices from camera matrices
                self.P1 = np.hstack([self.camera_matrix_upper, np.zeros((3, 1))])
                self.P2 = np.hstack([self.camera_matrix_lower @ self.R, self.camera_matrix_lower @ self.T])
                print("⚠️  Using computed projection matrices (may not work well with rectified coordinates)")
            
            print("✅ Loaded stereo calibration data")
            print(f"   Upper camera matrix: {self.camera_matrix_upper[0,0]:.1f} focal length")
            print(f"   Lower camera matrix: {self.camera_matrix_lower[0,0]:.1f} focal length")
            print(f"   Translation: {self.T.flatten()}")
            
        except Exception as e:
            print(f"❌ Error loading calibration: {e}")
            if 'calib_data' in locals():
                print(f"Available keys: {list(calib_data.keys())}")
            raise
    
    def triangulate_points(self, upper_points: np.ndarray, lower_points: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from stereo correspondences
        Updated to work directly with rectified coordinates (no coordinate transformation needed)
        """
        if len(upper_points) != len(lower_points):
            raise ValueError("Number of upper and lower points must match")
        
        if len(upper_points) == 0:
            return np.array([]).reshape(0, 3)
        
        # Ensure points are in correct format
        upper_points = np.array(upper_points, dtype=np.float32).reshape(-1, 2)
        lower_points = np.array(lower_points, dtype=np.float32).reshape(-1, 2)
        
        print(f"Triangulating {len(upper_points)} point pairs...")
        print(f"Upper points range: x({upper_points[:,0].min():.1f}-{upper_points[:,0].max():.1f}), y({upper_points[:,1].min():.1f}-{upper_points[:,1].max():.1f})")
        print(f"Lower points range: x({lower_points[:,0].min():.1f}-{lower_points[:,0].max():.1f}), y({lower_points[:,1].min():.1f}-{lower_points[:,1].max():.1f})")
        
        # For rectified coordinates, we can use the points directly with the rectified projection matrices
        # No coordinate transformation needed since tracks are already in rectified coordinates
        
        # Triangulate using the rectified projection matrices
        points_4d_hom = cv2.triangulatePoints(
            self.P1, self.P2,
            upper_points.T, lower_points.T
        )
        
        # Convert from homogeneous coordinates
        points_3d = points_4d_hom[:3] / points_4d_hom[3]
        points_3d = points_3d.T  # Shape: (N, 3)
        
        print(f"3D points range: x({points_3d[:,0].min():.1f}-{points_3d[:,0].max():.1f}), y({points_3d[:,1].min():.1f}-{points_3d[:,1].max():.1f}), z({points_3d[:,2].min():.1f}-{points_3d[:,2].max():.1f})")
        
        return points_3d

class TrackAnnotationTool:
    """Main class for the track annotation tool - Updated to use rectified coordinates"""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.coordinate_system = "original"  # Will be updated based on available data
        self.load_data()
        self.current_frame = 0
        self.temp_dir = Path(tempfile.mkdtemp())  # For generated images
        
        # Load stereo calibration
        self.load_stereo_calibration()
        
        # Load or initialize stereo matches
        self.load_stereo_matches()
    
    def load_stereo_calibration(self):
        """Load stereo calibration data - Updated paths"""
        calibration_path = "./calibration_results/stereo_calibration.npz"
        try:
            self.stereo_calib = StereoCalibration(calibration_path, baseline_mm=38.0)
        except Exception as e:
            print(f"❌ Could not load stereo calibration: {e}")
            self.stereo_calib = None
    
    def load_data(self):
        """Load tracking data and images - Updated to prioritize rectified coordinates"""
        print(f"Loading data from {self.data_dir}")
        
        # Check for rectified tracks first (preferred)
        upper_tracks_rectified = self.data_dir / "upper_tracks_rectified.csv"
        lower_tracks_rectified = self.data_dir / "lower_tracks_rectified.csv"
        
        upper_tracks_original = self.data_dir / "upper_tracks.csv"
        lower_tracks_original = self.data_dir / "lower_tracks.csv"
        
        if upper_tracks_rectified.exists() and lower_tracks_rectified.exists():
            # Use rectified coordinates (preferred for stereo analysis)
            print("✅ Using rectified coordinate tracks")
            self.coordinate_system = "rectified"
            self.upper_tracks = pd.read_csv(upper_tracks_rectified)
            self.lower_tracks = pd.read_csv(lower_tracks_rectified)
            tracks_type = "rectified"
            
            # Also try to load original tracks for comparison/fallback
            # try:
            #     self.upper_tracks_original = pd.read_csv(upper_tracks_original)
            #     self.lower_tracks_original = pd.read_csv(lower_tracks_original)
            #     print("✅ Also loaded original coordinate tracks for reference")
            # except Exception as e:
            #     print(f"⚠️  Could not load original tracks: {e}")
            #     self.upper_tracks_original = None
            #     self.lower_tracks_original = None
                
        elif upper_tracks_original.exists() and lower_tracks_original.exists():
            # Fallback to original coordinates
            print("⚠️  Using original coordinate tracks (rectified not available)")
            self.coordinate_system = "original"
            self.upper_tracks = pd.read_csv(upper_tracks_original)
            self.lower_tracks = pd.read_csv(lower_tracks_original)
            tracks_type = "original"
            self.upper_tracks_original = None
            self.lower_tracks_original = None
        else:
            raise FileNotFoundError(f"No track files found in {self.data_dir}")
        
        print(f"✅ Loaded {tracks_type} track files:")
        print(f"   Upper: {len(self.upper_tracks)} track points")
        print(f"   Lower: {len(self.lower_tracks)} track points")
        print(f"   Coordinate system: {self.coordinate_system}")
        
        # Add display_track_id column if it doesn't exist
        self.upper_tracks = self._add_display_track_id_column(self.upper_tracks, "upper")
        self.lower_tracks = self._add_display_track_id_column(self.lower_tracks, "lower")
        
        # Save updated CSVs if new column was added
        if self.coordinate_system == "rectified":
            self.upper_tracks.to_csv(upper_tracks_rectified, index=False)
            self.lower_tracks.to_csv(lower_tracks_rectified, index=False)
        else:
            self.upper_tracks.to_csv(upper_tracks_original, index=False)
            self.lower_tracks.to_csv(lower_tracks_original, index=False)
        
        # Look for visualization images based on coordinate system
        viz_dir = self.data_dir / "visualizations"
        if viz_dir.exists():
            if self.coordinate_system == "rectified":
                # Look for rectified visualizations first
                rectified_viz_dir = viz_dir / "rectified"
                if rectified_viz_dir.exists():
                    self.upper_images = sorted(list(rectified_viz_dir.glob("upper_rectified_*.png")))
                    self.lower_images = sorted(list(rectified_viz_dir.glob("lower_rectified_*.png")))
                    viz_source = "rectified"
                else:
                    # Fallback to original visualizations
                    self.upper_images = sorted(list(viz_dir.glob("upper_original_*.png")))
                    self.lower_images = sorted(list(viz_dir.glob("lower_original_*.png")))
                    viz_source = "original (fallback)"
            else:
                # Use original visualizations
                self.upper_images = sorted(list(viz_dir.glob("upper_original_*.png")))
                self.lower_images = sorted(list(viz_dir.glob("lower_original_*.png")))
                viz_source = "original"
            
            # If specific visualizations not found, look for any matching pattern
            if not self.upper_images or not self.lower_images:
                self.upper_images = sorted(list(viz_dir.glob("upper_*.png")))
                self.lower_images = sorted(list(viz_dir.glob("lower_*.png")))
                viz_source = "generic"
            
            print(f"✅ Found visualization images ({viz_source}):")
            print(f"   Upper: {len(self.upper_images)} images")
            print(f"   Lower: {len(self.lower_images)} images")
        else:
            # No visualization directory - we'll generate images on demand
            self.upper_images = []
            self.lower_images = []
            print("⚠️  No visualization directory found - will generate images on demand")
        
        # Convert tracks to SingleTrack objects for visualization
        self.upper_track_objects = self._convert_to_track_objects(self.upper_tracks)
        self.lower_track_objects = self._convert_to_track_objects(self.lower_tracks)
        
        # Determine frame range from track data
        max_upper_frame = self.upper_tracks['frame'].max() if len(self.upper_tracks) > 0 else 0
        max_lower_frame = self.lower_tracks['frame'].max() if len(self.lower_tracks) > 0 else 0
        max_image_frame = max(len(self.upper_images) - 1, len(self.lower_images) - 1) if self.upper_images or self.lower_images else 0
        
        self.max_frame = max(max_upper_frame, max_lower_frame, max_image_frame)
        
        # Generate colors for tracks
        self.track_colors = self._generate_track_colors()
        
        print(f"✅ Data loading complete:")
        print(f"   Frame range: 0 to {self.max_frame}")
        print(f"   Track objects: {len(self.upper_track_objects)} upper, {len(self.lower_track_objects)} lower")
        print(f"   Coordinate system: {self.coordinate_system}")

    def _save_tracks_csv(self, camera: str):
        """Save tracks CSV file - Updated to save to correct coordinate system file"""
        if self.coordinate_system == "rectified":
            if camera == "upper":
                csv_path = self.data_dir / "upper_tracks_rectified.csv"
                self.upper_tracks.to_csv(csv_path, index=False)
            else:
                csv_path = self.data_dir / "lower_tracks_rectified.csv"
                self.lower_tracks.to_csv(csv_path, index=False)
        else:
            if camera == "upper":
                csv_path = self.data_dir / "upper_tracks.csv"
                self.upper_tracks.to_csv(csv_path, index=False)
            else:
                csv_path = self.data_dir / "lower_tracks.csv"
                self.lower_tracks.to_csv(csv_path, index=False)
        
        print(f"💾 Saved {camera} tracks ({self.coordinate_system}) to {csv_path}")

    def generate_frame_image(self, frame_idx: int, camera: str, selected_track: Optional[List[int]] = None) -> Optional[str]:
        """Generate a single frame image using the visualize_tracks function with fixed canvas size"""
        try:
            # Validate frame_idx
            if frame_idx is None:
                print(f"❌ Error: frame_idx is None")
                return None
                
            print(f"DEBUG: Generating frame {frame_idx} for {camera}")
            
            # Get the appropriate data
            if camera == "upper":
                track_objects = self.upper_track_objects
                tracks_df = self.upper_tracks
                canvas_size = (720, 676)  # Fixed size for upper camera
                image_files = self.upper_images
            else:
                track_objects = self.lower_track_objects
                tracks_df = self.lower_tracks
                canvas_size = (827, 676)  # Fixed size for lower camera
                image_files = self.lower_images

            # Check if we have a pre-generated visualization image
            if frame_idx < len(image_files) and selected_track is None:
                # Use existing visualization image as base
                img_path = image_files[frame_idx]
                img = cv2.imread(str(img_path))
                if img is not None:
                    # Use the pre-generated image
                    # temp_path = self.temp_dir / f"{camera}_frame_{frame_idx}_track_{selected_track or 'all'}_{self.coordinate_system}.png"
                    # cv2.imwrite(str(temp_path), img)
                    return str(img_path)
            
            # No pre-generated image available            
            # Determine which tracks to show
            tracks_to_visualize = track_objects
            # if selected_track is not None:
                # Find the track object index that corresponds to the selected display track ID
            #     selected_track_objects = []
            #     for tid, track in enumerate(track_objects):
            #         # Check if this track object contains the selected track ID
            #         track_data = tracks_df[tracks_df['display_track_id'] == selected_track]
            #         if len(track_data) > 0:
            #             # Match by comparing track data
            #             track_frames = set(track.frame_indices)
            #             data_frames = set(track_data['frame'].tolist())
            #             if track_frames == data_frames:
            #                 selected_track_objects.append(track)
            #                 break
            
            #     tracks_to_visualize = selected_track_objects
            #     print(f"Visualizing selected track {selected_track} for {camera} at frame {frame_idx}")
        
            # # Check if we have any tracks to show
            # if not tracks_to_visualize:
            #     print(f"⚠️ No tracks to visualize for {camera} frame {frame_idx}")
            #     return self._create_empty_frame_image(frame_idx, camera, canvas_size)
            
            # Create output directory for this specific frame
            temp_output_dir = self.temp_dir / f"{camera}_frames"
            temp_output_dir.mkdir(exist_ok=True)
            
            # Use the visualize_tracks function with single frame and fixed canvas size
            try:
                visualize_tracks(
                    tracks= track_objects,
                    image_dir="",  # Not used for black background
                    prefix=f"{camera}_{self.coordinate_system}",
                    output_dir=str(temp_output_dir),
                    debug_frames=[frame_idx],  # Only this frame
                    target_tracks=selected_track,  
                    track_fade_frames=0,  # Show recently active tracks
                    show_trajectory_length=None,  # Limit trajectory length for clarity
                    show_legend=False,  # Show frame information
                    show_epipolar_lines=True,  # Disable for now
                    show_area=False,  # Show object areas
                    # fixed_canvas_size=canvas_size  # Use fixed canvas size
                )
            except Exception as e:
                print(f"⚠️ Error in visualize_tracks: {e}")
                return self._create_empty_frame_image(frame_idx, camera, canvas_size)
            
            # Find the generated image
            expected_filename = f"{camera}_{self.coordinate_system}_tracking_frame_{frame_idx:03d}.png"
            generated_path = temp_output_dir / expected_filename
            
            if generated_path.exists():
                return str(generated_path)
            else:
                # Look for any matching file
                pattern = f"*frame_{frame_idx:03d}.png"
                matching_files = list(temp_output_dir.glob(pattern))
                if matching_files:
                    return str(matching_files[0])
                else:
                    print(f"❌ Generated image not found, creating fallback")
                    return self._create_empty_frame_image(frame_idx, camera, canvas_size)
                    
        except Exception as e:
            print(f"❌ Error generating frame image: {e}")
            import traceback
            traceback.print_exc()
            return self._create_empty_frame_image(frame_idx, camera, canvas_size)

    def _create_empty_frame_image(self, frame_idx: int, camera: str, canvas_size: Tuple[int, int]) -> str:
        """Create a simple black frame with just frame information using fixed canvas size"""
        try:
            canvas_width, canvas_height = canvas_size
            img = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
            
            # Add frame information
            cv2.putText(img, f"Frame: {frame_idx}", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(img, f"Camera: {camera.upper()}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img, f"Coords: {self.coordinate_system.upper()}", (20, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(img, "No tracks available", (20, 170), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
            cv2.putText(img, f"Canvas: {canvas_width}x{canvas_height}", (20, 210), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            
            # Save the image
            temp_path = self.temp_dir / f"{camera}_empty_frame_{frame_idx}.png"
            cv2.imwrite(str(temp_path), img)
            return str(temp_path)
            
        except Exception as e:
            print(f"❌ Error creating empty frame: {e}")
            return None

    def _add_display_track_id_column(self, tracks_df: pd.DataFrame, camera: str) -> pd.DataFrame:
        """Add display_track_id column if it doesn't exist"""
        if 'display_track_id' not in tracks_df.columns:
            # Initialize display_track_id with original track_id
            tracks_df['display_track_id'] = tracks_df['track_id'].copy()
            print(f"✅ Added display_track_id column to {camera} tracks")
        else:
            print(f"✅ display_track_id column already exists in {camera} tracks")
        
        return tracks_df

    def _convert_to_track_objects(self, tracks_df: pd.DataFrame) -> List[SingleTrack]:
        """Convert track DataFrame to SingleTrack objects for visualization"""
        track_objects = []
        
        # Group by track_id to create SingleTrack objects
        for track_id in tracks_df['track_id'].unique():
            track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track_data) == 0:
                continue
            
            # Extract data for SingleTrack dataclass constructor
            positions = [(row['x'], row['y']) for _, row in track_data.iterrows()]
            frame_indices = track_data['frame'].tolist()
            areas = track_data['area'].tolist() if 'area' in track_data.columns else [1.0] * len(track_data)
            
            # Extract timestamps if available
            if 'timestamp' in track_data.columns:
                timestamps = track_data['timestamp'].tolist()
            else:
                # Fallback: generate timestamps from frame numbers
                timestamps = [f"frame_{frame:06d}_time_{frame/30.0:.3f}s" for frame in frame_indices]
            
            # Get motion pattern if available
            if 'motion_pattern' in track_data.columns:
                motion_patterns = track_data['motion_pattern'].unique()
                motion_pattern = motion_patterns[0] if len(motion_patterns) > 0 else 'unknown'
            else:
                motion_pattern = 'unknown'
            
            # Create SingleTrack object with the dataclass constructor
            track = SingleTrack(
                positions=positions,
                areas=areas,
                frame_indices=frame_indices,
                timestamps=timestamps,  # NEW: Include timestamps
                motion_pattern=motion_pattern
            )
            
            # Set the track_id as an additional attribute (not part of the dataclass)
            track.track_id = track_id
            
            track_objects.append(track)
        
        return track_objects

    def _generate_track_colors(self) -> List[Tuple[int, int, int]]:
        """Generate distinct colors for tracks"""
        all_track_ids = set()
        
        # Collect all unique track IDs from both cameras
        if hasattr(self, 'upper_tracks') and len(self.upper_tracks) > 0:
            all_track_ids.update(self.upper_tracks['track_id'].unique())
        if hasattr(self, 'lower_tracks') and len(self.lower_tracks) > 0:
            all_track_ids.update(self.lower_tracks['track_id'].unique())
        
        num_tracks = len(all_track_ids)
        
        if num_tracks == 0:
            return [(0, 255, 0)]  # Default green
        
        colors = []
        for i in range(num_tracks):
            # Generate colors using HSV color space for better distribution
            hue = (i / num_tracks) % 1.0
            saturation = 0.7 + (i % 3) * 0.1  # Vary saturation slightly
            value = 0.8 + (i % 2) * 0.2  # Vary brightness slightly
            
            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to BGR for OpenCV (integers 0-255)
            bgr_color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            colors.append(bgr_color)
        
        return colors

    def get_frame_data(self, frame_idx: int) -> Tuple[Dict, Dict]:
        """Get track data for a specific frame"""
        upper_data = {}
        lower_data = {}
        
        # Get upper tracks for this frame
        if hasattr(self, 'upper_tracks') and len(self.upper_tracks) > 0:
            frame_tracks = self.upper_tracks[self.upper_tracks['frame'] == frame_idx]
            for _, row in frame_tracks.iterrows():
                track_id = row['display_track_id'] if 'display_track_id' in row else row['track_id']
                upper_data[track_id] = {
                    'x': row['x'],
                    'y': row['y'],
                    'area': row['area'] if 'area' in row else 1.0,
                    'track_id': row['track_id']
                }
        
        # Get lower tracks for this frame
        if hasattr(self, 'lower_tracks') and len(self.lower_tracks) > 0:
            frame_tracks = self.lower_tracks[self.lower_tracks['frame'] == frame_idx]
            for _, row in frame_tracks.iterrows():
                track_id = row['display_track_id'] if 'display_track_id' in row else row['track_id']
                lower_data[track_id] = {
                    'x': row['x'],
                    'y': row['y'],
                    'area': row['area'] if 'area' in row else 1.0,
                    'track_id': row['track_id']
                }
        
        return upper_data, lower_data

    def get_track_info(self, track_id: int, camera: str) -> Optional[Dict]:
        """Get detailed information about a specific track"""
        try:
            if camera == "upper":
                tracks_df = self.upper_tracks
            else:
                tracks_df = self.lower_tracks
            
            # Filter by display_track_id
            track_data = tracks_df[tracks_df['display_track_id'] == track_id]
            
            if len(track_data) == 0:
                return None
            
            # Calculate statistics
            frames = track_data['frame'].tolist()
            positions = [(row['x'], row['y']) for _, row in track_data.iterrows()]
            areas = track_data['area'].tolist() if 'area' in track_data.columns else [1.0] * len(track_data)
            
            # Calculate velocity
            if len(positions) > 1:
                distances = []
                time_diffs = []
                for i in range(1, len(positions)):
                    dist = np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                                 (positions[i][1] - positions[i-1][1])**2)
                    distances.append(dist)
                    # Assume 30 fps if no timestamp info available
                    time_diffs.append(1.0 / 30.0)
                
                avg_velocity = np.mean(distances) / np.mean(time_diffs) if time_diffs else 0.0
            else:
                avg_velocity = 0.0
            
            # Get original track IDs that were merged into this display track
            original_track_ids = track_data['track_id'].unique().tolist()
            
            return {
                'display_track_id': track_id,
                'original_track_ids': original_track_ids,
                'total_points': len(track_data),
                'frame_range': f"{min(frames)} - {max(frames)}",
                'last_position': f"({positions[-1][0]:.1f}, {positions[-1][1]:.1f})",
                'avg_velocity': avg_velocity,
                f'{camera}_avg_area': np.mean(areas),
                f'{camera}_motion_pattern': track_data['motion_pattern'].iloc[0] if 'motion_pattern' in track_data.columns else 'unknown'
            }
        except Exception as e:
            print(f"Error getting track info: {e}")
            return None

    # Add a method to extract timestamps from image filenames or calculate from frame rate
    def _extract_timestamp_from_frame(self, frame_idx: int, camera: str) -> Optional[str]:
        """Extract timestamp from frame index using track data"""
        try:
            # Method 1: Get timestamp from track data
            if camera == "upper":
                tracks_df = self.upper_tracks
            else:
                tracks_df = self.lower_tracks
            
            # Find any track that has data for this frame
            frame_data = tracks_df[tracks_df['frame'] == frame_idx]
            
            if len(frame_data) > 0 and 'timestamp' in frame_data.columns:
                # Use the timestamp from the first track found in this frame
                timestamp = frame_data.iloc[0]['timestamp']
                if timestamp and timestamp != "":
                    return str(timestamp)
            
            # Method 2: Try to extract from image filename (fallback)
            if camera == "upper" and self.upper_images:
                if frame_idx < len(self.upper_images):
                    img_path = self.upper_images[frame_idx]
                    timestamp = self._parse_timestamp_from_filename(str(img_path))
                    if timestamp:
                        return timestamp
            elif camera == "lower" and self.lower_images:
                if frame_idx < len(self.lower_images):
                    img_path = self.lower_images[frame_idx]
                    timestamp = self._parse_timestamp_from_filename(str(img_path))
                    if timestamp:
                        return timestamp
            
            # Method 3: Calculate from frame rate (final fallback)
            frame_time_seconds = frame_idx / 30.0  # 30 fps
            return f"frame_{frame_idx:06d}_time_{frame_time_seconds:.3f}s"
            
        except Exception as e:
            print(f"Error extracting timestamp for frame {frame_idx}: {e}")
            return f"frame_{frame_idx:06d}_time_{frame_idx/30.0:.3f}s"
    
    def _parse_timestamp_from_filename(self, filename: str) -> Optional[str]:
        """Parse timestamp from image filename"""
        try:
            # Common timestamp patterns in filenames
            # Adjust these regex patterns based on your filename format
            
            # Pattern 1: ISO timestamp (2023-04-04T15:30:45.123)
            iso_pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{3})?)'
            match = re.search(iso_pattern, filename)
            if match:
                return match.group(1)
            
            # Pattern 2: Unix timestamp or milliseconds (1680624645123)
            unix_pattern = r'_(\d{10,13})_'  # 10-13 digits for unix timestamp
            match = re.search(unix_pattern, filename)
            if match:
                timestamp_ms = int(match.group(1))
                if timestamp_ms > 1e12:  # Milliseconds
                    timestamp_s = timestamp_ms / 1000.0
                else:  # Seconds
                    timestamp_s = float(timestamp_ms)
                
                # Convert to ISO format
                from datetime import datetime
                dt = datetime.fromtimestamp(timestamp_s)
                return dt.isoformat()
            
            # Pattern 3: Frame-based timestamp with time info
            frame_time_pattern = r'frame_(\d+).*?(\d+\.\d+)s'
            match = re.search(frame_time_pattern, filename)
            if match:
                frame_num = int(match.group(1))
                time_s = float(match.group(2))
                return f"frame_{frame_num:06d}_time_{time_s:.3f}s"
            
            # Pattern 4: Just extract frame number and calculate time
            frame_pattern = r'frame_(\d+)'
            match = re.search(frame_pattern, filename)
            if match:
                frame_num = int(match.group(1))
                time_s = frame_num / 30.0  # Assume 30 fps
                return f"frame_{frame_num:06d}_time_{time_s:.3f}s"
            
            print(f"No timestamp pattern found in filename: {filename}")
            return None
            
        except Exception as e:
            print(f"Error parsing timestamp from filename {filename}: {e}")
            return None

    # Update the load_stereo_matches method to include frame_timestamp:
    def load_stereo_matches(self):
        """Load or initialize stereo matches with detailed frame-by-frame structure including timestamps"""
        stereo_matches_file = self.data_dir / "stereo_matches.csv"
        
        if stereo_matches_file.exists():
            try:
                self.stereo_matches = pd.read_csv(stereo_matches_file)
                # Ensure we have the new column structure
                required_columns = [
                    'camera_track_id',  # e.g., 'upper_123' or 'lower_456'
                    'frame', 
                    'x', 
                    'y', 
                    'display_track_id', 
                    'matching_track_id',  # The corresponding track ID from the other camera
                    'timestamp',  # Timestamp of when this match was created
                    'original_track_id',  # The original track_id from the CSV
                    'frame_timestamp'  # NEW: Timestamp of when the frame was captured
                ]
                
                missing_columns = [col for col in required_columns if col not in self.stereo_matches.columns]
                if missing_columns:
                    print(f"⚠️  Stereo matches file missing columns: {missing_columns}")
                    print("⚠️  Reinitializing stereo matches with new structure")
                    self.stereo_matches = pd.DataFrame(columns=required_columns)
                else:
                    print(f"✅ Loaded {len(self.stereo_matches)} existing stereo match entries")
            except Exception as e:
                print(f"⚠️  Could not load stereo matches: {e}")
                self.stereo_matches = pd.DataFrame(columns=[
                    'camera_track_id', 'frame', 'x', 'y', 'display_track_id', 
                    'matching_track_id', 'timestamp', 'original_track_id', 'frame_timestamp'
                ])
        else:
            self.stereo_matches = pd.DataFrame(columns=[
                'camera_track_id', 'frame', 'x', 'y', 'display_track_id', 
                'matching_track_id', 'timestamp', 'original_track_id', 'frame_timestamp'
            ])
            print("✅ Initialized empty stereo matches with detailed structure including frame timestamps")

    # Update the create_stereo_match method to include frame timestamps:
    def create_stereo_match(self, upper_track: int, lower_track: int, current_frame: int) -> bool:
        """Create detailed stereo matches with frame-by-frame correspondence including frame timestamps"""
        try:
            current_timestamp = datetime.now().isoformat()
            
            # Get track data for both cameras
            upper_data = self.upper_tracks[self.upper_tracks['display_track_id'] == upper_track]
            lower_data = self.lower_tracks[self.lower_tracks['display_track_id'] == lower_track]
            
            if len(upper_data) == 0 or len(lower_data) == 0:
                print(f"❌ Cannot create match - track data not found")
                return False
            
            # Find common frames
            upper_frames = set(upper_data['frame'].tolist())
            lower_frames = set(lower_data['frame'].tolist())
            common_frames = sorted(list(upper_frames.intersection(lower_frames)))
            
            if len(common_frames) == 0:
                print(f"❌ Cannot create match - no common frames between tracks")
                return False
            
            print(f"Creating stereo match for {len(common_frames)} common frames: {min(common_frames)} to {max(common_frames)}")
            
            # Remove any existing matches for these tracks (allows overwriting)
            upper_camera_track_id = f"upper_{upper_track}"
            lower_camera_track_id = f"lower_{lower_track}"
            
            # Remove existing matches for these specific tracks
            self.stereo_matches = self.stereo_matches[
                ~((self.stereo_matches['camera_track_id'] == upper_camera_track_id) |
                  (self.stereo_matches['camera_track_id'] == lower_camera_track_id))
            ]
            
            # Create new match entries for each common frame
            new_matches = []
            
            for frame in common_frames:
                # Get upper track data for this frame
                upper_frame_data = upper_data[upper_data['frame'] == frame]
                lower_frame_data = lower_data[lower_data['frame'] == frame]
                
                if len(upper_frame_data) > 0 and len(lower_frame_data) > 0:
                    upper_row = upper_frame_data.iloc[0]
                    lower_row = lower_frame_data.iloc[0]
                    
                    # Extract frame timestamps
                    upper_frame_timestamp = self._extract_timestamp_from_frame(frame, "upper")
                    lower_frame_timestamp = self._extract_timestamp_from_frame(frame, "lower")
                    
                    # Use the upper timestamp as the canonical frame timestamp
                    # (both cameras should be synchronized)
                    frame_timestamp = upper_frame_timestamp or lower_frame_timestamp or f"frame_{frame}"
                    
                    # Create upper camera entry
                    upper_entry = {
                        'camera_track_id': upper_camera_track_id,
                        'frame': frame,
                        'x': upper_row['x'],
                        'y': upper_row['y'],
                        'display_track_id': upper_track,
                        'matching_track_id': f"lower_{lower_track}",
                        'timestamp': current_timestamp,
                        'original_track_id': upper_row['track_id'],
                        'frame_timestamp': frame_timestamp
                    }
                    
                    # Create lower camera entry
                    lower_entry = {
                        'camera_track_id': lower_camera_track_id,
                        'frame': frame,
                        'x': lower_row['x'],
                        'y': lower_row['y'],
                        'display_track_id': lower_track,
                        'matching_track_id': f"upper_{upper_track}",
                        'timestamp': current_timestamp,
                        'original_track_id': lower_row['track_id'],
                        'frame_timestamp': frame_timestamp
                    }
                    
                    new_matches.extend([upper_entry, lower_entry])
            
            if new_matches:
                # Add new matches to DataFrame
                new_matches_df = pd.DataFrame(new_matches)
                self.stereo_matches = pd.concat([self.stereo_matches, new_matches_df], ignore_index=True)
                
                # Save to file
                stereo_matches_file = self.data_dir / "stereo_matches.csv"
                self.stereo_matches.to_csv(stereo_matches_file, index=False)
                
                print(f"✅ Created detailed stereo match: Upper {upper_track} ↔ Lower {lower_track}")
                print(f"   {len(new_matches)} entries created for {len(common_frames)} frames")
                print(f"   Frame range: {min(common_frames)} - {max(common_frames)}")
                print(f"   Frame timestamps: {new_matches_df['frame_timestamp'].iloc[0]} to {new_matches_df['frame_timestamp'].iloc[-2]}")
                return True
            else:
                print(f"❌ No valid frame data found for creating matches")
                return False
                
        except Exception as e:
            print(f"❌ Error creating stereo match: {e}")
            import traceback
            traceback.print_exc()
            return False

    # Update the get_3d_track_data method to use frame timestamps for speed calculation:
    def get_3d_track_data(self, upper_track_id: int, lower_track_id: int) -> Dict:
        """Get 3D trajectory data for a stereo pair using detailed match data with proper time-based speed calculation"""
        try:
            upper_camera_track_id = f"upper_{upper_track_id}"
            lower_camera_track_id = f"lower_{lower_track_id}"
            
            # Get match data for both tracks
            upper_matches = self.stereo_matches[self.stereo_matches['camera_track_id'] == upper_camera_track_id]
            lower_matches = self.stereo_matches[self.stereo_matches['camera_track_id'] == lower_camera_track_id]
            
            if len(upper_matches) == 0 or len(lower_matches) == 0:
                return {"error": "No match data found for this stereo pair"}
            
            # Find common frames
            upper_frames = set(upper_matches['frame'].tolist())
            lower_frames = set(lower_matches['frame'].tolist())
            common_frames = sorted(list(upper_frames.intersection(lower_frames)))
            
            if len(common_frames) == 0:
                return {"error": "No common frames between matched tracks"}
            
            # Check if stereo calibration is available
            if self.stereo_calib is None:
                return {"error": "Stereo calibration not available"}
            
            # Extract matching points with timestamps
            upper_points = []
            lower_points = []
            frames = []
            frame_timestamps = []
            
            for frame in common_frames:
                upper_frame_data = upper_matches[upper_matches['frame'] == frame]
                lower_frame_data = lower_matches[lower_matches['frame'] == frame]
                
                if len(upper_frame_data) > 0 and len(lower_frame_data) > 0:
                    upper_points.append([upper_frame_data.iloc[0]['x'], upper_frame_data.iloc[0]['y']])
                    lower_points.append([lower_frame_data.iloc[0]['x'], lower_frame_data.iloc[0]['y']])
                    frames.append(frame)
                    
                    # Get frame timestamp
                    frame_timestamp = upper_frame_data.iloc[0].get('frame_timestamp', f"frame_{frame}")
                    frame_timestamps.append(frame_timestamp)
            
            if len(upper_points) == 0:
                return {"error": "No matching points found"}
            
            # Triangulate 3D points
            upper_points = np.array(upper_points)
            lower_points = np.array(lower_points)
            
            points_3d = self.stereo_calib.triangulate_points(upper_points, lower_points)
            
            # Calculate enhanced statistics using frame timestamps
            statistics = self._calculate_3d_statistics(points_3d, frames, frame_timestamps)
            
            return {
                "points_3d": points_3d,
                "frames": frames,
                "frame_timestamps": frame_timestamps,
                "upper_track_id": upper_track_id,
                "lower_track_id": lower_track_id,
                "num_points": len(points_3d),
                "match_method": "detailed_frame_by_frame",
                "statistics": statistics
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def _calculate_3d_statistics(self, points_3d: np.ndarray, frames: List[int], frame_timestamps: List[str]) -> Dict:
        """Calculate detailed 3D trajectory statistics using frame timestamps"""
        try:
            if len(points_3d) < 2:
                return {"error": "Need at least 2 points for statistics"}
            
            # Convert frame timestamps to seconds
            time_seconds = []
            for i, timestamp_str in enumerate(frame_timestamps):
                try:
                    # Try to parse different timestamp formats
                    if 'time_' in timestamp_str and 's' in timestamp_str:
                        # Format: frame_000123_time_4.100s
                        time_part = timestamp_str.split('time_')[1].replace('s', '')
                        time_seconds.append(float(time_part))
                    elif 'T' in timestamp_str:
                        # ISO format: 2023-04-04T15:30:45.123
                        from datetime import datetime
                        dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        # Use time relative to first frame
                        if i == 0:
                            base_time = dt
                            time_seconds.append(0.0)
                        else:
                            time_diff = (dt - base_time).total_seconds()
                            time_seconds.append(time_diff)
                    else:
                        # Fallback: assume 30 fps
                        frame_num = frames[i]
                        time_seconds.append(frame_num / 30.0)
                except Exception as e:
                    print(f"Error parsing timestamp {timestamp_str}: {e}")
                    # Fallback to frame-based timing
                    frame_num = frames[i]
                    time_seconds.append(frame_num / 30.0)
            
            time_seconds = np.array(time_seconds)
            
            # Calculate distances between consecutive points
            distances = np.linalg.norm(np.diff(points_3d, axis=0), axis=1)
            time_intervals = np.diff(time_seconds)
            
            # Filter out zero or negative time intervals
            valid_intervals = time_intervals > 0
            valid_distances = distances[valid_intervals]
            valid_time_intervals = time_intervals[valid_intervals]
            
            if len(valid_distances) == 0:
                return {"error": "No valid time intervals for speed calculation"}
            
            # Calculate speeds
            speeds = valid_distances / valid_time_intervals  # mm/s
            
            # Calculate velocity components
            velocity_components = np.diff(points_3d, axis=0) / valid_time_intervals.reshape(-1, 1)
            avg_velocity_xyz = np.mean(velocity_components, axis=0)
            
            statistics = {
                "total_distance_mm": np.sum(distances),
                "total_time_seconds": time_seconds[-1] - time_seconds[0],
                "valid_time_intervals": len(valid_distances),
                "avg_speed_total": np.mean(speeds),
                "max_speed": np.max(speeds),
                "min_speed": np.min(speeds),
                "std_speed": np.std(speeds),
                "avg_speed_xyz": {
                    "x": avg_velocity_xyz[0],
                    "y": avg_velocity_xyz[1], 
                    "z": avg_velocity_xyz[2]
                },
                "frame_rate_estimate": len(frames) / (time_seconds[-1] - time_seconds[0]) if (time_seconds[-1] - time_seconds[0]) > 0 else 30.0,
                "time_range": f"{time_seconds[0]:.3f}s to {time_seconds[-1]:.3f}s"
            }
            
            return statistics
            
        except Exception as e:
            return {"error": f"Error calculating statistics: {e}"}

    # Update the stereo matches table to show frame timestamps:
    def get_stereo_matches_table_data(self) -> List[Dict]:
        """Get data for the stereo matches table from detailed frame-by-frame data with timestamps"""
        table_data = []
        
        if len(self.stereo_matches) == 0:
            return table_data
        
        # Group by matching pairs to create summary rows
        # Find unique stereo pairs
        stereo_pairs = set()
        
        for _, row in self.stereo_matches.iterrows():
            camera_track = row['camera_track_id']
            matching_track = row['matching_track_id']
            
            # Create a canonical pair representation (always upper first)
            if camera_track.startswith('upper_') and matching_track.startswith('lower_'):
                pair = (camera_track, matching_track)
            elif camera_track.startswith('lower_') and matching_track.startswith('upper_'):
                pair = (matching_track, camera_track)  # Swap to put upper first
            else:
                continue  # Skip invalid entries
            
            stereo_pairs.add(pair)
        
        # Create table entries for each unique pair
        for upper_track_id, lower_track_id in stereo_pairs:
            try:
                # Extract track numbers
                upper_track_num = int(upper_track_id.split('_')[1])
                lower_track_num = int(lower_track_id.split('_')[1])
                
                # Get match data for this pair
                upper_matches = self.stereo_matches[self.stereo_matches['camera_track_id'] == upper_track_id]
                lower_matches = self.stereo_matches[self.stereo_matches['camera_track_id'] == lower_track_id]
                
                if len(upper_matches) == 0 or len(lower_matches) == 0:
                    continue
                
                # Get frame information
                upper_frames = set(upper_matches['frame'].tolist())
                lower_frames = set(lower_matches['frame'].tolist())
                common_frames = upper_frames.intersection(lower_frames)
                
                if common_frames:
                    frame_range = f"{min(common_frames)} - {max(common_frames)}"
                    
                    # Get time range from frame timestamps
                    frame_timestamps = upper_matches['frame_timestamp'].tolist()
                    if frame_timestamps and len(frame_timestamps) > 1:
                        time_range = f"{frame_timestamps[0]} to {frame_timestamps[-1]}"
                    else:
                        time_range = "N/A"
                    
                    # Calculate 3D distance if stereo calibration is available
                    distance_mm = "N/A"
                    avg_speed_mm_s = "N/A"
                    status_3d = "❌ No calib"
                    
                    if self.stereo_calib is not None:
                        try:
                            # Get 3D trajectory for this pair
                            data_3d = self.get_3d_track_data(upper_track_num, lower_track_num)
                            
                            if "error" not in data_3d and "statistics" in data_3d:
                                stats = data_3d["statistics"]
                                distance_mm = f"{stats['total_distance_mm']:.1f}"
                                avg_speed_mm_s = f"{stats['avg_speed_total']:.1f}"
                                status_3d = "✅ Available"
                            elif "error" not in data_3d:
                                # Fallback calculation
                                points_3d = data_3d["points_3d"]
                                if len(points_3d) > 1:
                                    distances = np.linalg.norm(np.diff(points_3d, axis=0), axis=1)
                                    total_distance = np.sum(distances)
                                    distance_mm = f"{total_distance:.1f}"
                                    
                                    # Calculate speed (using frame rate estimate)
                                    total_time = len(points_3d) / 30.0  # Fallback
                                    avg_speed = total_distance / total_time if total_time > 0 else 0
                                    avg_speed_mm_s = f"{avg_speed:.1f}"
                                    status_3d = "✅ Available"
                                else:
                                    status_3d = "⚠️ Too few points"
                            else:
                                status_3d = f"❌ {data_3d['error'][:20]}"
                        except Exception as e:
                            status_3d = f"❌ Error: {str(e)[:20]}"
                    else:
                        status_3d = "❌ No calib"
                else:
                    frame_range = "No overlap"
                    time_range = "N/A"
                    status_3d = "❌ No overlap"
                
                # Get creation timestamp
                creation_timestamp = upper_matches.iloc[0]['timestamp'] if len(upper_matches) > 0 else "Unknown"
                
                table_data.append({
                    'pair_id': f"{upper_track_num}↔{lower_track_num}",
                    'upper_track': upper_track_num,
                    'lower_track': lower_track_num,
                    'common_frames': len(common_frames),
                    'upper_frame_count': len(upper_frames),
                    'lower_frame_count': len(lower_frames),
                    'frame_range': frame_range,
                    'time_range': time_range,  # NEW: Time range from frame timestamps
                    '3d_status': status_3d,
                    'distance_mm': distance_mm,
                    'avg_speed_mm_s': avg_speed_mm_s,
                    'created_at': creation_timestamp
                })
            except Exception as e:
                print(f"Error processing stereo pair {upper_track_id}, {lower_track_id}: {e}")
                continue
        
        return table_data




    def get_stereo_match_info(self, upper_track: Optional[int], lower_track: Optional[int]) -> Dict:
        """Get information about stereo matches for the selected tracks"""
        info = {
            'upper_matched': False,
            'lower_matched': False,
            'upper_paired_with': None,
            'lower_paired_with': None,
            'upper_frame_count': 0,
            'lower_frame_count': 0,
            'upper_frame_range': None,
            'lower_frame_range': None
        }
        
        if upper_track is not None:
            upper_camera_track_id = f"upper_{upper_track}"
            upper_matches = self.stereo_matches[self.stereo_matches['camera_track_id'] == upper_camera_track_id]
            if len(upper_matches) > 0:
                info['upper_matched'] = True
                # Extract the matched lower track ID
                matching_track = upper_matches.iloc[0]['matching_track_id']
                if matching_track.startswith('lower_'):
                    info['upper_paired_with'] = int(matching_track.split('_')[1])
                info['upper_frame_count'] = len(upper_matches)
                frames = upper_matches['frame'].tolist()
                info['upper_frame_range'] = f"{min(frames)}-{max(frames)}" if frames else None
        
        if lower_track is not None:
            lower_camera_track_id = f"lower_{lower_track}"
            lower_matches = self.stereo_matches[self.stereo_matches['camera_track_id'] == lower_camera_track_id]
            if len(lower_matches) > 0:
                info['lower_matched'] = True
                # Extract the matched upper track ID
                matching_track = lower_matches.iloc[0]['matching_track_id']
                if matching_track.startswith('upper_'):
                    info['lower_paired_with'] = int(matching_track.split('_')[1])
                info['lower_frame_count'] = len(lower_matches)
                frames = lower_matches['frame'].tolist()
                info['lower_frame_range'] = f"{min(frames)}-{max(frames)}" if frames else None
        
        return info


    def merge_tracks(self, camera: str, track1_id: int, track2_id: int, current_frame: int) -> bool:
        """Merge two tracks - placeholder implementation"""
        try:
            if camera == "upper":
                tracks_df = self.upper_tracks
            else:
                tracks_df = self.lower_tracks
            
            # Simple merge: update display_track_id of track2 to track1
            mask = tracks_df['display_track_id'] == track2_id
            tracks_df.loc[mask, 'display_track_id'] = track1_id
            
            # Save the updated tracks
            self._save_tracks_csv(camera)
            
            print(f"✅ Merged {camera} tracks {track2_id} → {track1_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error merging tracks: {e}")
            return False

    def split_track(self, camera: str, track_id: int, split_frame: int) -> bool:
        """Split a track at the specified frame - placeholder implementation"""
        try:
            if camera == "upper":
                tracks_df = self.upper_tracks
            else:
                tracks_df = self.lower_tracks
            
            # Get track data
            track_data = tracks_df[tracks_df['display_track_id'] == track_id]
            
            if len(track_data) == 0:
                return False
            
            # Find new track ID
            max_display_id = tracks_df['display_track_id'].max()
            new_track_id = max_display_id + 1
            
            # Split: points after split_frame get new track ID
            mask = (tracks_df['display_track_id'] == track_id) & (tracks_df['frame'] > split_frame)
            tracks_df.loc[mask, 'display_track_id'] = new_track_id
            
            # Save the updated tracks
            self._save_tracks_csv(camera)
            
            print(f"✅ Split {camera} track {track_id} at frame {split_frame} → new track {new_track_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error splitting track: {e}")
            return False

# Update the layout to show coordinate system information
def create_app() -> dash.Dash:
    """Create and configure the Dash application - Updated for coordinate system awareness"""
    
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
                            html.Div(id="current-frame-display", className="text-center mt-2"),
                            
                            # Add separator
                            html.Hr(className="my-3"),
                            
                            # Add random selection controls
                            html.H6("🎲 Random Stereo Selection", className="mb-2"),
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Y-tolerance (px):", className="form-label", style={"fontSize": "0.8rem"}),
                                    dbc.Input(
                                        id="y-tolerance-input", 
                                        type="number", 
                                        value=1, 
                                        min=0, 
                                        max=50,
                                        step=0.1, 
                                        size="sm"
                                    ),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Actions:", className="form-label", style={"fontSize": "0.8rem"}),
                                    html.Div([
                                        dbc.Button(
                                            "Select 50 Random", 
                                            id="btn-select-random", 
                                            color="primary", 
                                            size="sm", 
                                            className="w-100 mb-1"
                                        ),
                                        dbc.Button(
                                            "Clear Random", 
                                            id="btn-clear-random", 
                                            color="secondary", 
                                            size="sm", 
                                            className="w-100",
                                            disabled=True
                                        ),
                                    ])
                                ], width=6)
                            ], className="mb-2"),
                            html.Div(id="random-selection-status", className="text-muted text-center", style={"fontSize": "0.7rem"})        
                        ])
                    ])
                ], width=4),
                
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("🔧 Track Editing"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Lower Track ID:", className="form-label"),
                                    dbc.Input(id="merge-lower-track1", type="number", placeholder="Track 1", size="sm"),
                                    dbc.Input(id="merge-lower-track2", type="number", placeholder="Track 2", size="sm", className="mt-1"),
                                    dbc.Button("🔗 Merge Lower", id="btn-merge-lower", color="warning", size="sm", className="w-100 mt-1"),
                                    dbc.Button("✂️ Split Lower", id="btn-split-lower", color="info", size="sm", className="w-100 mt-1"),
                                ], width=6),
                                dbc.Col([
                                    html.Label("Upper Track ID:", className="form-label"),
                                    dbc.Input(id="merge-upper-track1", type="number", placeholder="Track 1", size="sm"),
                                    dbc.Input(id="merge-upper-track2", type="number", placeholder="Track 2", size="sm", className="mt-1"),
                                    dbc.Button("🔗 Merge Upper", id="btn-merge-upper", color="warning", size="sm", className="w-100 mt-1"),
                                    dbc.Button("✂️ Split Upper", id="btn-split-upper", color="info", size="sm", className="w-100 mt-1"),
                                ], width=6),
                                
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
                        dbc.CardHeader("📷 Lower Camera"),
                        dbc.CardBody([
                            dcc.Graph(
                                id="lower-image", 
                                style={"height": "938px"},
                                config={'displayModeBar': False, "scrollZoom": False, "doubleClick": "reset"}
                            )
                        ], style={"padding": "0"})
                    ])
                ], width=6),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📷 Upper Camera"),
                        dbc.CardBody([
                            dcc.Graph(
                                id="upper-image", 
                                style={"height": "938px"},
                                config={'displayModeBar': False, "scrollZoom": False, "doubleClick": "reset"}
                            )
                        ], style={"padding": "0"})
                    ])
                ], width=6),
                
            ], className="mb-3"),    
            
            # Track information panel
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("📊 Track Information"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Lower Camera Selection"),
                                    html.Div(id="lower-track-info-display", style={"minHeight": "120px"})
                                ], width=6),
                                dbc.Col([
                                    html.H6("Upper Camera Selection"),
                                    html.Div(id="upper-track-info-display", style={"minHeight": "120px"})
                                ], width=6),                                
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
        dcc.Store(id="random-selected-tracks-store"),
        
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
            
            # Store tool data (coordinate system is determined by the tool, not dropdown)
            tool_data = {
                "data_dir": selected_dir,
                "max_frame": tool.max_frame,
                "coordinate_system": tool.coordinate_system,  # Get from tool instance
                "initialized": True
            }
            
            # Create status message with coordinate system info
            coord_status = "✅ RECTIFIED (stereo-ready)" if tool.coordinate_system == "rectified" else "⚠️ ORIGINAL (may need rectification for stereo)"
            
            status_msg = dbc.Alert([
                html.P(f"✅ Loaded data from {Path(selected_dir).name}", className="mb-1"),
                html.P(f"📊 {tool.max_frame + 1} frames", className="mb-1"),
                html.P(f"📐 Coordinate system: {coord_status}", className="mb-0")
            ], color="success" if tool.coordinate_system == "rectified" else "warning")
            
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
    
    # Add a new callback for random stereo track selection
    @app.callback(
        [Output("random-selected-tracks-store", "data"),
        Output("random-selection-status", "children"),
        Output("btn-clear-random", "disabled")],
        [Input("btn-select-random", "n_clicks"),
        Input("btn-clear-random", "n_clicks")],
        [State("tool-store", "data"),
        State("frame-slider", "value"),
        State("y-tolerance-input", "value")],
        prevent_initial_call=True
    )
    def handle_random_stereo_selection(select_clicks, clear_clicks, tool_data, current_frame, y_tolerance):
        if not callback_context.triggered or not tool_data:
            return None, "No data available", True
        
        trigger_id = callback_context.triggered[0]['prop_id'].split('.')[0]
        
        if trigger_id == "btn-clear-random":
            return None, "Random selection cleared", True
        
        if trigger_id == "btn-select-random":
            try:
                tool = get_tool_instance(tool_data)
                if not tool:
                    return None, "Tool not available", True
                
                # Get current frame data for both cameras
                upper_data, lower_data = tool.get_frame_data(current_frame)
                
                if len(upper_data) == 0:
                    return None, f"No upper tracks in frame {current_frame}", True
                
                if len(lower_data) == 0:
                    return None, f"No lower tracks in frame {current_frame}", True
                
                # Get upper tracks that are present in the current frame
                upper_frame_tracks = list(upper_data.keys())
                
                if len(upper_frame_tracks) == 0:
                    return None, f"No upper tracks in current frame {current_frame}", True
                
                # Select up to 50 random upper tracks from those present in current frame
                import random
                num_to_select = min(50, len(upper_frame_tracks))
                selected_upper_tracks = random.sample(upper_frame_tracks, num_to_select)
                
                # For each selected upper track, find ALL corresponding lower tracks based on y-position in CURRENT FRAME
                corresponding_lower_tracks = set()  # Use set to avoid duplicates
                total_matches = 0
                
                print(f"DEBUG: Looking for matches in frame {current_frame} with tolerance {y_tolerance}px")
                
                for upper_track_id in selected_upper_tracks:
                    # Get the upper track's y-position in the current frame
                    upper_y = upper_data[upper_track_id]['y']
                    
                    print(f"DEBUG: Upper track {upper_track_id} at y={upper_y:.2f}")
                    
                    # Find ALL lower tracks within tolerance in the current frame
                    matching_lower_tracks = []
                    
                    for lower_track_id, lower_track_data in lower_data.items():
                        lower_y = lower_track_data['y']
                        y_diff = abs(upper_y - lower_y)
                        
                        print(f"DEBUG:   Lower track {lower_track_id} at y={lower_y:.2f}, diff={y_diff:.2f}")
                        
                        # Check if within tolerance
                        if y_diff <= y_tolerance:
                            matching_lower_tracks.append(lower_track_id)
                            print(f"DEBUG:     Match found: {lower_track_id} (diff={y_diff:.2f})")
                    
                    # Add all matching lower tracks
                    if matching_lower_tracks:
                        corresponding_lower_tracks.update(matching_lower_tracks)
                        total_matches += len(matching_lower_tracks)
                        print(f"DEBUG: Upper {upper_track_id} matched to {len(matching_lower_tracks)} lower tracks: {matching_lower_tracks}")
                    else:
                        print(f"DEBUG: No matches found for upper track {upper_track_id}")
                
                # Convert to lists for storage
                corresponding_lower_tracks = list(corresponding_lower_tracks)
                
                selected_tracks = {
                    'upper': selected_upper_tracks,
                    'lower': corresponding_lower_tracks
                }
                
                # Calculate some statistics for the status message
                upper_with_matches = sum(1 for upper_id in selected_upper_tracks 
                                    if any(abs(upper_data[upper_id]['y'] - lower_data[lower_id]['y']) <= y_tolerance 
                                            for lower_id in lower_data.keys()))
                
                avg_matches_per_upper = total_matches / len(selected_upper_tracks) if len(selected_upper_tracks) > 0 else 0
                
                status_msg = f"Frame {current_frame}: {len(selected_upper_tracks)} upper → {len(corresponding_lower_tracks)} lower tracks | {upper_with_matches}/{len(selected_upper_tracks)} upper tracks have matches | Avg {avg_matches_per_upper:.1f} matches/upper (tol: {y_tolerance}px)"
                
                print(f"DEBUG: Final selection - Upper: {selected_upper_tracks}")
                print(f"DEBUG: Final selection - Lower: {corresponding_lower_tracks}")
                print(f"DEBUG: Total matches found: {total_matches}")
                
                return selected_tracks, status_msg, False
                
            except Exception as e:
                print(f"ERROR in random selection: {e}")
                import traceback
                traceback.print_exc()
                return None, f"Error: {e}", True
        
        return None, "No action", True

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
        Output("upper-image", "figure", allow_duplicate=True),
        [Input("frame-slider", "value"),
        Input("upper-selected-track-store", "data"),
        Input("random-selected-tracks-store", "data")],
        [State("tool-store", "data")],
        prevent_initial_call=True
    )
    def update_upper_display(frame_idx, selected_track, random_tracks, tool_data):
        if not tool_data:
            return go.Figure()
            
        try:
            tool = get_tool_instance(tool_data)
            if not tool:
                return go.Figure()
                
            upper_data, _ = tool.get_frame_data(frame_idx)
            
            # Extract upper tracks from random selection
            upper_random_tracks = None
            if random_tracks and 'upper' in random_tracks:
                upper_random_tracks = random_tracks['upper']
                
            return create_image_figure(tool, frame_idx, "upper", upper_data, selected_track, upper_random_tracks)
        except Exception as e:
            print(f"Error updating upper display: {e}")
            return go.Figure()

    @app.callback(
        Output("lower-image", "figure", allow_duplicate=True),
        [Input("frame-slider", "value"),
        Input("lower-selected-track-store", "data"),
        Input("random-selected-tracks-store", "data")],
        [State("tool-store", "data")],
        prevent_initial_call=True
    )
    def update_lower_display(frame_idx, selected_track, random_tracks, tool_data):
        if not tool_data:
            return go.Figure()
            
        try:
            tool = get_tool_instance(tool_data)
            if not tool:
                return go.Figure()
                
            _, lower_data = tool.get_frame_data(frame_idx)
            
            # Extract lower tracks from random selection
            lower_random_tracks = None
            if random_tracks and 'lower' in random_tracks:
                lower_random_tracks = random_tracks['lower']
                
            return create_image_figure(tool, frame_idx, "lower", lower_data, selected_track, lower_random_tracks)
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
                        dbc.Alert([
                            html.P(f"Upper Track {upper_track} is already matched to Lower Track {stereo_info['upper_paired_with']}", className="mb-1"),
                            html.P(f"Frames: {stereo_info['upper_frame_range']} ({stereo_info['upper_frame_count']} entries)", className="mb-0 small text-muted")
                        ], color="info")
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
                        dbc.Alert([
                            html.P(f"Lower Track {lower_track} is already matched to Upper Track {stereo_info['lower_paired_with']}", className="mb-1"),
                            html.P(f"Frames: {stereo_info['lower_frame_range']} ({stereo_info['lower_frame_count']} entries)", className="mb-0 small text-muted")
                        ], color="info")
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
                # Show preview of what would be matched
                upper_data = tool.upper_tracks[tool.upper_tracks['display_track_id'] == upper_track]
                lower_data = tool.lower_tracks[tool.lower_tracks['display_track_id'] == lower_track]
                
                if len(upper_data) > 0 and len(lower_data) > 0:
                    upper_frames = set(upper_data['frame'].tolist())
                    lower_frames = set(lower_data['frame'].tolist())
                    common_frames = upper_frames.intersection(lower_frames)
                    
                    status_elements.append(
                        dbc.Alert([
                            html.P(f"Ready to create stereo match: Upper {upper_track} ↔ Lower {lower_track}", className="mb-1"),
                            html.P(f"Will create {len(common_frames)} frame-by-frame correspondences", className="mb-0 small")
                        ], color="success")
                    )
            
            # Show overwrite warning if either track is already matched
            if both_selected and (stereo_info.get('upper_matched', False) or stereo_info.get('lower_matched', False)):
                status_elements.append(
                    dbc.Alert([
                        html.P("⚠️ Creating this match will overwrite existing match data for these tracks", className="mb-0")
                    ], color="warning")
                )
            
            # Button is enabled if both tracks are selected (allows overwriting)
            button_disabled = not both_selected
            
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
        Input("tool-store", "data")],  
        prevent_initial_call=True  
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
                    html.P("No stereo matches created yet.", className="text-muted text-center p-3"),
                    html.P("Create matches by selecting tracks from both cameras and clicking 'Create Stereo Match'.", className="text-muted text-center small")
                ])
            
            # Create table with clickable rows
            table_rows = []
            for i, row_data in enumerate(table_data):
                table_rows.append(
                    html.Tr([
                        html.Td(str(row_data['pair_id'])),
                        html.Td(str(row_data['upper_track'])),
                        html.Td(str(row_data['lower_track'])),
                        html.Td(f"{row_data['common_frames']} ({row_data['upper_frame_count']}/{row_data['lower_frame_count']})"),
                        html.Td(row_data['frame_range']),
                        html.Td(row_data['time_range']),  # NEW column
                        html.Td(row_data['3d_status']),
                        html.Td(row_data['distance_mm']),
                        html.Td(row_data['avg_speed_mm_s']),
                        html.Td(row_data['created_at'].split('T')[0] if 'T' in str(row_data['created_at']) else str(row_data['created_at'])[:10])
                    ], 
                    id={"type": "stereo-match-row", "upper": row_data['upper_track'], "lower": row_data['lower_track']},
                    style={"cursor": "pointer"},
                    className="table-row-hover"
                    )
                )
            
            # Create table with updated headers
            # Create table with updated headers including time range
            table_header = html.Thead([
                html.Tr([
                    html.Th("Pair", style={"width": "8%"}),
                    html.Th("Upper Track", style={"width": "8%"}),
                    html.Th("Lower Track", style={"width": "8%"}),
                    html.Th("Frames (U/L)", style={"width": "10%"}),
                    html.Th("Frame Range", style={"width": "12%"}),
                    html.Th("Time Range", style={"width": "15%"}),  # NEW column
                    html.Th("3D Status", style={"width": "8%"}),
                    html.Th("Distance (mm)", style={"width": "10%"}),
                    html.Th("Avg Speed (mm/s)", style={"width": "12%"}),
                    html.Th("Created", style={"width": "9%"})
                ])
            ])

            
            table = dbc.Table([table_header, html.Tbody(table_rows)], 
                            striped=True, 
                            hover=True, 
                            responsive=True, 
                            size="sm")
            
            return html.Div([
                table,
                html.P(f"Total: {len(table_data)} stereo track pairs with detailed frame-by-frame correspondences", 
                    className="text-muted text-center mt-2 small")
            ])
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
        [Input({"type": "stereo-match-row", "upper": ALL, "lower": ALL}, "n_clicks")],
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
            
            # Extract track IDs from the triggered component
            import json
            try:
                component_id = json.loads(triggered_id.split('.')[0])
                upper_track_id = component_id['upper']
                lower_track_id = component_id['lower']
            except:
                return False, go.Figure(), "", None, None, None
            
            # Get 3D data using the new method
            data_3d = tool.get_3d_track_data(upper_track_id, lower_track_id)
            
            if "error" in data_3d:
                return True, go.Figure(), f"Error: {data_3d['error']}", f"{upper_track_id}↔{lower_track_id}", upper_track_id, lower_track_id
            
            # Create 3D plot (same as before)
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
                name=f"Upper {upper_track_id} ↔ Lower {lower_track_id}",
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
                title=f"3D Trajectory - Upper {upper_track_id} ↔ Lower {lower_track_id} ({data_3d['match_method']})",
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
                html.P(f"📊 Detailed 3D Track Statistics:"),
                html.P(f"• Stereo Pair: Upper {data_3d['upper_track_id']} ↔ Lower {data_3d['lower_track_id']}"),
                html.P(f"• Match Method: {data_3d['match_method']}"),
                html.P(f"• 3D Points: {data_3d['num_points']}"),
                html.P(f"• Frame range: {min(frames)} - {max(frames)}"),
                html.P(f"• Total distance: {np.sum(np.linalg.norm(np.diff(points_3d, axis=0), axis=1)):.2f} mm") if len(points_3d) > 1 else html.P("• Total distance: 0 mm"),
            ])
            
            return True, fig, info_text, f"{upper_track_id}↔{lower_track_id}", upper_track_id, lower_track_id
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
                       track_data: Dict, selected_track: Optional[int], 
                       random_tracks: Optional[List[int]] = None) -> go.Figure:
    """Create a plotly figure for displaying images with tracks - Fixed canvas version with random selection"""
    
    # Generate the visualization image on-the-fly
    if selected_track is not None:
        img_path = tool.generate_frame_image(frame_idx, camera, [selected_track])
    else:
        img_path = tool.generate_frame_image(frame_idx, camera, random_tracks)
    
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
        
        # Get image dimensions (should match our fixed canvas sizes)
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
        
        # Add invisible clickable points for track selection (only if no track is selected)
        if selected_track is None:
            # Get the SAME track objects that were used for visualization
            if camera == "upper":
                track_objects = tool.upper_track_objects
                tracks_df = tool.upper_tracks
            else:
                track_objects = tool.lower_track_objects
                tracks_df = tool.lower_tracks
            
            # CALCULATE THE SAME CANVAS OFFSET AS visualize_tracks
            # Collect all positions from ALL track objects (same as visualize_tracks)
            all_x = []
            all_y = []
            for track in track_objects:
                for pos in track.positions:
                    all_x.append(pos[0])
                    all_y.append(pos[1])
            
            if all_x and all_y:
                # Use the SAME logic as visualize_tracks
                min_x, max_x = int(min(all_x)), int(max(all_x))
                min_y, max_y = int(min(all_y)), int(max(all_y))
                padding = 50  # Same padding as visualize_tracks
                
                # Same offset calculation as visualize_tracks
                offset_x = padding - min_x
                offset_y = padding - min_y
                
                # Create clickable points based on track_objects that are active in this frame
                for tid, track in enumerate(track_objects):
                    # Check if this track is active in the current frame
                    if frame_idx in track.frame_indices:
                        # Get the position for this frame
                        frame_idx_in_track = track.frame_indices.index(frame_idx)
                        track_position = track.positions[frame_idx_in_track]
                        
                        # Get the display_track_id for this track object
                        original_track_id = getattr(track, 'track_id', None)
                        
                        # Find the display_track_id for this original track_id in this frame
                        matching_rows = tracks_df[
                            (tracks_df['track_id'] == original_track_id) & 
                            (tracks_df['frame'] == frame_idx)
                        ]
                        
                        if len(matching_rows) > 0:
                            display_track_id = matching_rows.iloc[0]['display_track_id']
                            area = matching_rows.iloc[0]['area'] if 'area' in matching_rows.columns else 1.0
                            
                            # Check if this track should be included based on random selection
                            include_track = True
                            if random_tracks is not None:
                                include_track = display_track_id in random_tracks
                            
                            # Only add clickable points for tracks in the random selection (or all if no random selection)
                            if include_track:
                                # Apply the SAME coordinate transformation as visualize_tracks
                                display_x = track_position[0] + offset_x
                                display_y = track_position[1] + offset_y
                                
                                # Only add clickable points that are within the image bounds
                                if 0 <= display_x < img_width and 0 <= display_y < img_height:
                                    # Add semi-visible clickable overlay
                                    marker_color = "rgba(255,0,0,0.3)" if random_tracks is not None else "rgba(0,0,0,0)"
                                    marker_size = 12 if random_tracks is not None else 15
                                    
                                    fig.add_trace(go.Scatter(
                                        x=[display_x],
                                        y=[display_y],
                                        mode='markers',
                                        marker=dict(
                                            size=marker_size,
                                            color=marker_color,
                                            line=dict(width=1 if random_tracks is not None else 0, color="red" if random_tracks is not None else "rgba(0,0,0,0)")
                                        ),
                                        customdata=[display_track_id],  # Use display_track_id for click detection
                                        name=f"Track {display_track_id}" + (" (Random)" if random_tracks is not None else ""),
                                        showlegend=False,
                                        hovertemplate=f"<b>Track {display_track_id}</b>" + 
                                                     (" (Random Selection)" if random_tracks is not None else "") + "<br>" +
                                                     f"Raw Position: ({track_position[0]:.1f}, {track_position[1]:.1f})<br>" +
                                                     f"Display Position: ({display_x:.1f}, {display_y:.1f})<br>" +
                                                     f"Offset: ({offset_x}, {offset_y})<br>" +
                                                     f"Area: {area:.1f}<br>" +
                                                     f"Original ID: {original_track_id}<br>" +
                                                     f"Track Object Index: {tid}<br>" +
                                                     f"Click to select<br>" +
                                                     "<extra></extra>"
                                    ))
                                else:
                                    print(f"DEBUG: Track {display_track_id} outside bounds: ({display_x:.1f}, {display_y:.1f}) vs image {img_width}x{img_height}")
        
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
