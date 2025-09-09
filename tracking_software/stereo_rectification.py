import numpy as np
import cv2
import pickle
import os
from typing import Tuple, List, Dict

class StereoRectifier:
    """Handle stereo rectification for track visualization"""
    
    def __init__(self, calibration_dir: str):
        """Initialize with calibration data"""
        self.calibration_dir = calibration_dir
        self.camera_matrices = {}
        self.dist_coeffs = {}
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.roi1 = None
        self.roi2 = None
        self.map1_left = None
        self.map1_right = None
        self.map2_left = None
        self.map2_right = None
        self.image_size = None
        
        self.load_calibration_data()
        # No need to compute maps - they're already in the file!
    
    def load_calibration_data(self):
        """Load stereo calibration data from .npz file"""
        
        # Look for stereo_calibration.npz file
        stereo_file = os.path.join(self.calibration_dir, 'stereo_calibration.npz')
        if not os.path.exists(stereo_file):
            raise FileNotFoundError(f"Stereo calibration file not found: {stereo_file}")
        
        print(f"Loading calibration data from: {stereo_file}")
        
        # Load the .npz file
        calib_data = np.load(stereo_file)
        
        # Print available keys for debugging
        print(f"Available calibration data keys: {list(calib_data.keys())}")
        
        # Extract data using your specific key names
        try:
            # Camera matrices and distortion coefficients
            self.camera_matrices['left'] = calib_data['left_camera_matrix']
            self.camera_matrices['right'] = calib_data['right_camera_matrix']
            self.dist_coeffs['left'] = calib_data['left_distortion']
            self.dist_coeffs['right'] = calib_data['right_distortion']
            
            # Stereo calibration results
            self.R = calib_data['R']  # Rotation between cameras
            self.T = calib_data['T']  # Translation between cameras
            
            # Pre-computed rectification data (this is great - saves computation time!)
            self.R1 = calib_data['R1']  # Left rectification rotation
            self.R2 = calib_data['R2']  # Right rectification rotation
            self.P1 = calib_data['P1']  # Left projection matrix
            self.P2 = calib_data['P2']  # Right projection matrix
            self.Q = calib_data['Q']    # Disparity-to-depth mapping matrix
            
            # ROIs (regions of interest after rectification)
            self.roi1 = tuple(calib_data['roi_left'])   # Left ROI
            self.roi2 = tuple(calib_data['roi_right'])  # Right ROI
            
            # Pre-computed rectification maps (even better!)
            self.map1_left = calib_data['left_map1']
            self.map2_left = calib_data['left_map2']
            self.map1_right = calib_data['right_map1']
            self.map2_right = calib_data['right_map2']
            
            # Image size
            self.image_size = tuple(calib_data['image_size'])
            
            print("✓ Loaded calibration data:")
            print(f"  Left camera matrix shape: {self.camera_matrices['left'].shape}")
            print(f"  Right camera matrix shape: {self.camera_matrices['right'].shape}")
            print(f"  Left distortion coeffs shape: {self.dist_coeffs['left'].shape}")
            print(f"  Right distortion coeffs shape: {self.dist_coeffs['right'].shape}")
            print(f"  Rotation matrix shape: {self.R.shape}")
            print(f"  Translation vector shape: {self.T.shape}")
            print(f"  Image size: {self.image_size}")
            print(f"  Left ROI: {self.roi1}")
            print(f"  Right ROI: {self.roi2}")
            print(f"  Rectification maps loaded: ✓")
            print(f"  Stereo error: {calib_data['stereo_error']:.4f}")
            print(f"  Baseline: {calib_data['baseline']:.2f} mm")
            
        except KeyError as e:
            raise ValueError(f"Missing calibration data key: {e}. Available keys: {list(calib_data.keys())}")
    
    def rectify_points(self, points: List[Tuple[float, float]], camera: str) -> List[Tuple[float, float]]:
        """
        Rectify a list of 2D points from distorted to rectified coordinates
        
        Args:
            points: List of (x, y) points in distorted image coordinates
            camera: 'left' or 'right'
            
        Returns:
            List of (x, y) points in rectified coordinates
        """
        if not points:
            return []
        
        # Convert to numpy array
        points_array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        
        # Undistort and rectify points
        if camera == 'left':
            rectified_points = cv2.undistortPoints(
                points_array,
                self.camera_matrices['left'],
                self.dist_coeffs['left'],
                R=self.R1,
                P=self.P1
            )
        else:  # right
            rectified_points = cv2.undistortPoints(
                points_array,
                self.camera_matrices['right'],
                self.dist_coeffs['right'],
                R=self.R2,
                P=self.P2
            )
        
        # Convert back to list of tuples
        rectified_points = rectified_points.reshape(-1, 2)
        return [(float(pt[0]), float(pt[1])) for pt in rectified_points]
    
    def rectify_track_positions(self, tracks: List, camera: str):
        """
        Rectify all positions in a list of tracks
        
        Args:
            tracks: List of SingleTrack objects
            camera: 'left' or 'right'
            
        Returns:
            List of tracks with rectified positions
        """
        rectified_tracks = []
        
        print(f"Rectifying {len(tracks)} tracks for {camera} camera...")
        
        for track in tracks:
            # Rectify all positions in this track
            original_positions = [(pos[0], pos[1]) for pos in track.positions]
            rectified_positions = self.rectify_points(original_positions, camera)
            
            # Create a copy of the track with rectified positions
            rectified_track = type(track)(
                positions=rectified_positions,
                areas=track.areas.copy(),
                frame_indices=track.frame_indices.copy(),
                timestamps=track.timestamps.copy(),
                motion_pattern=track.motion_pattern
            )
            
            rectified_tracks.append(rectified_track)
        
        return rectified_tracks
    
    def get_rectified_image_size(self) -> Tuple[int, int]:
        """Get the size of rectified images"""
        return self.image_size
    
    def get_rectified_roi(self, camera: str) -> Tuple[int, int, int, int]:
        """Get the region of interest for rectified images"""
        if camera == 'left':
            return self.roi1
        else:
            return self.roi2
    
    def save_rectification_info(self, output_dir: str):
        """Save rectification information for reference"""
        
        info_file = os.path.join(output_dir, 'stereo_rectification_info.txt')
        
        with open(info_file, 'w') as f:
            f.write("STEREO RECTIFICATION INFORMATION\n")
            f.write("="*50 + "\n\n")
            
            f.write("Calibration File:\n")
            f.write(f"  Source: {os.path.join(self.calibration_dir, 'stereo_calibration.npz')}\n\n")
            
            f.write("Image Properties:\n")
            f.write(f"  Original size: {self.image_size[0]} x {self.image_size[1]}\n")
            f.write(f"  Left ROI (x,y,w,h): {self.roi1}\n")
            f.write(f"  Right ROI (x,y,w,h): {self.roi2}\n\n")
            
            f.write("Camera Matrices:\n")
            f.write(f"  Left camera matrix:\n{self.camera_matrices['left']}\n\n")
            f.write(f"  Right camera matrix:\n{self.camera_matrices['right']}\n\n")
            
            f.write("Distortion Coefficients:\n")
            f.write(f"  Left:  {self.dist_coeffs['left'].flatten()}\n")
            f.write(f"  Right: {self.dist_coeffs['right'].flatten()}\n\n")
            
            f.write("Stereo Calibration Results:\n")
            f.write(f"  Rotation matrix (R):\n{self.R}\n\n")
            f.write(f"  Translation vector (T): {self.T.flatten()}\n")
            f.write(f"  Baseline: {np.linalg.norm(self.T):.2f} units\n\n")
            
            f.write("Rectification Matrices:\n")
            f.write(f"  R1 (Left rectification rotation):\n{self.R1}\n\n")
            f.write(f"  R2 (Right rectification rotation):\n{self.R2}\n\n")
            
            f.write("Projection Matrices (after rectification):\n")
            f.write(f"  P1 (Left projection):\n{self.P1}\n\n")
            f.write(f"  P2 (Right projection):\n{self.P2}\n\n")
            
            f.write("Disparity-to-Depth Mapping:\n")
            f.write(f"  Q matrix:\n{self.Q}\n\n")
            
        print(f"✓ Rectification info saved: {info_file}")