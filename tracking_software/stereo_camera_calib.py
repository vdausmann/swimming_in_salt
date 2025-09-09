import argparse
import cv2
import numpy as np
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Stereo Camera Calibration')
    parser.add_argument('--left_images', type=str, required=True,
                       help='Directory containing left camera calibration images')
    parser.add_argument('--right_images', type=str, required=True,
                       help='Directory containing right camera calibration images')
    parser.add_argument('--left_calib', type=str, required=True,
                       help='Left camera calibration file (.npz)')
    parser.add_argument('--right_calib', type=str, required=True,
                       help='Right camera calibration file (.npz)')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save stereo calibration results')
    parser.add_argument('--chess_rows', type=int, default=8,
                       help='Number of inner corners rows in chessboard')
    parser.add_argument('--chess_cols', type=int, default=11,
                       help='Number of inner corners columns in chessboard')
    parser.add_argument('--square_size', type=float, default=1.5,
                       help='Size of chessboard squares in mm')
    parser.add_argument('--image_pattern', type=str, default='*.jpg',
                       help='Image file pattern (e.g., *.jpg, *.png)')
    return parser.parse_args()

def load_single_camera_calibration(calib_file):
    """Load single camera calibration results"""
    data = np.load(calib_file)
    return (data['camera_matrix'], 
            data['distortion_coefficients'],
            data['image_size'])

def find_synchronized_image_pairs(left_dir, right_dir, pattern):
    """Find matching image pairs between left and right directories"""
    left_images = sorted(glob.glob(os.path.join(left_dir, pattern)))
    right_images = sorted(glob.glob(os.path.join(right_dir, pattern)))
    
    print(f"Looking for images in:")
    print(f"  Left: {left_dir}")
    print(f"  Right: {right_dir}")
    print(f"  Pattern: {pattern}")
    
    # Extract basenames for matching
    left_basenames = [os.path.basename(img) for img in left_images]
    right_basenames = [os.path.basename(img) for img in right_images]
    
    # Find common basenames
    common_names = set(left_basenames) & set(right_basenames)
    
    if not common_names:
        print("No matching image pairs found!")
        print(f"Left images found: {len(left_images)}")
        print(f"Right images found: {len(right_images)}")
        return [], []
    
    # Create matched pairs
    matched_left = []
    matched_right = []
    
    for name in sorted(common_names):
        left_path = os.path.join(left_dir, name)
        right_path = os.path.join(right_dir, name)
        
        if os.path.exists(left_path) and os.path.exists(right_path):
            matched_left.append(left_path)
            matched_right.append(right_path)
    
    print(f"Found {len(matched_left)} synchronized image pairs")
    return matched_left, matched_right

def stereo_calibrate(args):
    """Perform stereo camera calibration"""
    
    # Load individual camera calibrations
    left_K, left_D, left_size = load_single_camera_calibration(args.left_calib)
    right_K, right_D, right_size = load_single_camera_calibration(args.right_calib)
    
    print("Loaded camera calibrations:")
    print(f"Left camera matrix:\n{left_K}")
    print(f"Right camera matrix:\n{right_K}")
    print(f"Left image size: {left_size}")
    print(f"Right image size: {right_size}")
    
    # Find synchronized image pairs
    left_images, right_images = find_synchronized_image_pairs(
        args.left_images, args.right_images, args.image_pattern)
    
    if not left_images:
        return None
    
    # Prepare chessboard
    chessboard_size = (args.chess_rows, args.chess_cols)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    objp *= args.square_size
    
    # Storage for calibration data
    objpoints = []  # 3D points
    left_imgpoints = []  # 2D points in left images
    right_imgpoints = []  # 2D points in right images
    
    print("\nProcessing stereo image pairs...")
    valid_pairs = 0
    
    for left_img_path, right_img_path in zip(left_images, right_images):
        # Load images
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
        if left_img is None or right_img is None:
            print(f"✗ Failed to load: {os.path.basename(left_img_path)}")
            continue
        
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners in both images
        left_ret, left_corners = cv2.findChessboardCorners(left_gray, chessboard_size, None)
        right_ret, right_corners = cv2.findChessboardCorners(right_gray, chessboard_size, None)
        
        # Only use pairs where both images have valid corners
        if left_ret and right_ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 
            left_corners_refined = cv2.cornerSubPix(left_gray, left_corners, (11,11), (-1,-1), criteria)
            right_corners_refined = cv2.cornerSubPix(right_gray, right_corners, (11,11), (-1,-1), criteria)
            
            objpoints.append(objp)
            left_imgpoints.append(left_corners_refined)
            right_imgpoints.append(right_corners_refined)
            valid_pairs += 1
            
            print(f"✓ Valid pair: {os.path.basename(left_img_path)}")
        else:
            print(f"✗ Invalid pair: {os.path.basename(left_img_path)}")
    
    if valid_pairs < 5:
        print(f"Error: Need at least 5 valid stereo pairs, found {valid_pairs}")
        return None
    
    print(f"\nUsing {valid_pairs} valid stereo pairs for calibration")
    
    # Stereo calibration
    image_size = tuple(left_size.astype(int))
    print(f"Stereo calibration image size: {image_size}")
    
    print("Running stereo calibration...")
    
    # Stereo calibration flags
    flags = cv2.CALIB_FIX_INTRINSIC  # Use pre-calibrated camera matrices
    
    ret, left_K_new, left_D_new, right_K_new, right_D_new, R, T, E, F = cv2.stereoCalibrate(
        objpoints, left_imgpoints, right_imgpoints,
        left_K, left_D, right_K, right_D,
        image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5),
        flags=flags
    )
    
    print("\nSTEREO CALIBRATION RESULTS:")
    print("=" * 60)
    print(f"RMS stereo calibration error: {ret:.4f} pixels")
    print(f"Image size: {image_size}")
    print(f"Rotation matrix R:\n{R}")
    print(f"Translation vector T: {T.flatten()}")
    print(f"Essential matrix E:\n{E}")
    print(f"Fundamental matrix F:\n{F}")
    
    # Calculate baseline (distance between cameras)
    baseline = np.linalg.norm(T)
    print(f"Baseline (camera separation): {baseline:.2f} mm")
    
    # Stereo rectification
    print("\nComputing stereo rectification...")
    
    R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
        left_K_new, left_D_new, right_K_new, right_D_new,
        image_size, R, T,
        alpha=1
    )
    
    # Compute rectification maps
    left_map1, left_map2 = cv2.initUndistortRectifyMap(
        left_K_new, left_D_new, R1, P1, image_size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(
        right_K_new, right_D_new, R2, P2, image_size, cv2.CV_16SC2)
    
    # Save all results
    os.makedirs(args.output_dir, exist_ok=True)
    stereo_file = os.path.join(args.output_dir, 'stereo_calibration.npz')
    
    np.savez(stereo_file,
             # Stereo calibration results
             stereo_error=ret,
             R=R, T=T, E=E, F=F,
             baseline=baseline,
             
             # Camera parameters
             left_camera_matrix=left_K_new,
             left_distortion=left_D_new,
             right_camera_matrix=right_K_new,
             right_distortion=right_D_new,
             
             # Rectification
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             roi_left=roi_left, roi_right=roi_right,
             
             # Rectification maps
             left_map1=left_map1, left_map2=left_map2,
             right_map1=right_map1, right_map2=right_map2,
             
             # Metadata
             image_size=image_size,
             valid_pairs=valid_pairs,
             square_size=args.square_size)
    
    print(f"\nStereo calibration saved to: {stereo_file}")
    
    # Save detailed report
    save_stereo_calibration_report(args, ret, R, T, baseline, left_K_new, right_K_new, 
                                  valid_pairs, args.output_dir)
    
    return ret, R, T, Q

def save_stereo_calibration_report(args, stereo_error, R, T, baseline, left_K, right_K, 
                                  valid_pairs, output_dir):
    """Save detailed stereo calibration report."""
    report_file = os.path.join(output_dir, 'stereo_calibration_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("STEREO CAMERA CALIBRATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Parameters
        f.write("CALIBRATION PARAMETERS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Left images directory: {args.left_images}\n")
        f.write(f"Right images directory: {args.right_images}\n")
        f.write(f"Chessboard size: {args.chess_rows} x {args.chess_cols}\n")
        f.write(f"Square size: {args.square_size} mm\n")
        f.write(f"Valid image pairs used: {valid_pairs}\n\n")
        
        # Results
        f.write("STEREO CALIBRATION RESULTS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"RMS stereo calibration error: {stereo_error:.4f} pixels\n")
        f.write(f"Baseline (camera separation): {baseline:.2f} mm\n\n")
        
        f.write("ROTATION MATRIX (R):\n")
        f.write(f"{R}\n\n")
        
        f.write("TRANSLATION VECTOR (T):\n")
        f.write(f"{T.flatten()}\n\n")
        
        # Camera matrices
        f.write("LEFT CAMERA MATRIX:\n")
        f.write(f"{left_K}\n\n")
        
        f.write("RIGHT CAMERA MATRIX:\n")
        f.write(f"{right_K}\n\n")
        
        # Camera parameters
        left_fx, left_fy = left_K[0,0], left_K[1,1]
        left_cx, left_cy = left_K[0,2], left_K[1,2]
        right_fx, right_fy = right_K[0,0], right_K[1,1]
        right_cx, right_cy = right_K[0,2], right_K[1,2]
        
        f.write("CAMERA PARAMETERS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Left focal length:  fx={left_fx:.2f}, fy={left_fy:.2f}\n")
        f.write(f"Left principal point: ({left_cx:.1f}, {left_cy:.1f})\n")
        f.write(f"Right focal length: fx={right_fx:.2f}, fy={right_fy:.2f}\n")
        f.write(f"Right principal point: ({right_cx:.1f}, {right_cy:.1f})\n\n")
        
        # Quality assessment
        f.write("QUALITY ASSESSMENT:\n")
        f.write("-" * 30 + "\n")
        if stereo_error < 1.0:
            f.write("★ EXCELLENT: Stereo error < 1.0 pixels\n")
        elif stereo_error < 2.0:
            f.write("★ GOOD: Stereo error < 2.0 pixels\n")
        elif stereo_error < 3.0:
            f.write("★ ACCEPTABLE: Stereo error < 3.0 pixels\n")
        else:
            f.write("⚠ POOR: Stereo error > 3.0 pixels - consider recalibrating\n")
        
        f.write("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    args = parse_args()
    
    result = stereo_calibrate(args)
    
    if result:
        ret, R, T, Q = result
        print("\n✓ Stereo calibration completed successfully!")
        print(f"✓ Files saved in: {args.output_dir}")
        print(f"✓ Use the Q matrix for depth computation from rectified image pairs")
    else:
        print("\n✗ Stereo calibration failed!")