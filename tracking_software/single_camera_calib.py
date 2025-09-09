import argparse
import cv2
import numpy as np
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Single Camera Calibration')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing calibration images')
    parser.add_argument('--camera_name', type=str, required=True, 
                       choices=['upper', 'lower', 'left', 'right'],
                       help='Camera identifier for output files')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save calibration results')
    parser.add_argument('--chess_rows', type=int, default=8,
                       help='Number of inner corners rows in chessboard')
    parser.add_argument('--chess_cols', type=int, default=11,
                       help='Number of inner corners columns in chessboard')
    parser.add_argument('--square_size', type=float, default=1.5,
                       help='Size of chessboard squares in mm (if known)')
    parser.add_argument('--image_pattern', type=str, default='*.jpg',
                       help='Image file pattern (e.g., *.jpg, *.png)')
    return parser.parse_args()

def single_camera_calibrate(args):
    # Convert paths to absolute
    image_dir = os.path.abspath(args.image_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare object points (3D points in real world space)
    chessboard_size = (args.chess_rows, args.chess_cols)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    objp *= args.square_size  # Scale by actual square size if known
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images
    image_pattern = os.path.join(image_dir, args.image_pattern)
    images = sorted(glob.glob(image_pattern))
    
    if not images:
        print(f"No images found matching pattern: {image_pattern}")
        return None
    
    print(f"Found {len(images)} calibration images for {args.camera_name} camera")
    
    # Process each calibration image
    valid_images = 0
    for img_file in images:
        img = cv2.imread(img_file)
        if img is None:
            print(f"Failed to load image: {img_file}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners_refined)
            valid_images += 1
            
            print(f"✓ Processed: {os.path.basename(img_file)}")
        else:
            print(f"✗ No corners found in: {os.path.basename(img_file)}")
    
    if valid_images < 3:
        print(f"Error: Need at least 3 valid images for calibration, found {valid_images}")
        return None
    
    print(f"Valid images for calibration: {valid_images}")
    
    # Perform camera calibration
    image_size = gray.shape[::-1]  # (width, height)
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)
    
    print(f"\n{args.camera_name.upper()} CAMERA CALIBRATION RESULTS:")
    print("=" * 50)
    print(f"Calibration RMS error: {ret:.4f} pixels")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients: {dist_coeffs.flatten()}")
    
    # Calculate reprojection error
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], 
                                         camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    
    mean_error = total_error / len(objpoints)
    print(f"Mean reprojection error: {mean_error:.4f} pixels")
    
    # Save calibration results
    output_file = os.path.join(output_dir, f'{args.camera_name}_calibration.npz')
    np.savez(output_file,
             camera_matrix=camera_matrix,
             distortion_coefficients=dist_coeffs,
             rotation_vectors=np.array(rvecs),
             translation_vectors=np.array(tvecs),
             calibration_error=ret,
             mean_reprojection_error=mean_error,
             image_size=np.array(image_size),
             square_size=args.square_size,
             chessboard_size=np.array(chessboard_size))
    
    print(f"\nCalibration saved to: {output_file}")
    
    # Save calibration report
    save_calibration_report(args, camera_matrix, dist_coeffs, ret, mean_error, 
                           valid_images, output_dir)
    
    return camera_matrix, dist_coeffs, ret, mean_error

def save_calibration_report(args, camera_matrix, dist_coeffs, calibration_error, 
                           mean_error, valid_images, output_dir):
    """Save detailed calibration report."""
    report_file = os.path.join(output_dir, f'{args.camera_name}_calibration_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"{args.camera_name.upper()} CAMERA CALIBRATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Parameters
        f.write("CALIBRATION PARAMETERS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Image directory: {args.image_dir}\n")
        f.write(f"Camera name: {args.camera_name}\n")
        f.write(f"Chessboard size: {args.chess_rows} x {args.chess_cols}\n")
        f.write(f"Square size: {args.square_size} mm\n")
        f.write(f"Valid images used: {valid_images}\n\n")
        
        # Results
        f.write("CALIBRATION RESULTS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"RMS calibration error: {calibration_error:.4f} pixels\n")
        f.write(f"Mean reprojection error: {mean_error:.4f} pixels\n\n")
        
        f.write("CAMERA MATRIX:\n")
        f.write(f"{camera_matrix}\n\n")
        
        f.write("DISTORTION COEFFICIENTS:\n")
        f.write(f"k1: {dist_coeffs[0][0]:.6f}\n")
        f.write(f"k2: {dist_coeffs[0][1]:.6f}\n")
        f.write(f"p1: {dist_coeffs[0][2]:.6f}\n")
        f.write(f"p2: {dist_coeffs[0][3]:.6f}\n")
        f.write(f"k3: {dist_coeffs[0][4]:.6f}\n\n")
        
        # Focal lengths and principal point
        fx, fy = camera_matrix[0,0], camera_matrix[1,1]
        cx, cy = camera_matrix[0,2], camera_matrix[1,2]
        f.write("CAMERA PARAMETERS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Focal length X: {fx:.2f} pixels\n")
        f.write(f"Focal length Y: {fy:.2f} pixels\n")
        f.write(f"Principal point: ({cx:.1f}, {cy:.1f})\n")
        f.write(f"Aspect ratio: {fx/fy:.4f}\n\n")
        
        # Quality assessment
        f.write("QUALITY ASSESSMENT:\n")
        f.write("-" * 30 + "\n")
        if mean_error < 0.5:
            f.write("★ EXCELLENT: Reprojection error < 0.5 pixels\n")
        elif mean_error < 1.0:
            f.write("★ GOOD: Reprojection error < 1.0 pixels\n")
        elif mean_error < 2.0:
            f.write("★ ACCEPTABLE: Reprojection error < 2.0 pixels\n")
        else:
            f.write("⚠ POOR: Reprojection error > 2.0 pixels - consider recalibrating\n")
        
        f.write("\n" + "=" * 60 + "\n")

if __name__ == "__main__":
    args = parse_args()
    
    result = single_camera_calibrate(args)
    
    if result:
        camera_matrix, dist_coeffs, ret, mean_error = result
        print("\n✓ Camera calibration completed successfully!")
        print(f"✓ Files saved in: {args.output_dir}")
    else:
        print("\n✗ Calibration failed!")