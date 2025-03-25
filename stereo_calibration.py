import argparse
import cv2
import numpy as np
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Stereo Camera Calibration')
    parser.add_argument('--image_dir', type=str, default='.',
                       help='Directory containing calibration images')
    parser.add_argument('--left_pattern', type=str, default='*_left.jpg',
                       help='Pattern for left camera images')
    parser.add_argument('--right_pattern', type=str, default='*_right.jpg',
                       help='Pattern for right camera images')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save calibration results')
    parser.add_argument('--chess_rows', type=int, default=8,
                       help='Number of inner corners rows in chessboard')
    parser.add_argument('--chess_cols', type=int, default=11,
                       help='Number of inner corners columns in chessboard')
    return parser.parse_args()

def stereo_calibrate(args):
    # Convert relative paths to absolute
    image_dir = os.path.abspath(args.image_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full paths for image patterns
    left_pattern = os.path.join(image_dir, args.left_pattern)
    right_pattern = os.path.join(image_dir, args.right_pattern)
    
    # Prepare object points
    chessboard_size = (args.chess_rows, args.chess_cols)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    
    # Arrays to store object points and image points
    objpoints = []      # 3D points in real world space
    left_imgpoints = [] # 2D points in left image plane
    right_imgpoints = [] # 2D points in right image plane
    
    # Get list of image pairs
    left_images = sorted(glob.glob(left_pattern))
    right_images = sorted(glob.glob(right_pattern))
    
    print(f"Found {len(left_images)} image pairs")
    
    # Add this debug code before corner detection
    for left_img, right_img in zip(left_images, right_images):
        img_left = cv2.imread(left_img)
        img_right = cv2.imread(right_img)
        #print(f"Left image shape: {img_left.shape if img_left is not None else 'Failed to load'}")
        #print(f"Right image shape: {img_right.shape if img_right is not None else 'Failed to load'}")
    
    # Process each image pair
    for left_img_file, right_img_file in zip(left_images, right_images):
        # Read images
        left_img = cv2.imread(left_img_file)
        if left_img is None:
            print(f"Failed to load image: {left_img_file}")
            continue
        right_img = cv2.imread(right_img_file)
        if right_img is None:
            print(f"Failed to load image: {right_img_file}")
            continue
        
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        left_ret, left_corners = cv2.findChessboardCorners(left_gray, chessboard_size, None)
        right_ret, right_corners = cv2.findChessboardCorners(right_gray, chessboard_size, None)
        
        # Add this where you detect chessboard corners
        gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (args.chess_cols, args.chess_rows), None)
        print(f"Chessboard found: {ret} in {left_img_file}")
        
        if left_ret and right_ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            left_corners = cv2.cornerSubPix(left_gray, left_corners, (11,11), (-1,-1), criteria)
            right_corners = cv2.cornerSubPix(right_gray, right_corners, (11,11), (-1,-1), criteria)
            
            # Draw and display corners
            #cv2.drawChessboardCorners(left_img, chessboard_size, left_corners, left_ret)
            #cv2.drawChessboardCorners(right_img, chessboard_size, right_corners, right_ret)
            
            #cv2.imshow('Left Corners', left_img)
            #cv2.imshow('Right Corners', right_img)
            #cv2.waitKey(500)
            
            objpoints.append(objp)
            left_imgpoints.append(left_corners)
            right_imgpoints.append(right_corners)
            
            print(f"Processed pair: {left_img_file} and {right_img_file}")
        
        else:
    
    cv2.destroyAllWindows()
    
    # Add this before calibrateCamera call
    print("Number of object points:", len(objpoints))
    print("Number of image points:", len(left_imgpoints))
    
    # Calibrate each camera individually
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, left_imgpoints, left_gray.shape[::-1], None, None)
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, right_imgpoints, right_gray.shape[::-1], None, None)
    
    # Stereo calibration
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, left_imgpoints, right_imgpoints,
        mtx_left, dist_left,
        mtx_right, dist_right,
        left_gray.shape[::-1], criteria=criteria_stereo, flags=flags)
    
    # Save calibration results
    output_file = os.path.join(output_dir, 'stereo_calibration.npz')
    np.savez(output_file,
             cameraMatrix1=cameraMatrix1, distCoeffs1=distCoeffs1,
             cameraMatrix2=cameraMatrix2, distCoeffs2=distCoeffs2,
             R=R, T=T, E=E, F=F)
    
    print(f"Calibration completed and saved to '{output_file}'")
    return cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F

if __name__ == "__main__":
    args = parse_args()
    stereo_calibrate(args)