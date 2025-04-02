import argparse
import cv2
import numpy as np
import os
import glob

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Depth Image from Stereo Images')
    parser.add_argument('--image_dir', type=str, default='../calibration_images/stereo',
                       help='Directory containing stereo image pairs')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save depth images')
    parser.add_argument('--calib_file', type=str, default='stereo_calibration.npz',
                       help='File containing stereo calibration results')
    return parser.parse_args()

def generate_depth_image(args):
    # Convert relative paths to absolute
    image_dir = os.path.abspath(args.image_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load calibration results
    calib_data = np.load(args.calib_file)
    cameraMatrix1 = calib_data['cameraMatrix1']
    distCoeffs1 = calib_data['distCoeffs1']
    cameraMatrix2 = calib_data['cameraMatrix2']
    distCoeffs2 = calib_data['distCoeffs2']
    R = calib_data['R']
    T = calib_data['T']
    
    # Get list of image pairs
    left_pattern = os.path.join(image_dir, 'left', '*.jpg')
    right_pattern = os.path.join(image_dir, 'right', '*.jpg')
    left_images = sorted(glob.glob(left_pattern))
    right_images = sorted(glob.glob(right_pattern))
    
    print(f"Found {len(left_images)} image pairs")
    
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
        
        # Stereo rectification
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, left_gray.shape[::-1], R, T)
        
        # Compute rectification maps
        map1x, map1y = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, left_gray.shape[::-1], cv2.CV_32FC1)
        map2x, map2y = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, right_gray.shape[::-1], cv2.CV_32FC1)
        
        # Apply rectification
        left_rectified = cv2.remap(left_gray, map1x, map1y, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_gray, map2x, map2y, cv2.INTER_LINEAR)
        
        # Compute disparity map
        stereo = cv2.StereoBM_create(numDisparities=32, blockSize=15)
        disparity = stereo.compute(left_rectified, right_rectified)
        
        # Normalize the disparity map for visualization
        #disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        #disparity_normalized = np.uint8(disparity_normalized)
        
        # Save the disparity map
        disparity_file = os.path.join(output_dir, f'disparity_{os.path.basename(left_img_file)}')
        cv2.imwrite(disparity_file, disparity)#_normalized)
        
        print(f"Disparity map saved to '{disparity_file}'")
        
        # Reproject to 3D
        points_3D = cv2.reprojectImageTo3D(disparity, Q)
        
        # Save the depth image
        depth_file = os.path.join(output_dir, f'depth_{os.path.basename(left_img_file)}.npy')
        np.save(depth_file, points_3D)
        
        print(f"Depth image saved to '{depth_file}'")

if __name__ == "__main__":
    args = parse_args()
    generate_depth_image(args)