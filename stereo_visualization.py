import cv2
import numpy as np
import os
import argparse
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend
import matplotlib.pyplot as plt

def visualize_stereo_calibration(left_img_path, right_img_path, calib_file, output_dir='.'):
    # Debug: Print image paths
    print(f"Loading images from:\nLeft: {left_img_path}\nRight: {right_img_path}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load and verify images
    img_left = cv2.imread(left_img_path)
    img_right = cv2.imread(right_img_path)
    
    # Debug: Print image shapes and pixel value ranges
    print(f"Left image shape: {img_left.shape if img_left is not None else 'Failed to load'}")
    print(f"Left image value range: [{np.min(img_left)} - {np.max(img_left)}]")
    print(f"Right image shape: {img_right.shape if img_right is not None else 'Failed to load'}")
    print(f"Right image value range: [{np.min(img_right)} - {np.max(img_right)}]")
    
    if img_left is None or img_right is None:
        raise ValueError(f"Could not read images from {left_img_path} or {right_img_path}")
        
    h, w = img_left.shape[:2]

    # Save original images before rectification
    cv2.imwrite(os.path.join(output_dir, 'original_left.jpg'), img_left)
    cv2.imwrite(os.path.join(output_dir, 'original_right.jpg'), img_right)
    
    # Load calibration data
    calib_data = np.load(calib_file)
    cameraMatrix1 = calib_data['cameraMatrix1']
    distCoeffs1 = calib_data['distCoeffs1']
    cameraMatrix2 = calib_data['cameraMatrix2']
    distCoeffs2 = calib_data['distCoeffs2']
    R = calib_data['R']
    T = calib_data['T']

    # Debug calibration data
    print("Calibration Data Shapes:")
    print(f"cameraMatrix1: {cameraMatrix1.shape}")
    print(f"distCoeffs1: {distCoeffs1.shape}")
    print(f"R: {R.shape}")
    print(f"T: {T.shape}")

    # Try undistortion first without rectification
    undist_left = cv2.undistort(img_left, cameraMatrix1, distCoeffs1)
    undist_right = cv2.undistort(img_right, cameraMatrix2, distCoeffs2)
    
    # Save undistorted images
    cv2.imwrite(os.path.join(output_dir, 'undistorted_left.jpg'), undist_left)
    cv2.imwrite(os.path.join(output_dir, 'undistorted_right.jpg'), undist_right)
    
    print(f"Undistorted left range: [{np.min(undist_left)} - {np.max(undist_left)}]")
    print(f"Undistorted right range: [{np.min(undist_right)} - {np.max(undist_right)}]")

    # Compute rectification parameters
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1, distCoeffs1,
        cameraMatrix2, distCoeffs2,
        (w, h), R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0  # Change to 0 for valid rect maps
    )

    # Debug rectification matrices
    print(f"R1 shape: {R1.shape}, R2 shape: {R2.shape}")
    print(f"P1 shape: {P1.shape}, P2 shape: {P2.shape}")
    print(f"ROI1: {roi1}, ROI2: {roi2}")  # Add ROI debug info

    # Compute rectification maps with float32 type
    map1x, map1y = cv2.initUndistortRectifyMap(
        cameraMatrix1, distCoeffs1, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(
        cameraMatrix2, distCoeffs2, R2, P2, (w, h), cv2.CV_32FC1)

    # Verify map ranges
    print(f"Map ranges - x1: [{np.min(map1x)} - {np.max(map1x)}], y1: [{np.min(map1y)} - {np.max(map1y)}]")

    # Use original images instead of undistorted ones for remapping
    rect_left = cv2.remap(img_left, map1x, map1y, 
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0,0,0))
    rect_right = cv2.remap(img_right, map2x, map2y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0,0,0))

    # Debug intermediate results
    print(f"Rect image shapes - left: {rect_left.shape}, right: {rect_right.shape}")
    print(f"Rectified left range: [{np.min(rect_left)} - {np.max(rect_left)}]")
    print(f"Rectified right range: [{np.min(rect_right)} - {np.max(rect_right)}]")

    # Draw horizontal lines for visualization (on BGR images)
    line_interval = 30
    for i in range(0, h, line_interval):
        cv2.line(rect_left, (0, i), (w, i), (0, 255, 0), 1)
        cv2.line(rect_right, (0, i), (w, i), (0, 255, 0), 1)

    # Convert to RGB only once, just before displaying with matplotlib
    rect_left_rgb = cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB)
    rect_right_rgb = cv2.cvtColor(rect_right, cv2.COLOR_BGR2RGB)

    # Create matplotlib figure
    plt.figure(figsize=(15, 5))
    
    # Plot left image
    plt.subplot(121)
    plt.imshow(rect_left_rgb)
    plt.title('Rectified Left Image')
    plt.axis('off')
    
    # Plot right image
    plt.subplot(122)
    plt.imshow(rect_right_rgb)
    plt.title('Rectified Right Image')
    plt.axis('off')
    
    # Save visualization
    plt.savefig(os.path.join(output_dir, 'rectified_pair_matplotlib.png'))
    plt.show()

    # Save individual images in BGR format (no conversion needed)
    cv2.imwrite(os.path.join(output_dir, 'rectified_left.jpg'), rect_left)
    cv2.imwrite(os.path.join(output_dir, 'rectified_right.jpg'), rect_right)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stereo Calibration Visualization')
    parser.add_argument('--left', required=True, help='Left image path')
    parser.add_argument('--right', required=True, help='Right image path')
    parser.add_argument('--calib', required=True, help='Calibration file path')
    parser.add_argument('--output', default='./output', help='Output directory')
    args = parser.parse_args()
    
    visualize_stereo_calibration(args.left, args.right, args.calib, args.output)