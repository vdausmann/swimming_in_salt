import argparse
import cv2
import numpy as np
import glob
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Stereo Camera Calibration')
    parser.add_argument('--image_dir', type=str, default='../calibration_images/stereo',
                       help='Directory containing stereo image pairs')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='Directory to save calibration results')
    parser.add_argument('--chess_rows', type=int, default=8,
                       help='Number of inner corners rows in chessboard')
    parser.add_argument('--chess_cols', type=int, default=11,
                       help='Number of inner corners columns in chessboard')
    parser.add_argument('--save_images', type=bool, default=True,
                       help='Save rectified image pairs')
    return parser.parse_args()

def stereo_calibrate(args):
    # Convert relative paths to absolute
    image_dir = os.path.abspath(args.image_dir)
    output_dir = os.path.abspath(args.output_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create directory for rectified images if needed
    rectified_dir = os.path.join(output_dir, 'rectified_images')
    if args.save_images:
        os.makedirs(rectified_dir, exist_ok=True)
    
    # Load calibration files - note that what was previously "left" is actually "upper" camera
    # and what was previously "right" is actually "lower" camera
    upper_calib = np.load('Rcalib.npz')  
    lower_calib = np.load('Lcalib.npz')  
    
    mtx_upper = upper_calib['camera_matrix']
    dist_upper = upper_calib['distortion_coefficients']
    mtx_lower = lower_calib['camera_matrix']
    dist_lower = lower_calib['distortion_coefficients']
    rvecs_l = lower_calib['rotation_vectors']
    tvecs_l = lower_calib['translation_vectors']
    rvecs_u = upper_calib['rotation_vectors']
    tvecs_u = upper_calib['translation_vectors']

    
    # Prepare object points
    chessboard_size = (args.chess_rows, args.chess_cols)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1,2)
    
    # Arrays to store object points and image points
    objpoints = []         # 3D points in real world space
    upper_imgpoints = []   # 2D points in upper image plane (previously left)
    lower_imgpoints = []   # 2D points in lower image plane (previously right)
    
    # Get list of image pairs - folder structure may need to be updated if you've 
    # organized images as upper/lower instead of left/right
    upper_pattern = os.path.join(image_dir, 'right', '*.jpg')  # Update folder name if needed
    lower_pattern = os.path.join(image_dir, 'left', '*.jpg') # Update folder name if needed
    upper_images = sorted(glob.glob(upper_pattern))
    lower_images = sorted(glob.glob(lower_pattern))
    
    print(f"Found {len(upper_images)} image pairs")
    
    # Process each image pair
    for upper_img_file, lower_img_file in zip(upper_images, lower_images):
        # Read images
        upper_img = cv2.imread(upper_img_file)
        if upper_img is None:
            print(f"Failed to load image: {upper_img_file}")
            continue
        lower_img = cv2.imread(lower_img_file)
        if lower_img is None:
            print(f"Failed to load image: {lower_img_file}")
            continue
        
        # Convert to grayscale
        upper_gray = cv2.cvtColor(upper_img, cv2.COLOR_BGR2GRAY)
        lower_gray = cv2.cvtColor(lower_img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        upper_ret, upper_corners = cv2.findChessboardCorners(upper_gray, chessboard_size, None)
        lower_ret, lower_corners = cv2.findChessboardCorners(lower_gray, chessboard_size, None)
        
        if upper_ret and lower_ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            upper_corners = cv2.cornerSubPix(upper_gray, upper_corners, (11,11), (-1,-1), criteria)
            lower_corners = cv2.cornerSubPix(lower_gray, lower_corners, (11,11), (-1,-1), criteria)
            
            objpoints.append(objp)
            upper_imgpoints.append(upper_corners)
            lower_imgpoints.append(lower_corners)
            
            print(f"Processed pair: {upper_img_file} and {lower_img_file}")
        
    cv2.destroyAllWindows()
    
    print("Number of object points:", len(objpoints))
    print("Number of image points:", len(upper_imgpoints))
    
    # Stereo calibration
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, upper_imgpoints, lower_imgpoints,
        mtx_upper, dist_upper,
        mtx_lower, dist_lower,
        upper_gray.shape[::-1], criteria=criteria_stereo, flags=flags)
    
    print(f"Stereo calibration RMS error: {retval}")
    
    # Save calibration results with clearer naming
    output_file = os.path.join(output_dir, 'stereo_calibration.npz')
    np.savez(output_file,
             cameraMatrixUpper=cameraMatrix1, distCoeffsUpper=distCoeffs1,
             cameraMatrixLower=cameraMatrix2, distCoeffsLower=distCoeffs2,
             R=R, T=T, E=E, F=F)
    
    print(f"Calibration completed and saved to '{output_file}'")
    print("Note: Camera 1 is the UPPER camera, Camera 2 is the LOWER camera")
    
    # Compute stereo rectification
    image_size = upper_gray.shape[::-1]
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        cameraMatrix1, distCoeffs1,
        cameraMatrix2, distCoeffs2,
        image_size, R, T,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=1)
    
    # Calculate undistortion and rectification maps
    map1_x, map1_y = cv2.initUndistortRectifyMap(
        cameraMatrix1, distCoeffs1, R1, P1, image_size, cv2.CV_32FC1)
    map2_x, map2_y = cv2.initUndistortRectifyMap(
        cameraMatrix2, distCoeffs2, R2, P2, image_size, cv2.CV_32FC1)
    
    # Save rectification maps
    np.savez(os.path.join(output_dir, 'stereo_rectification.npz'),
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             map1_x=map1_x, map1_y=map1_y,
             map2_x=map2_x, map2_y=map2_y)
    
    alignment_verror, alignment_herror = measure_rectification_alignment(
        upper_imgpoints, lower_imgpoints, map1_x, map1_y, map2_x, map2_y
    )
    print(f"Average vertical alignment error: {alignment_verror}")
    print(f"Average horizontal alignment error: {alignment_herror}")
    
    return (cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F,
            objpoints, upper_imgpoints, lower_imgpoints, rvecs_l, tvecs_l, rvecs_u, tvecs_u)

def measure_rectification_alignment(upper_imgpoints, lower_imgpoints, map1_x, map1_y, map2_x, map2_y):
    total_vertical_diff = 0
    total_horizontal_diff = 0
    total_points = 0
    
    for pts_u, pts_l in zip(upper_imgpoints, lower_imgpoints):
        for (u_x, u_y), (l_x, l_y) in zip(pts_u.reshape(-1, 2), pts_l.reshape(-1, 2)):
            u_map_x = map1_x[int(u_y), int(u_x)]
            u_map_y = map1_y[int(u_y), int(u_x)]
            l_map_x = map2_x[int(l_y), int(l_x)]
            l_map_y = map2_y[int(l_y), int(l_x)]
            
            total_vertical_diff += abs(u_map_y - l_map_y)
            total_horizontal_diff += abs(u_map_x - l_map_x)
            total_points += 1

    if total_points == 0:
        return 0, 0
    return (total_vertical_diff / total_points,
            total_horizontal_diff / total_points)

def compute_stereo_reprojection_error(objpoints, upper_imgpoints, lower_imgpoints,
                                      cameraMatrix1, distCoeffs1,
                                      cameraMatrix2, distCoeffs2,
                                      R, T):
    # The upper camera is assumed to have R=I and T=0
    rvec_id, _ = cv2.Rodrigues(np.eye(3))
    tvec_id = np.zeros((3, 1), dtype=np.float64)

    # Convert stereo R matrix to a rotation vector for the lower camera
    rvec_lower, _ = cv2.Rodrigues(R)

    total_error = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        # Project points into the upper camera
        projected_upper, _ = cv2.projectPoints(objpoints[i], rvec_id, tvec_id,
                                               cameraMatrix1, distCoeffs1)
        # Compare with detected corners
        error_upper = cv2.norm(upper_imgpoints[i], projected_upper, cv2.NORM_L2)

        # Project points into the lower camera
        projected_lower, _ = cv2.projectPoints(objpoints[i], rvec_lower, T,
                                               cameraMatrix2, distCoeffs2)
        # Compare with detected corners
        error_lower = cv2.norm(lower_imgpoints[i], projected_lower, cv2.NORM_L2)

        total_error += error_upper + error_lower
        total_points += (len(projected_upper) + len(projected_lower))

    mean_error = total_error / total_points if total_points else 0
    return mean_error

def compute_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    total_points = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        total_points += len(imgpoints2)
    mean_error = total_error / total_points
    return mean_error

if __name__ == "__main__":
    args = parse_args()
    (cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F,
     objpoints, upper_imgpoints, lower_imgpoints, rvecs_l, tvecs_l, rvecs_u, tvecs_u) = stereo_calibrate(args)
    
    # # Compute stereo reprojection error
    # stereo_error = compute_stereo_reprojection_error(
    #     objpoints,      # same as used in stereoCalibrate
    #     upper_imgpoints,
    #     lower_imgpoints,
    #     cameraMatrix1, distCoeffs1,
    #     cameraMatrix2, distCoeffs2,
    #     R, T
    # )
    # print(f"Stereo reprojection error: {stereo_error}")

    # Calculate and print the left and right camera reprojection errors
    print("Upper camera reprojection error: ", compute_reprojection_errors(objpoints, upper_imgpoints, rvecs_l, tvecs_l, cameraMatrix1, distCoeffs1))
    print("Right camera reprojection error: ", compute_reprojection_errors(objpoints, lower_imgpoints, rvecs_u, tvecs_u, cameraMatrix2, distCoeffs2))