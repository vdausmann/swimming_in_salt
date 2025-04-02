import cv2
import numpy as np
import os

# Global variables to store line coordinates
vertical_lines = []  # For left and right boundaries
horizontal_lines_upper = []  # For upper image boundaries
horizontal_lines_lower = []  # For lower image boundaries
current_phase = 'upper'  # 'upper', 'lower', or 'vertical'
current_image = None

def draw_line(event, x, y, flags, param):
    global vertical_lines, horizontal_lines_upper, horizontal_lines_lower, current_phase

    if event == cv2.EVENT_LBUTTONDOWN:
        if current_phase == 'upper':
            if len(horizontal_lines_upper) < 2 and y < stacked_image.shape[0]//2:
                horizontal_lines_upper.append(y)
                phase_name = "upper boundary" if len(horizontal_lines_upper) == 1 else "lower boundary"
                print(f"Selected upper image {phase_name} at y = {y}")
                if len(horizontal_lines_upper) == 2:
                    current_phase = 'lower'
                    print("\nNow select the boundaries in the lower image")
        elif current_phase == 'lower':
            if len(horizontal_lines_lower) < 2 and y > stacked_image.shape[0]//2:
                y = y - stacked_image.shape[0]//2  # Convert to lower image coordinates
                horizontal_lines_lower.append(y)
                phase_name = "upper boundary" if len(horizontal_lines_lower) == 1 else "lower boundary"
                print(f"Selected lower image {phase_name} at y = {y}")
                if len(horizontal_lines_lower) == 2:
                    current_phase = 'vertical'
                    print("\nNow select the left and right boundaries")
        elif current_phase == 'vertical':
            if len(vertical_lines) < 2:
                vertical_lines.append(x)
                side = "left" if len(vertical_lines) == 1 else "right"
                print(f"Selected {side} boundary at x = {x}")

def manually_define_roi(upper_image_path, lower_image_path, output_file, map1_x, map1_y, map2_x, map2_y):
    global vertical_lines, horizontal_lines_upper, horizontal_lines_lower, current_phase, current_image, stacked_image

    # Load and process images
    upper_image = cv2.imread(upper_image_path)
    lower_image = cv2.imread(lower_image_path)

    # Undistort and rectify images
    upper_image = cv2.remap(upper_image, map1_x, map1_y, cv2.INTER_LINEAR)
    lower_image = cv2.remap(lower_image, map2_x, map2_y, cv2.INTER_LINEAR)

    # Rotate images 90° counterclockwise
    upper_image = cv2.rotate(upper_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    lower_image = cv2.rotate(lower_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Stack images vertically
    stacked_image = np.vstack([upper_image, lower_image])

    # Create window and set mouse callback
    cv2.namedWindow('Stacked Image')
    cv2.setMouseCallback('Stacked Image', draw_line)

    while True:
        display_image = stacked_image.copy()

        # Draw horizontal lines for upper image
        for i, y in enumerate(horizontal_lines_upper):
            cv2.line(display_image, (0, y), (display_image.shape[1], y), (0, 255, 0), 2)
            label = "Upper image upper boundary" if i == 0 else "Upper image lower boundary"
            cv2.putText(display_image, label, (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw horizontal lines for lower image
        for i, y in enumerate(horizontal_lines_lower):
            display_y = y + stacked_image.shape[0]//2
            cv2.line(display_image, (0, display_y), (display_image.shape[1], display_y), (0, 255, 0), 2)
            label = "Lower image upper boundary" if i == 0 else "Lower image lower boundary"
            cv2.putText(display_image, label, (10, display_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Draw vertical lines
        for x in vertical_lines:
            cv2.line(display_image, (x, 0), (x, display_image.shape[0]), (0, 255, 0), 2)
            label = "Left boundary" if vertical_lines.index(x) == 0 else "Right boundary"
            cv2.putText(display_image, label, (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Add instructions
        if current_phase == 'upper':
            instruction = "Select upper image upper boundary" if len(horizontal_lines_upper) == 0 else "Select upper image lower boundary"
        elif current_phase == 'lower':
            instruction = "Select lower image upper boundary" if len(horizontal_lines_lower) == 0 else "Select lower image lower boundary"
        else:
            instruction = "Select left boundary" if len(vertical_lines) == 0 else "Select right boundary"
        cv2.putText(display_image, instruction, (10, display_image.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow('Stacked Image', display_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or (len(horizontal_lines_upper) == 2 and len(horizontal_lines_lower) == 2 and len(vertical_lines) == 2):
            break

    cv2.destroyAllWindows()

    # Save ROI coordinates to file
    np.savez(output_file, 
             horizontal_lines_upper=np.array(horizontal_lines_upper),
             horizontal_lines_lower=np.array(horizontal_lines_lower),
             vertical_lines=np.array(vertical_lines))
    print(f"\nSaved ROI coordinates to {output_file}")

if __name__ == "__main__":
    # Load rectification maps
    rectification_file = "/Users/vdausmann/oyster_project/planktracker3D/calibration/stereo_rectification.npz"
    rect_data = np.load(rectification_file)
    map1_x, map1_y = rect_data['map1_x'], rect_data['map1_y']
    map2_x, map2_y = rect_data['map2_x'], rect_data['map2_y']

    manually_define_roi(
        upper_image_path="/Users/vdausmann/oyster_project/images/test_sample2/upper/frame_20250307_113354_916157_left.jpg",
        lower_image_path="/Users/vdausmann/oyster_project/images/test_sample2/lower/frame_20250307_113354_916157_right.jpg",
        output_file="/Users/vdausmann/oyster_project/planktracker3D/calibration/roi_coordinates.npz",
        map1_x=map1_x,
        map1_y=map1_y,
        map2_x=map2_x,
        map2_y=map2_y
    )