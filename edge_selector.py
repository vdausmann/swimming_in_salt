import cv2
import numpy as np
import os

# Global variables to store line coordinates
lines_upper = []
lines_lower = []
current_image = None
labels = ["Front Left", "Front Right", "Back Left", "Back Right"]

def draw_line(event, x, y, flags, param):
    global lines_upper, lines_lower, current_image

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(lines_upper) < 4:
            lines_upper.append(x)
            print(f"Upper Image: Selected {labels[len(lines_upper) - 1]} edge at x = {x}")
        elif len(lines_lower) < 4:
            lines_lower.append(x)
            print(f"Lower Image: Selected {labels[len(lines_lower) - 1]} edge at x = {x}")
        else:
            print("Already have 4 edges for each image")

def manually_define_edges(upper_image_path, lower_image_path, output_file, map1_x, map1_y, map2_x, map2_y):
    global lines_upper, lines_lower, current_image

    # Load images
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
        # Display stacked image
        display_image = stacked_image.copy()

        # Draw lines on the image for verification
        for i, x in enumerate(lines_upper):
            cv2.line(display_image, (x, 0), (x, upper_image.shape[0]), (0, 255, 0), 2)
            cv2.putText(display_image, labels[i], (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        for i, x in enumerate(lines_lower):
            cv2.line(display_image, (x, upper_image.shape[0]), (x, stacked_image.shape[0]), (0, 255, 0), 2)
            cv2.putText(display_image, labels[i], (x, upper_image.shape[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow('Stacked Image', display_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

    # Save line coordinates to file
    np.savez(output_file, lines_upper=lines_upper, lines_lower=lines_lower)
    print(f"Saved line coordinates to {output_file}")

# Example usage
if __name__ == "__main__":
    # Load rectification maps
    rectification_file = "/Users/vdausmann/oyster_project/planktracker3D/calibration/stereo_rectification.npz"
    rect_data = np.load(rectification_file)
    map1_x, map1_y = rect_data['map1_x'], rect_data['map1_y']
    map2_x, map2_y = rect_data['map2_x'], rect_data['map2_y']

    manually_define_edges(
        upper_image_path="/Users/vdausmann/oyster_project/images/test_sample2/upper/frame_20250307_113354_916157_left.jpg",
        lower_image_path="/Users/vdausmann/oyster_project/images/test_sample2/lower/frame_20250307_113354_916157_right.jpg",
        output_file="/Users/vdausmann/oyster_project/planktracker3D/calibration/line_coordinates.npz",
        map1_x=map1_x,
        map1_y=map1_y,
        map2_x=map2_x,
        map2_y=map2_y
    )