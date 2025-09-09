from picamera2 import Picamera2, Preview
import cv2
import os
from datetime import datetime

# Path to save calibration images
save_path = '/home/vdausmann/planktracker3D/calibration_images'
os.makedirs(save_path, exist_ok=True)

# Initialize the camera
picam2 = Picamera2()

# Set full resolution for the HQ Cameras
full_resolution = (4056, 3040)

# Configure the camera for still image capture
still_config = picam2.create_still_configuration(main={"size": full_resolution})
picam2.configure(still_config)

# Set fixed exposure time (in microseconds)
fixed_exposure_time = 60000  # Example: 10000 microseconds (10 milliseconds)
picam2.set_controls({"ExposureTime": fixed_exposure_time})
picam2.set_controls({"AnalogueGain": 1.0})

# Start the camera
picam2.start()

# Create a named window for the preview
cv2.namedWindow('Preview', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Preview', 1280, 720)

print("Press the space bar to capture an image. Press 'q' to quit.")

while True:
    # Capture a frame for preview
    frame = picam2.capture_array()

    # Resize the frame for preview
    preview_frame = cv2.resize(frame, (1280, 720))

    # Display the frame in a window
    cv2.imshow('Preview', preview_frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    if key != 255:
        print(f"Key pressed: {key}")  # Debug statement to print key codes

    if key == ord(' '):  # Space bar pressed
        print("Space bar pressed")  # Debug statement
        # Capture an image
        frame = picam2.capture_array()
        height, width, channels = frame.shape

        midpoint = width // 2
        left_half = frame[:, :midpoint]
        right_half = frame[:, midpoint:]

        new_h, new_w = 1520, 2 * 1014
        left_half = cv2.resize(left_half, (new_w, new_h))
        right_half = cv2.resize(right_half, (new_w, new_h))

        # Generate a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_path, f"calib_{timestamp}")

        # Save the frame as an image file
        cv2.imwrite(filename + '_left.jpg', left_half)
        cv2.imwrite(filename + '_right.jpg', right_half)
        print(f"Captured {filename}")

    elif key == ord('q'):  # 'q' key pressed
        print("Q key pressed")  # Debug statement
        break

# Stop the camera and close the preview window
picam2.stop()
cv2.destroyAllWindows()
