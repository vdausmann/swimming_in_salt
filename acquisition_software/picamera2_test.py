from picamera2 import Picamera2
import time
from datetime import datetime
import cv2
import os
import numpy as np

save_path = '../images'

# Initialize the camera
picam2 = Picamera2()

# Set full resolution for the HQ Cameras
factor = 0.5
full_resolution = (int(4056*factor), int(3040*factor))

# Configure the camera for still image capture
still_config = picam2.create_still_configuration(main={"size": full_resolution})
picam2.configure(still_config)

# Start the camera
picam2.start()

# Desired frame rate
desired_fps = 10
interval = 1 / desired_fps  # Time interval between captures

# Duration for capturing images
capture_duration = 10  # Duration in seconds
end_time = time.time() + capture_duration

while time.time() < end_time:
    # Capture an image
    frame = picam2.capture_array()
    height , width, channels = frame.shape
    
    midpoint = width//2
    left_half = frame[:, :midpoint]
    right_half = frame[:, midpoint:]

    new_h, new_w = 1520, 2*1014
    left_half = cv2.resize(left_half, (new_w, new_h))
    right_half = cv2.resize(right_half, (new_w, new_h))
    
    
    # Generate a timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = os.path.join(save_path, f"frame_{timestamp}")
    
    # Save the frame as an image file
    cv2.imwrite(filename+'_left.jpg', left_half)
    cv2.imwrite(filename+'_right.jpg', right_half)
    print(f"Captured {filename}")

    # Wait to maintain the desired frame rate
    time.sleep(interval)

# Stop the camera
picam2.stop()
