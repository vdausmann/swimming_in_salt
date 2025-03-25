import cv2
import os
import numpy as np
from datetime import datetime

# Path to the video file
video_path = 'test_video.mjpeg'
save_path = 'images'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# Desired frame rate
desired_fps = 10
interval = 1 / desired_fps  # Time interval between captures

# Get the original frame rate of the video
original_fps = cap.get(cv2.CAP_PROP_FPS)
print(original_fps)
frame_interval = int(original_fps / desired_fps)

# Set full resolution for the HQ Cameras
#factor = 0.5
#full_resolution = (int(4056*factor), int(3040*factor))

frame_count = 0

while True:
    # Read a frame from the video
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video

    # Process every nth frame to match the desired FPS
    if frame_count % frame_interval == 0:
        height, width, channels = frame.shape
        print(frame.shape)
        
        midpoint = width // 2
        left_half = frame[:, :midpoint]
        right_half = frame[:, midpoint:]

        new_h, new_w = height//2, width//2
        left_half = cv2.resize(left_half, (new_w, new_h))
        right_half = cv2.resize(right_half, (new_w, new_h))

        # Generate a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = os.path.join(save_path, f"frame_{timestamp}")

        # Save the frame as an image file
        cv2.imwrite(filename + '_left.jpg', left_half)
        cv2.imwrite(filename + '_right.jpg', right_half)
        print(f"Captured {filename}")

    frame_count += 1

# Release the video capture object
cap.release()
