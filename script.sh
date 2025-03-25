#!/bin/bash

# Record a video for 5 seconds
raspicam-vid -o video.h264 -t 5000 -fps 30

# Convert the video to MP4 format
MP4Box -add video.h264 video.mp4

# Extract frames from the video
ffmpeg -i video.mp4 -vf fps=1 images/image_%04d.jpg

# Clean up
rm video.h264 video.mp4