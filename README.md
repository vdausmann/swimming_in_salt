# README.md

# Swimming in Salt Project Repo

The software library contains all files needed for recording and analysis of images taken using the stereo camera device with the preliminary name "PlankTracker3D".

# Capturing Images RASPBERRY PI!

All software for the acquisition of images can be found in the folder "acquisition_software". The software is designed to run on a raspberry pi 4B equipped with the arducam CamArray Hat Kit (https://www.arducam.com/arducam-12mp2-synchronized-stereo-camera-bundle-kit-for-raspberry-pi-two-12-3mp-imx477-camera-modules-with-cs-lens-and-arducam-camarray-stereo-camera-hat.html; https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/quick-start/). Images can be captured using the GUI App written in python. The GUI application designed for capturing images and recording videos using the Picamera2 library. The application allows users to set exposure times and process recorded videos to extract frames as images. It can be started via the shell script planktracker3d_app.sh. The software applies the splitting of left and right (upper and lower) images and restores the correct image aspect ratio in the desired resolution. So far the resolution can only be changed in the python code for the app by changing the scale_factor variable.

A **second** way of capturing images is the python script "record_calibration_images.py". This will open a direct low-res live stream from the two cameras (undistorted). Hitting the space bar will record a single image from both cameras and save it a given location (in the correct aspect ratio). Hit "q" to end the stream.

There a more ways to record images or videos using the rpicam-apps (https://github.com/raspberrypi/rpicam-apps?tab=readme-ov-file) from the command line. 

Recorded video files can be split into single images structured the same way as the GUI_app using the script video_processing.py.


# Analysis of recorded images

## Overview
The software enables users to record, calibrate, and analyze images captured with the stereo camera system. The main workflow consists of three steps:

Camera Calibration
Tracking Pipeline
Track Annotation

Each step is supported by dedicated software modules which are found in the folder tracking_software.

Running the pipeline will create a folder named *"swimming_in_salt_data"* in the same directory as this repository. This will hold the results for each sample.

## 1. Camera Calibration (run_calibration.py) - ONLY IF NEEDED TO BE UPDATED
Purpose:
Calibrates the stereo camera system to ensure accurate measurement and depth estimation.

How it works:

The calibration module uses images of a known pattern (e.g., a checkerboard) to determine the camera parameters.
It computes the relationship between the two cameras, correcting for lens distortion and aligning their views.
This step is essential for accurate 3D tracking.

## 2. Tracking Pipeline (run_tracking_pipeline.py)
Purpose:
Detects and tracks objects in the recorded stereo images.

Usage: 
e.g. for all samples at a given base directory: 
```python run_tracking_pipeline.py --all-samples --base-dir /path/to/images/```


How it works:

The pipeline processes images from both cameras.
It detects objects in each frame and follows their movement over time.
The software matches tracks from both cameras to estimate the 3D position (including depth) of each object.
The main output is a file called upper_ or lower_tracks_rectified.csv, which contains the found tracks, their calculated positions and timestamps.

## 3. Track Annotation (track_annotation_app.py)
Purpose:
Allows users to review and manually annotate the detected tracks.

How it works:

The annotation app provides a graphical interface to inspect tracked objects.
Users can correct or confirm matches, improving the accuracy of the results.

**Output: stereo_matches.csv**
This CSV file contains the results of the tracking pipeline.
Each row represents a matched object track, including its position in 3D space and other relevant data.
The file can be used for further analysis, such as calculating velocities or studying movement patterns.

## Summary
The Swimming in Salt software guides users from camera calibration to object tracking and annotation, producing a detailed record of tracked objects in stereo images. The workflow ensures accurate 3D measurements and provides tools for both automated and manual analysis.


## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd swimming_in_salt
   ```

2. create conda environment and install dependencies:
   
   ```
   conda create --name oyster python=3.8
   conda activate oyster
   ```

   then install dependencies with:
   ```
   conda env update --file environment.yml
   ``` 
  
   
## Usage
**RASPI:**
1. Run the application:
   ```
   python src/gui_app.py
   ```

2. Use the GUI to capture images or record videos. Adjust the exposure time as needed through the settings dialog.

**Analysis Computer:**
0. Calibrate system:
Set configs as needed in the script run_calibration.py:
Example: 
```
CONFIG = {
    # Image directories
    'left_images': '../swimming_in_salt_data/planktracker_2ndcalibration_August2025/small_water/lower/',
    'right_images': '../swimming_in_salt_data/planktracker_2ndcalibration_August2025/small_water/upper/',
    
    # Output directory
    'output_dir': 'calibration_results',
    
    # Calibration parameters
    'chess_rows': 8,
    'chess_cols': 11,
    'square_size': 1.5,  # mm
    'image_pattern': '*.jpg',
    
    # Camera names
    'left_camera_name': 'lower',
    'right_camera_name': 'upper',
    
    # ROI selection
    'use_roi': True,  # Set to False to skip ROI selection
    'roi_margin': 10,  # Additional margin around selected ROI (pixels)
    'reuse_saved_roi': True,  # Set to True to automatically reuse saved ROI
    'roi_save_file': 'saved_roi.npz',  # File to save/load ROI coordinates
    
    # Visualization
    'create_visualizations': True,  # Set to False to skip visualizations
    'max_vis_images': 5,  # Maximum number of images to show in pipeline visualization
}
```
run the calibration, results will be saved in the calibration_results folder 
**TODO: CAREFULL, this will overwrite the current calibration, need to change that new versions of calib can be saved and later selected from.** 
```python run_calibration.py```

1. Run detection and trackiing per sample.

Some general configs can be changed in the script run_tracking_pipeline.py.

To run the whole pipeline on a folder containing several samples:
```python run_tracking_pipeline.py --all-samples --base-dir /path/to/images/```

2. Annotate tracks, create stereo matches manually.

- start the annotation app:
```python track_annotation_app.py```

- connect to http://localhost:8050/ in the browser.

- select sample from dropdown

- scroll through tracks, merge, split, create stereo matching pairs in upper and lower camera.

- [TODO] random selection

- [TODO] annotation of 2D tracks.

- ...

- stereo matches are automatically saved in the results directory of the sample (../swimming_in_salt_data/results) but can be overwritten if need be.


## Dependencies

This project requires the following Python libraries:
- OpenCV
- Picamera2
- [Any GUI framework used, e.g., Tkinter or PyQt]

## License

This project is licensed under the GNU General Public License (GPL)