# README.md

# Swimming in Salt Project Repo

The software library contains all codes needed for the recording and analysis of images taken using the stereo camera device with the preliminary name "PlankTracker3D".

# Capturing Images

Images can be captured using the GUI App written in python. The GUI application designed for capturing images and recording videos using the Picamera2 library. The application allows users to set exposure times and process recorded videos to extract frames as images. It can be started via the shell script planktracker3d_app.sh. The applies the splitting of left and right (upper and lower) images and restores the correct image aspect ratio in the desired resolution. So far the resolution can only be changed in the python code for the app by changing the scale_factor variable.

A **second** way of capturing images is the python script "record_calibration_images.py". This will open a direct low-res live stream from the two cameras (undistorted). Hitting the space bar will record a single image from both cameras and save it a given location (in the correct aspect ratio). Hit "q" to end the stream.

There a more ways to record images or videos using the rpicam-apps (https://github.com/raspberrypi/rpicam-apps?tab=readme-ov-file) from the command line. A recorded video can be split into single images structured the same way as the GUI_app using the script video_processing.

# Calibration of the cameras

All code nescessary for the calibration of the stero camera system can be found in the calibration folder. Docs will be added.

# Analysis of recorded images

So far the script "analyze_stereo_image_dir.py" will: 
- apply the calibrations that were gathered during calibration
- detect all objects in the combined field of view of both cameras
- calculate tracks of objects moving in the field of view
- try to match tracks from the upper and lower images in order to calculate the disparity of the tracked objects which will allow us to calculate the depth in the volume.
- output a dataframe that can be used to calculate velocities (in xy dimensions in umatched tracks and in xyz dimensions for matched tracks)


## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd planktracker3D
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```
   python src/gui_app.py
   ```

2. Use the GUI to capture images or record videos. Adjust the exposure time as needed through the settings dialog.

3. Process recorded videos using the video processing functionality to extract frames.

## Dependencies

This project requires the following Python libraries:
- OpenCV
- Picamera2
- [Any GUI framework used, e.g., Tkinter or PyQt]

## License

This project is licensed under the MIT License - see the LICENSE file for details.