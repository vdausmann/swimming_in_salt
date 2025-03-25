# README.md

# PlankTracker3D

PlankTracker3D is a GUI application designed for capturing images and recording videos using the Picamera2 library. The application allows users to set exposure times and process recorded videos to extract frames as images.

## Features

- Capture still images with adjustable exposure time.
- Record videos with specified settings.
- Process recorded videos to extract frames and save them as images.
- User-friendly GUI for easy interaction.

## Project Structure

```
planktracker3D
├── src
│   ├── gui_app.py               # Main entry point for the GUI application
│   ├── picamera2_test.py         # Logic for capturing still images
│   ├── video_processing.py        # Processes recorded videos
│   └── ui
│       ├── main_window.py        # Defines the main window of the GUI
│       └── settings_dialog.py     # Dialog for adjusting settings
├── images                        # Directory for storing captured images and videos
├── requirements.txt              # Lists project dependencies
└── README.md                     # Documentation for the project
```

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