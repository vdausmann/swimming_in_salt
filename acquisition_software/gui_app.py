from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QMessageBox
import sys
import os
import json
from picamera2 import Picamera2
import cv2
from datetime import datetime
import time

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Plank Tracker 3D")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()
        self.picam2 = Picamera2()
        self.exposure_time = 10000  # Default exposure time in microseconds
        self.duration = 10  # Default duration in seconds
        self.fps = 10  # Default frames per second
        self.output_dir = 'images'  # Default output directory

        # Set full resolution for the HQ Cameras
        factor = 1
        self.x_res = int(4056 * factor)
        self.y_res = int(3040 * factor)
        self.full_resolution = (self.x_res, self.y_res)

    def initUI(self):
        self.sample_name_label = QtWidgets.QLabel("Sample Name:", self)
        self.sample_name_label.setGeometry(50, 20, 100, 30)

        self.sample_name_input = QtWidgets.QLineEdit(self)
        self.sample_name_input.setGeometry(160, 20, 200, 30)

        self.capture_button = QtWidgets.QPushButton("Capture Images", self)
        self.capture_button.setGeometry(50, 70, 200, 50)
        self.capture_button.clicked.connect(self.capture_images)

        self.settings_button = QtWidgets.QPushButton("Settings", self)
        self.settings_button.setGeometry(50, 140, 200, 50)
        self.settings_button.clicked.connect(self.open_settings)

    def capture_images(self):
        sample_name = self.sample_name_input.text()
        if not sample_name:
            QMessageBox.warning(self, "Error", "Sample name is required.")
            return

        output_dir = os.path.join(self.output_dir, sample_name)
        try:
            os.makedirs(output_dir, exist_ok=False)
        except FileExistsError:
            QMessageBox.warning(self, "Error", f"Sample {sample_name} already exists.")
            return

        # Save settings as metadata
        metadata = {
            "exposure_time": self.exposure_time,
            "duration": self.duration,
            "fps": self.fps,
            "output_dir": self.output_dir
        }
        metadata_path = os.path.join(self.output_dir, sample_name, f"{sample_name}_metadata.json")
        with open(metadata_path, 'w') as metadata_file:
            json.dump(metadata, metadata_file)
        print(f"Metadata saved as {metadata_path}")

        # Configure the camera for still image capture
        still_config = self.picam2.create_still_configuration(main={"size": self.full_resolution})
        self.picam2.configure(still_config)

        self.picam2.start()
        self.picam2.set_controls({"ExposureTime": self.exposure_time})

        desired_fps = self.fps
        interval = 1 / desired_fps  # Time interval between captures
        capture_duration = self.duration  # Duration in seconds
        end_time = time.time() + capture_duration

        while time.time() < end_time:
            frame = self.picam2.capture_array()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            save_path = os.path.join(output_dir, f"frame_{timestamp}.jpg")
            cv2.imwrite(save_path, frame)
            print(f"Captured {save_path}")

            time.sleep(interval)

        self.picam2.stop()
        #QMessageBox.information(self, "Success", f"Images saved in {output_dir}")
        print(f"Images captured and saved in {output_dir}")

        self.process_images(output_dir)

    def process_images(self, output_dir):
        left_dir = os.path.join(output_dir, "left")
        right_dir = os.path.join(output_dir, "right")
        os.makedirs(left_dir, exist_ok=True)
        os.makedirs(right_dir, exist_ok=True)

        for filename in os.listdir(output_dir):
            if filename.endswith(".jpg"):
                file_path = os.path.join(output_dir, filename)
                frame = cv2.imread(file_path)
                height, width, channels = frame.shape

                midpoint = width // 2
                left_half = frame[:, :midpoint]
                right_half = frame[:, midpoint:]

                new_h, new_w = int(self.y_res/2), int(self.x_res/2)
                left_half = cv2.resize(left_half, (new_w, new_h))
                right_half = cv2.resize(right_half, (new_w, new_h))

                left_path = os.path.join(left_dir, filename.replace(".jpg", "_left.jpg"))
                right_path = os.path.join(right_dir, filename.replace(".jpg", "_right.jpg"))

                cv2.imwrite(left_path, left_half)
                cv2.imwrite(right_path, right_half)
                print(f"Processed {file_path} into {left_path} and {right_path}")

                os.remove(file_path)

        QMessageBox.information(self, "Success", f"Images processed and saved in {left_dir} and {right_dir}")
        print(f"Images processed and saved in {left_dir} and {right_dir}")

    def open_settings(self):
        self.settings_dialog = SettingsDialog(self)
        self.settings_dialog.show()

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setGeometry(150, 150, 300, 300)

        self.exposure_label = QtWidgets.QLabel("Exposure Time (µs):", self)
        self.exposure_label.setGeometry(10, 10, 150, 30)

        self.exposure_input = QtWidgets.QLineEdit(self)
        self.exposure_input.setGeometry(160, 10, 100, 30)
        self.exposure_input.setText(str(parent.exposure_time))

        self.duration_label = QtWidgets.QLabel("Duration (s):", self)
        self.duration_label.setGeometry(10, 50, 150, 30)

        self.duration_input = QtWidgets.QLineEdit(self)
        self.duration_input.setGeometry(160, 50, 100, 30)
        self.duration_input.setText(str(parent.duration))

        self.fps_label = QtWidgets.QLabel("FPS:", self)
        self.fps_label.setGeometry(10, 90, 150, 30)

        self.fps_input = QtWidgets.QLineEdit(self)
        self.fps_input.setGeometry(160, 90, 100, 30)
        self.fps_input.setText(str(parent.fps))

        self.output_dir_label = QtWidgets.QLabel("Output Directory:", self)
        self.output_dir_label.setGeometry(10, 130, 150, 30)

        self.output_dir_input = QtWidgets.QLineEdit(self)
        self.output_dir_input.setGeometry(160, 130, 100, 30)
        self.output_dir_input.setText(parent.output_dir)

        self.apply_button = QtWidgets.QPushButton("Apply", self)
        self.apply_button.setGeometry(10, 170, 250, 30)
        self.apply_button.clicked.connect(self.apply_settings)

    def apply_settings(self):
        parent = self.parent()
        parent.exposure_time = int(self.exposure_input.text())
        parent.duration = int(self.duration_input.text())
        parent.fps = int(self.fps_input.text())
        parent.output_dir = self.output_dir_input.text()
        QMessageBox.information(self, "Success", "Settings applied.")
        self.close()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())