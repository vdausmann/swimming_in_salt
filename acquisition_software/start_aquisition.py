import subprocess
import time
import os

def capture_images(num_images, interval, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i in range(num_images):
        # Construct the filename
        filename = os.path.join(output_dir, f"image_{i+1}.jpg")
        
        # Call rpicam-still to capture the image
        command = ["rpicam-jpeg", "-o", filename, "-n"]
        try:
            subprocess.run(command, check=True)
            print(f"Captured {filename}")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while capturing the image: {e}")
        
        # Wait for the specified interval before capturing the next image
        if i < num_images - 1:  # No need to wait after the last image
            time.sleep(interval)

def main():
    # Parameters
    num_images = 5             # Number of images to capture
    interval = .2               # Time between captures in seconds
    output_dir = "../images"    # Directory to save images
    timeout = 0
    
    # Start capturing images
    capture_images(num_images, interval, output_dir)

if __name__ == "__main__":
    main()
