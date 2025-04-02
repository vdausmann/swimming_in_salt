import numpy as np
import matplotlib.pyplot as plt
import cv2

# Load the disparity map
disparity = cv2.imread('/home/vdausmann/planktracker3D/calibration/output/disparity_calib_20250303_140149_646738_left.jpg', cv2.IMREAD_GRAYSCALE)

# Display the disparity map
plt.imshow(disparity, cmap='gray')
plt.title('Disparity Map')
plt.colorbar()
plt.show()

# Load the depth image
depth = np.load('/home/vdausmann/planktracker3D/calibration/output/depth_calib_20250303_140149_646738_left.jpg.npy')

# Display the depth image (Z coordinate)
plt.imshow(depth[:, :, 2], cmap='jet')
plt.title('Depth Image')
plt.colorbar()
plt.show()