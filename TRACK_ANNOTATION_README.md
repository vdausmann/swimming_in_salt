# 🎯 Track Annotation Tool - Usage Instructions

## 📋 Prerequisites

You need to use your **cv conda environment** which should already have:

- Python 3.x
- pandas
- numpy
- opencv (cv2)

## 🚀 Quick Start

### 1. Install Additional Packages

```bash
# Activate your cv environment first
conda activate cv

# Install the web app dependencies
pip install dash plotly pillow
```

### 2. Test the Installation

```bash
# Test with your cv Python
~/miniforge3/envs/cv/bin/python test_track_annotation.py /Users/vdausmann/swimming_in_salt/detection_results/20250312_21.2_01
```

### 3. Launch the Application

```bash
# Launch with your cv Python
~/miniforge3/envs/cv/bin/python track_annotation_launcher.py /Users/vdausmann/swimming_in_salt/detection_results/20250312_21.2_01
```

### 4. Open in Browser

- The app will start on `http://localhost:8050`
- Open this URL in your web browser

## 🎛️ Using the Tool

### Navigation

- **Frame Slider**: Move between frames
- **Arrow Buttons**: First/Previous/Next/Last frame
- **Keyboard**: Use arrow keys for frame navigation (when focused on the slider)

### Track Selection

- **Click on any track marker** (colored dots) to select it
- Selected tracks are highlighted with larger markers and white borders
- Track information appears in the bottom panel

### Editing Operations

- **Split Track**: Splits the selected track at the current frame
- **Delete Track**: Completely removes the selected track
- **Undo**: Reverses the last edit operation
- **Export**: Saves edited tracks with timestamp

### Visual Features

- **Side-by-side stereo display**: Upper and lower camera views
- **Color-coded tracks**: Each track has a unique color
- **Track trajectories**: See movement patterns over time
- **Hover information**: Detailed data on mouse hover

## 📁 Data Requirements

Your data directory should contain:

- `upper_tracks.csv` - Upper camera tracking results
- `lower_tracks.csv` - Lower camera tracking results
- `upper_*.png` - Upper camera image sequence
- `lower_*.png` - Lower camera image sequence

## 🔧 Troubleshooting

### Environment Issues

```bash
# Check if you're in the right environment
which python
# Should show: /Users/vdausmann/miniforge3/envs/cv/bin/python

# If not, activate the environment
conda activate cv
```

### Missing Packages

```bash
# Install missing packages in cv environment
conda activate cv
pip install dash plotly pillow
```

### Data Loading Issues

```bash
# Test data loading first
~/miniforge3/envs/cv/bin/python test_track_annotation.py <your_data_directory>
```

## 💡 Tips for Effective Annotation

1. **Start with problematic tracks**: Focus on tracks that switch identities or have gaps
2. **Use the frame-by-frame navigation**: Carefully examine critical transition points
3. **Split before merging**: Split problematic tracks first, then you can manually merge later
4. **Save frequently**: Use the Export button to save your progress
5. **Use Undo liberally**: Don't hesitate to undo and retry edits

## 🎯 Workflow Example

1. Load your tracking results
2. Navigate to a frame where you see track switching
3. Click on the problematic track to select it
4. Use Split Track to separate the incorrectly merged segments
5. Continue through frames to validate the split
6. Export your corrected tracks when done

The exported files will have timestamps and can be used in your analysis pipeline.
