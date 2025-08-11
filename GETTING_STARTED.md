# Track Annotation & Validation Tool - INSTRUCTIONS

## What I've Created for You

I've built a comprehensive Plotly Dash application for annotating and validating your plankton tracking data. Here's what's included:

### 🎯 **Main Application**
- `track_annotation_app.py` - The main Dash web application
- `track_annotation_launcher.py` - Easy-to-use launcher with dependency installation
- `test_track_annotation.py` - Test script to verify everything works

### 📋 **Supporting Files**
- `track_annotation_requirements.txt` - Required Python packages
- `setup_track_annotation.sh` - Automated setup script
- `TRACK_ANNOTATION_README.md` - Detailed documentation

## 🚀 Quick Start

### Step 1: Install Dependencies
```bash
cd /Users/vdausmann/swimming_in_salt
pip3 install -r track_annotation_requirements.txt
```

### Step 2: Test the Installation
```bash
python3 test_track_annotation.py
```

### Step 3: Launch the App
```bash
python3 track_annotation_launcher.py /Users/vdausmann/swimming_in_salt/detection_results/20250312_21.2_01
```

The app will open in your browser at: **http://localhost:8050**

## ✨ Key Features

### 🖼️ **Dual Image Display**
- Side-by-side upper and lower camera views
- Real-time track overlay with distinct colors
- Interactive track selection by clicking

### 🎮 **Navigation Controls**
- Frame slider for quick jumping
- Previous/Next buttons
- Play/Pause for automatic advancement
- Keyboard arrow key support

### ✏️ **Track Editing**
- **Split Track** - Divide a track at current frame
- **Delete Track** - Remove entire track
- **Undo** - Revert last edit operation
- **Export** - Save edited tracks with timestamp

### 📊 **Track Information Panel**
Shows detailed statistics for selected tracks:
- Track ID and camera source
- Duration and frame range  
- Average object area
- Total distance traveled
- Motion pattern classification

## 🎯 **How to Use**

1. **Select a Track**: Click on any colored circle/line in either image
2. **Navigate Frames**: Use slider, buttons, or arrow keys
3. **Edit Tracks**:
   - To split: Select track → Navigate to split point → Click "Split Track"
   - To delete: Select track → Click "Delete Track"
   - To undo: Click "Undo Last Edit"
4. **Export Results**: Click "Export Tracks" to save your changes

## 🔧 **Technical Details**

### Input Requirements
Your data directory must contain:
- `upper_tracks.csv` - Tracking results for upper camera
- `lower_tracks.csv` - Tracking results for lower camera
- `upper_*.png` - Upper camera image sequence
- `lower_*.png` - Lower camera image sequence

### Output Files
Edited tracks are exported as:
- `upper_tracks_edited_YYYYMMDD_HHMMSS.csv`
- `lower_tracks_edited_YYYYMMDD_HHMMSS.csv`

### Dependencies
- dash (web framework)
- plotly (interactive visualizations)
- pandas (data handling)
- numpy (numerical operations)
- pillow (image processing)
- opencv-python (computer vision)

## 🛠️ **Troubleshooting**

### "Module not found" errors
```bash
pip3 install -r track_annotation_requirements.txt
```

### App won't start
- Check port 8050 isn't in use
- Try different port: `python3 track_annotation_launcher.py [data_dir] 8051`

### No tracks visible
- Verify CSV files have correct format
- Check image files exist and match CSV frame indices

## 🎨 **Advanced Features**

### Track Visualization
- **Color coding**: Each track gets a unique HSV-generated color
- **Trajectory lines**: Show complete path history
- **Selection highlighting**: Yellow border for selected tracks
- **Hover information**: Details on mouse-over

### Editing Workflow
- All operations are tracked for undo functionality
- Real-time updates to visualization
- Timestamped exports preserve original data

### Performance Optimizations
- Efficient image loading and caching
- Optimized track rendering for large datasets
- Responsive UI with smooth frame transitions

## 📈 **Integration with Your Pipeline**

This tool is designed to work seamlessly with your existing tracking workflow:

1. **Run Detection**: Use your object detection on image sequences
2. **Generate Tracks**: Run `tracking_matching_objs.py` to create track CSVs
3. **Annotate & Validate**: Use this tool to review and edit tracks
4. **Export Results**: Save corrected tracks for further analysis

The tool maintains full compatibility with your existing data formats and can be integrated into your PISCO analysis pipeline.

## 🎉 **Ready to Go!**

Your track annotation tool is ready! The interface is intuitive and powerful, allowing you to efficiently review and correct tracking results from your stereo camera setup.

Happy tracking! 🔬✨
