#!/bin/bash

# Track Annotation Tool Setup Script
# Uses the cv conda environment

echo "🎯 Track Annotation Tool Setup"
echo "================================"

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  This script is designed for macOS"
fi

# Set the Python path for cv environment
PYTHON_PATH="$HOME/miniforge3/envs/cv/bin/python"
PIP_PATH="$HOME/miniforge3/envs/cv/bin/pip"

echo "🔍 Checking cv conda environment..."

# Check if the Python path exists
if [ ! -f "$PYTHON_PATH" ]; then
    echo "❌ cv conda environment not found at: $PYTHON_PATH"
    echo "💡 Make sure you have created the cv environment:"
    echo "   conda create -n cv python=3.9"
    echo "   conda activate cv"
    exit 1
fi

echo "✅ Found cv environment: $PYTHON_PATH"

# Test basic functionality
echo "🧪 Testing basic functionality..."
$PYTHON_PATH -c "import pandas, numpy; print('✅ Basic packages available')" || {
    echo "❌ Basic packages missing in cv environment"
    echo "💡 Install basic packages:"
    echo "   conda activate cv"
    echo "   conda install pandas numpy opencv"
    exit 1
}

# Install required packages for the app
echo "📦 Installing Dash application packages..."
$PIP_PATH install dash plotly pillow || {
    echo "❌ Failed to install packages"
    echo "💡 Try manually:"
    echo "   conda activate cv"
    echo "   pip install dash plotly pillow"
    exit 1
}

echo "✅ Package installation completed"

# Make scripts executable
chmod +x test_track_annotation.py
chmod +x track_annotation_launcher.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Test the installation:"
echo "   $PYTHON_PATH test_track_annotation.py [your_data_directory]"
echo ""
echo "2. Launch the application:"
echo "   $PYTHON_PATH track_annotation_launcher.py [your_data_directory]"
echo ""
echo "Example with your data:"
echo "   $PYTHON_PATH track_annotation_launcher.py /Users/vdausmann/swimming_in_salt/detection_results/20250312_21.2_01"

# #!/bin/bash

# # Track Annotation App Setup and Run Script

# echo "Track Annotation & Validation Tool Setup"
# echo "========================================"

# # Check if Python is available
# if ! command -v python3 &> /dev/null; then
#     echo "Error: Python 3 is required but not installed."
#     echo "Please install Python 3 and try again."
#     exit 1
# fi

# # Check if pip is available
# if ! command -v pip3 &> /dev/null; then
#     echo "Error: pip3 is required but not installed."
#     echo "Please install pip3 and try again."
#     exit 1
# fi

# echo "Installing required Python packages..."
# pip3 install -r track_annotation_requirements.txt

# if [ $? -eq 0 ]; then
#     echo "✓ Dependencies installed successfully!"
# else
#     echo "✗ Error installing dependencies. Please check the error messages above."
#     exit 1
# fi

# echo ""
# echo "Setup complete!"
# echo ""
# echo "To run the Track Annotation App:"
# echo "  python3 track_annotation_app.py [data_directory]"
# echo ""
# echo "Example:"
# echo "  python3 track_annotation_app.py /Users/vdausmann/swimming_in_salt/detection_results/20250312_21.2_01"
# echo ""
# echo "The app will start on http://localhost:8050"
# echo ""
# echo "Controls:"
# echo "  - Click on track markers to select tracks"
# echo "  - Use arrow keys or navigation buttons to move between frames"
# echo "  - Use 'Split Track' to split a track at the current frame"
# echo "  - Use 'Delete Track' to remove the selected track"
# echo "  - Use 'Undo Last Edit' to revert the most recent change"
# echo "  - Use 'Export Tracks' to save your edits"
