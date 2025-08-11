#!/Users/vdausmann/miniforge3/envs/cv/bin/python
"""
Track Annotation Tool Launcher

This script handles package installation and launches the Dash application
for track annotation and validation. Uses the cv conda environment.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_and_install_packages():
    """Check for required packages and install if missing"""
    print("🔍 Checking required packages...")
    
    required_packages = [
        'dash',
        'plotly', 
        'pandas',
        'numpy',
        'pillow'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            # Use the conda environment's pip
            pip_path = Path(sys.executable).parent / "pip"
            subprocess.check_call([str(pip_path), "install"] + missing_packages)
            print("✅ Package installation completed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install packages: {e}")
            print("💡 Try manually installing with:")
            print(f"   conda activate cv")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    
    return True

def validate_data_directory(data_dir):
    """Validate that the data directory contains required files"""
    if not os.path.exists(data_dir):
        print(f"❌ Data directory does not exist: {data_dir}")
        return False
    
    required_files = ["upper_tracks.csv", "lower_tracks.csv"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files in {data_dir}:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Make sure you've run your tracking script first to generate the CSV files")
        return False
    
    # Check for images
    upper_images = list(Path(data_dir).glob("upper_*.png"))
    lower_images = list(Path(data_dir).glob("lower_*.png"))
    
    if len(upper_images) == 0 or len(lower_images) == 0:
        print(f"❌ No image files found in {data_dir}")
        print("   Expected: upper_*.png and lower_*.png files")
        return False
    
    print(f"✅ Data directory validated: {len(upper_images)} upper images, {len(lower_images)} lower images")
    return True

def launch_app(data_dir):
    """Launch the track annotation application"""
    print("🚀 Launching Track Annotation Tool...")
    
    try:
        # Import and run the app
        from track_annotation_app import create_app
        
        app = create_app(data_dir)
        print("\n🌐 Starting web server...")
        print("   URL: http://localhost:8050")
        print("   Press Ctrl+C to stop")
        
        app.run(debug=True, host='localhost', port=8050)
        
    except ImportError as e:
        print(f"❌ Failed to import app: {e}")
        print("💡 Make sure track_annotation_app.py is in the same directory")
        return False
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        return False

def main():
    """Main launcher function"""
    print("🎯 Track Annotation Tool Launcher")
    print("=" * 40)
    print(f"Using Python: {sys.executable}")
    
    # Check if we're in the right environment
    if "miniforge3/envs/cv" not in sys.executable:
        print("⚠️  Warning: Not running in expected cv environment")
        print("💡 Consider running with:")
        print("   ~/miniforge3/envs/cv/bin/python track_annotation_launcher.py <data_dir>")
        print()
    
    # Get data directory
    if len(sys.argv) < 2:
        print("❌ Please provide a data directory:")
        print("   ~/miniforge3/envs/cv/bin/python track_annotation_launcher.py <data_directory>")
        print("\nExample:")
        print("   ~/miniforge3/envs/cv/bin/python track_annotation_launcher.py /Users/vdausmann/swimming_in_salt/detection_results/20250312_21.2_01")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    
    # Validate data directory
    if not validate_data_directory(data_dir):
        sys.exit(1)
    
    # Check and install packages
    if not check_and_install_packages():
        sys.exit(1)
    
    # Launch the application
    launch_app(data_dir)

if __name__ == "__main__":
    main()
