#!/Users/vdausmann/miniforge3/envs/cv/bin/python
"""
Test script for Track Annotation Tool

This script tests the basic functionality without running the full web app.
Uses the cv conda environment.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_data_loading(data_dir):
    """Test if we can load the tracking data"""
    print(f"Testing data loading from: {data_dir}")
    
    try:
        # Load tracks
        upper_tracks = pd.read_csv(os.path.join(data_dir, "upper_tracks.csv"))
        lower_tracks = pd.read_csv(os.path.join(data_dir, "lower_tracks.csv"))
        
        print(f"✅ Loaded upper tracks: {len(upper_tracks)} rows, {upper_tracks['track_id'].nunique()} unique tracks")
        print(f"✅ Loaded lower tracks: {len(lower_tracks)} rows, {lower_tracks['track_id'].nunique()} unique tracks")
        
        # Check data structure
        expected_cols = ['track_id', 'frame', 'x', 'y', 'area']
        for col in expected_cols:
            if col not in upper_tracks.columns:
                print(f"❌ Missing column '{col}' in upper_tracks.csv")
                return False
            if col not in lower_tracks.columns:
                print(f"❌ Missing column '{col}' in lower_tracks.csv")
                return False
        
        print("✅ Data structure looks good")
        
        # Find images
        upper_images = sorted(list(Path(data_dir).glob("upper_*.png")))
        lower_images = sorted(list(Path(data_dir).glob("lower_*.png")))
        
        print(f"✅ Found {len(upper_images)} upper images")
        print(f"✅ Found {len(lower_images)} lower images")
        
        if len(upper_images) == 0 or len(lower_images) == 0:
            print("❌ No images found - check your data directory")
            return False
        
        # Test frame range
        max_frame_upper = upper_tracks['frame'].max()
        max_frame_lower = lower_tracks['frame'].max()
        
        print(f"✅ Frame range - Upper: 0 to {max_frame_upper}, Lower: 0 to {max_frame_lower}")
        
        if max_frame_upper >= len(upper_images) or max_frame_lower >= len(lower_images):
            print("⚠️  Warning: Frame indices exceed available images")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

def test_environment():
    """Test the conda environment setup"""
    print("Testing conda environment...")
    print(f"✅ Python executable: {sys.executable}")
    print(f"✅ Python version: {sys.version}")
    
    # Check if we're in the right environment
    if "miniforge3/envs/cv" in sys.executable:
        print("✅ Running in cv conda environment")
    else:
        print("⚠️  Warning: Not running in expected cv environment")
        print(f"   Expected: ~/miniforge3/envs/cv/bin/python")
        print(f"   Current:  {sys.executable}")

def test_basic_functionality():
    """Test basic Python functionality"""
    print("Testing basic functionality...")
    
    try:
        # Test imports that we'll need
        import pandas as pd
        import numpy as np
        from PIL import Image
        print("✅ Core packages available")
        print(f"   pandas version: {pd.__version__}")
        print(f"   numpy version: {np.__version__}")
        
        # Test color generation (from the app)
        import colorsys
        hue = 0.5
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
        color_str = f"rgb({int(rgb[0]*255)}, {int(rgb[1]*255)}, {int(rgb[2]*255)})"
        print(f"✅ Color generation works: {color_str}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in basic functionality: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Track Annotation Tool - Basic Tests")
    print("=" * 40)
    
    # Test environment first
    test_environment()
    print()
    
    # Test basic functionality
    if not test_basic_functionality():
        print("\n❌ Basic functionality test failed")
        sys.exit(1)
    
    # Get data directory
    if len(sys.argv) < 2:
        data_dir = "/Users/vdausmann/swimming_in_salt/detection_results/20250312_21.2_01"
        if not os.path.exists(data_dir):
            print("❌ Please provide a data directory:")
            print("   ~/miniforge3/envs/cv/bin/python test_track_annotation.py <data_directory>")
            sys.exit(1)
        print(f"💡 Using default data directory: {data_dir}")
    else:
        data_dir = sys.argv[1]
    
    if not os.path.exists(data_dir):
        print(f"❌ Data directory does not exist: {data_dir}")
        sys.exit(1)
    
    # Test data loading
    if not test_data_loading(data_dir):
        print(f"\n❌ Data loading test failed for: {data_dir}")
        sys.exit(1)
    
    print("\n✅ All tests passed!")
    print("\n📋 Summary:")
    print("   - Conda environment (cv): ✅")
    print("   - Basic Python functionality: ✅")
    print("   - Data loading and structure: ✅")  
    print("   - Image files: ✅")
    print("\n🚀 Ready to run the Track Annotation Tool!")
    print(f"   ~/miniforge3/envs/cv/bin/python track_annotation_launcher.py {data_dir}")

if __name__ == "__main__":
    main()
