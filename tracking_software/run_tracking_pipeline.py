import os
import sys
import pandas as pd
import glob
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add current directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import your modules
from tracking_matching_objs import track_objects, save_tracks, save_tracking_report, visualize_tracks, visualize_tracks_with_colors
from stereo_rectification import StereoRectifier

base_dir = '../swimming_in_salt_data/images/'
sample_name = '20250328_24.4_14_01'  # Example sample name

# Configuration
PIPELINE_CONFIG = {
    # Sample configuration
    'sample_name': sample_name,  # This will create subdirectories in detection and tracking results
    
    # Detection configuration (controls the detection script)
    'detection_config': {
        # Input directories (should match the sample name)
        'left_images': os.path.join(base_dir, sample_name, "right/"),
        'right_images': os.path.join(base_dir, sample_name, "left/"),

        # Calibration directory (where ROI info is stored)
        'calibration_dir': 'calibration_results',

        # Image parameters
        'image_pattern': '*.jpg',
        
        # Detection parameters
        'threshold': 10,
        'min_area': 5,
        'background_frames': 100,  # Number of frames to use for background calculation
        
        # Camera names (should match calibration)
        'left_camera_name': 'lower',
        'right_camera_name': 'upper',
        
        # Processing options
        'save_processed_images': False,
        'save_detection_images': True,
        'create_visualizations': True,
        'verbose': True,
    },
    
    # Tracking parameters
    'tracking_params': {
        'max_distance': 20,
        'max_frames_gap': 5,
        'min_area': 4,
    },
    
    # Processing options
    'run_detection': True,  # Set to True to run detection
    'run_tracking': True,
    'create_visualizations': True,  # Enable tracking visualizations
    
    # Visualization parameters
    'visualization_params': {
        'target_tracks': None,  # None = visualize all tracks
        'max_tracks_to_visualize': None,  # None = no limit, visualize all tracks
        'track_fade_frames': 0,  # Number of frames to keep inactive tracks visible (0 = no fade, keep forever)
        'show_trajectory_length': None,  # Maximum number of trajectory points to show (None = show all)
        'show_legend': False,  # Remove frame number, active tracks, total tracks legend
        'show_epipolar_lines': True,  # Add optional epipolar lines (requires stereo calibration)
        # 'epipolar_line_spacing': 50,  # Pixel spacing between epipolar lines
        # 'epipolar_line_color': (128, 128, 128),  # Gray color for epipolar lines
        # 'epipolar_line_thickness': 1,  # Line thickness
        'show_area': False,  # Show object area next to each detection

    },
    
    # Rectification options
    'use_stereo_rectification': True,  # Enable stereo rectification
    'calibration_dir': './calibration_results',  # Path to calibration data
}

# Dynamically set directories based on sample name
PIPELINE_CONFIG['detection_output'] = f'../swimming_in_salt_data/results/detection_results/{PIPELINE_CONFIG["sample_name"]}'
PIPELINE_CONFIG['tracking_output'] = f'../swimming_in_salt_data/results/tracking_results/{PIPELINE_CONFIG["sample_name"]}'

# Add sample name and output directory to detection config
PIPELINE_CONFIG['detection_config']['sample_name'] = PIPELINE_CONFIG['sample_name']
PIPELINE_CONFIG['detection_config']['output_dir_base'] = '../swimming_in_salt_data/results/detection_results'

def configure_detection_module():
    """Configure the detection module with our pipeline settings"""
    try:
        from run_detection import CONFIG as DETECTION_CONFIG
        
        # Update the detection CONFIG with our pipeline settings
        for key, value in PIPELINE_CONFIG['detection_config'].items():
            DETECTION_CONFIG[key] = value
        
        # Set the full output directory path
        DETECTION_CONFIG['output_dir'] = PIPELINE_CONFIG['detection_output']
        
        print(f"✓ Detection module configured:")
        print(f"  Sample name: {DETECTION_CONFIG['sample_name']}")
        print(f"  Output directory: {DETECTION_CONFIG['output_dir']}")
        print(f"  Left images: {DETECTION_CONFIG['left_images']}")
        print(f"  Right images: {DETECTION_CONFIG['right_images']}")
        print(f"  Threshold: {DETECTION_CONFIG['threshold']}")
        print(f"  Min area: {DETECTION_CONFIG['min_area']}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import detection module: {e}")
        return False
    except Exception as e:
        print(f"✗ Failed to configure detection module: {e}")
        return False

def preprocess_detection_csvs(csv_files: list, camera_name: str):
    """Preprocess detection CSVs to match expected column names for tracking WITH timestamp preservation"""
    
    # Create temporary directory for processing
    temp_dir = tempfile.mkdtemp(prefix=f'tracking_temp_{camera_name}_')
    processed_files = []
    
    print(f"Creating temporary processing directory: {temp_dir}")
    
    for csv_file in csv_files:
        # Load CSV with timestamp extraction
        from tracking_matching_objs import load_csv_with_timestamps
        df = load_csv_with_timestamps(csv_file)
        
        # Check if it's empty
        if len(df) == 0:
            continue
        
        # Map column names to what tracking expects
        df_processed = df.copy()
        
        # Rename columns to match tracking expectations
        column_mapping = {
            'centroid_x': 'x',
            'centroid_y': 'y',
            'area': 'area'  # This should already be correct
        }
        
        df_processed = df_processed.rename(columns=column_mapping)
        
        # Ensure we have the required columns
        required_columns = ['x', 'y', 'area', 'frame', 'timestamp']
        missing_columns = [col for col in required_columns if col not in df_processed.columns]
        
        if missing_columns:
            print(f"⚠ Warning: Missing columns in {csv_file}: {missing_columns}")
            continue
        
        # Save to temporary file with all required columns
        filename = os.path.basename(csv_file)
        temp_file = os.path.join(temp_dir, filename)
        df_processed[required_columns].to_csv(temp_file, index=False)
        processed_files.append(temp_file)
    
    print(f"✅ Processed {len(processed_files)} CSV files for {camera_name} with timestamps (temporary)")
    return processed_files, temp_dir

def create_tracking_summary_report(output_dir: str, left_tracks, right_tracks):
    """Create a summary report comparing left and right tracking results"""
    
    summary_file = os.path.join(output_dir, 'tracking_summary.txt')
    
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRACKING PIPELINE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        # Sample information
        f.write("SAMPLE INFORMATION:\n")
        f.write("-"*25 + "\n")
        f.write(f"Sample name:              {PIPELINE_CONFIG['sample_name']}\n")
        f.write(f"Detection input:          {PIPELINE_CONFIG['detection_output']}\n")
        f.write(f"Tracking output:          {PIPELINE_CONFIG['tracking_output']}\n\n")
        
        # Detection configuration used
        f.write("DETECTION CONFIGURATION:\n")
        f.write("-"*30 + "\n")
        f.write(f"Left images:              {PIPELINE_CONFIG['detection_config']['left_images']}\n")
        f.write(f"Right images:             {PIPELINE_CONFIG['detection_config']['right_images']}\n")
        f.write(f"Threshold:                {PIPELINE_CONFIG['detection_config']['threshold']}\n")
        f.write(f"Min area:                 {PIPELINE_CONFIG['detection_config']['min_area']} pixels\n")
        f.write(f"Background frames:        {PIPELINE_CONFIG['detection_config']['background_frames']}\n\n")
        
        # Basic statistics
        f.write("TRACKING STATISTICS:\n")
        f.write("-"*40 + "\n")
        f.write(f"Left camera tracks:       {len(left_tracks) if left_tracks else 0}\n")
        f.write(f"Right camera tracks:      {len(right_tracks) if right_tracks else 0}\n")
        f.write(f"Total tracks:             {(len(left_tracks) if left_tracks else 0) + (len(right_tracks) if right_tracks else 0)}\n\n")
        
        # Track length analysis
        for camera_name, tracks in [("Left", left_tracks), ("Right", right_tracks)]:
            if tracks:
                track_lengths = [len(track.positions) for track in tracks]
                f.write(f"{camera_name.upper()} CAMERA ANALYSIS:\n")
                f.write("-"*30 + "\n")
                f.write(f"Total tracks:        {len(tracks)}\n")
                f.write(f"Average length:      {sum(track_lengths)/len(track_lengths):.1f} frames\n")
                f.write(f"Longest track:       {max(track_lengths)} frames\n")
                f.write(f"Shortest track:      {min(track_lengths)} frames\n")
                
                # Track length distribution
                short_tracks = sum(1 for length in track_lengths if length <= 5)
                medium_tracks = sum(1 for length in track_lengths if 5 < length <= 20)
                long_tracks = sum(1 for length in track_lengths if length > 20)
                
                f.write(f"Short tracks (≤5):   {short_tracks} ({short_tracks/len(tracks)*100:.1f}%)\n")
                f.write(f"Medium tracks (6-20): {medium_tracks} ({medium_tracks/len(tracks)*100:.1f}%)\n")
                f.write(f"Long tracks (>20):   {long_tracks} ({long_tracks/len(tracks)*100:.1f}%)\n\n")
        
        # Parameters used
        f.write("TRACKING PARAMETERS:\n")
        f.write("-"*25 + "\n")
        for param, value in PIPELINE_CONFIG['tracking_params'].items():
            f.write(f"{param}: {value}\n")
        
        f.write(f"stereo_rectification: {PIPELINE_CONFIG.get('use_stereo_rectification', False)}\n")
        
        # Output files information
        f.write("\nOUTPUT FILES:\n")
        f.write("-"*20 + "\n")
        f.write("Original coordinate tracks:\n")
        f.write(f"  - {PIPELINE_CONFIG['detection_config']['left_camera_name']}_tracks.csv\n")
        f.write(f"  - {PIPELINE_CONFIG['detection_config']['right_camera_name']}_tracks.csv\n")
        
        if PIPELINE_CONFIG.get('use_stereo_rectification', False):
            f.write("\nRectified coordinate tracks (for stereo analysis):\n")
            f.write(f"  - {PIPELINE_CONFIG['detection_config']['left_camera_name']}_tracks_rectified.csv\n")
            f.write(f"  - {PIPELINE_CONFIG['detection_config']['right_camera_name']}_tracks_rectified.csv\n")
        
        f.write("\nVisualization folders:\n")
        f.write("  - visualizations/ (original coordinates)\n")
        if PIPELINE_CONFIG.get('use_stereo_rectification', False):
            f.write("  - visualizations/rectified/ (rectified coordinates)\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"✓ Tracking summary saved to: {summary_file}")

def save_rectified_tracks(tracks, rectifier, camera_name, output_dir):
    """Save tracks with rectified coordinates to CSV"""
    
    # Determine camera side
    camera_side = 'left' if camera_name == PIPELINE_CONFIG['detection_config']['left_camera_name'] else 'right'
    
    print(f"Saving rectified tracks for {camera_name} ({camera_side})...")
    
    # Rectify track positions
    rectified_tracks = rectifier.rectify_track_positions(tracks, camera_side)
    
    # Save rectified tracks to CSV
    rectified_tracks_file = os.path.join(output_dir, f"{camera_name}_tracks_rectified.csv")
    save_tracks(rectified_tracks, rectified_tracks_file)
    
    # Save detailed rectified report
    rectified_report_file = os.path.join(output_dir, f"{camera_name}_tracking_report_rectified.txt")
    save_tracking_report(
        rectified_tracks, 
        rectified_report_file,
        {**PIPELINE_CONFIG['tracking_params'], 
         'input_dir': PIPELINE_CONFIG['detection_output'],
         'camera': camera_name,
         'coordinate_system': 'rectified',
         'sample_name': PIPELINE_CONFIG['sample_name'],
         'note': 'Tracks computed in original coordinates, then rectified for stereo analysis'},
        []  # No CSV files since these are post-processed tracks
    )
    
    print(f"✓ Rectified tracks saved: {rectified_tracks_file}")
    print(f"✓ Rectified report saved: {rectified_report_file}")
    
    return rectified_tracks

def run_complete_pipeline():
    """Run the complete detection and tracking pipeline"""
    
    print("COMPLETE OBJECT DETECTION AND TRACKING PIPELINE")
    print("="*80)
    print(f"📂 Sample: {PIPELINE_CONFIG['sample_name']}")
    print(f"📁 Detection output: {PIPELINE_CONFIG['detection_output']}")
    print(f"📁 Tracking output: {PIPELINE_CONFIG['tracking_output']}")
    print("="*80)
    
    # Step 1: Object Detection (if needed)
    if PIPELINE_CONFIG['run_detection']:
        print("\nSTEP 1: OBJECT DETECTION")
        print("-"*40)
        
        # Configure detection module with our settings
        if not configure_detection_module():
            print("✗ Failed to configure detection module")
            return False
        
        try:
            from run_detection import process_detection_pipeline
            success = process_detection_pipeline()
            if success:
                print("✓ Detection completed successfully")
            else:
                print("✗ Detection failed")
                return False
        except Exception as e:
            print(f"✗ Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print("\nSTEP 1: SKIPPING DETECTION (already done)")
        print("-"*40)
        print(f"Using existing detection results from: {PIPELINE_CONFIG['detection_output']}")
        
        # Verify detection results exist
        if not os.path.exists(PIPELINE_CONFIG['detection_output']):
            print(f"✗ Detection output directory not found: {PIPELINE_CONFIG['detection_output']}")
            print("   Set 'run_detection': True to run detection first, or check the sample_name")
            return False
    
    # Step 2: Object Tracking
    if PIPELINE_CONFIG['run_tracking']:
        print("\nSTEP 2: OBJECT TRACKING")
        print("-"*40)
        
        # Create tracking output directory (with sample subdirectory)
        os.makedirs(PIPELINE_CONFIG['tracking_output'], exist_ok=True)
        print(f"✓ Created output directory: {PIPELINE_CONFIG['tracking_output']}")
        
        # Store tracks for summary
        all_tracks = {'left': None, 'right': None}
        
        # Track for both cameras
        for camera in ['left', 'right']:
            camera_name = PIPELINE_CONFIG['detection_config']['left_camera_name'] if camera == 'left' else PIPELINE_CONFIG['detection_config']['right_camera_name']
            if camera_name == 'upper':
                canvas_size = (720, 676)  # Fixed size for upper camera
            else:
                canvas_size = (827, 676)
            print(f"\nTracking {camera} camera ({camera_name})...")
            
            # Get detection CSV files
            csv_files = sorted(glob.glob(
                os.path.join(PIPELINE_CONFIG['detection_output'], f"{camera_name}_*_objects.csv")
            ))
            
            if not csv_files:
                print(f"⚠ No detection files found for {camera_name}")
                continue
            
            print(f"Found {len(csv_files)} detection files for {camera_name}")
            
            # Preprocess detection files (temporary)
            processed_csv_files, temp_dir = preprocess_detection_csvs(csv_files, camera_name)
            
            if not processed_csv_files:
                print(f"⚠ No valid processed files for {camera_name}")
                # Clean up empty temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue
            
            # Run tracking
            print(f"Running tracking on {len(processed_csv_files)} files...")
            try:
                tracks = track_objects(
                    processed_csv_files,
                    max_distance=PIPELINE_CONFIG['tracking_params']['max_distance'],
                    max_frames_gap=PIPELINE_CONFIG['tracking_params']['max_frames_gap'],
                    min_area=PIPELINE_CONFIG['tracking_params']['min_area']
                )
                
                # Clean up temporary files immediately after tracking
                print(f"✓ Cleaning up temporary files: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                
                all_tracks[camera] = tracks
                
                # Save results
                tracks_file = os.path.join(PIPELINE_CONFIG['tracking_output'], f"{camera_name}_tracks.csv")
                save_tracks(tracks, tracks_file)
                
                # Save report
                report_file = os.path.join(PIPELINE_CONFIG['tracking_output'], f"{camera_name}_tracking_report.txt")
                save_tracking_report(
                    tracks, 
                    report_file,
                    {**PIPELINE_CONFIG['tracking_params'], 
                     'input_dir': PIPELINE_CONFIG['detection_output'],
                     'camera': camera_name,
                     'sample_name': PIPELINE_CONFIG['sample_name']},
                    csv_files  # Use original csv_files for report, not temp files
                )
                
                print(f"✓ {camera_name}: {len(tracks)} tracks found")
                print(f"✓ Results saved: {tracks_file}")
                
                # Save rectified tracks if rectification is enabled
                if PIPELINE_CONFIG.get('use_stereo_rectification', False):
                    try:
                        # Initialize rectifier (only once)
                        if 'rectifier' not in locals():
                            print("Initializing stereo rectifier...")
                            rectifier = StereoRectifier(PIPELINE_CONFIG['calibration_dir'])
                        
                        # Save rectified tracks
                        rectified_tracks = save_rectified_tracks(
                            tracks, 
                            rectifier, 
                            camera_name, 
                            PIPELINE_CONFIG['tracking_output']
                        )
                        
                    except Exception as e:
                        print(f"✗ Rectified track saving failed for {camera_name}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Step 2.5: Create Visualizations
                if PIPELINE_CONFIG['create_visualizations'] and tracks:
                    print(f"\nCreating visualizations for {camera_name}...")
                    
                    # Use all tracks (no filtering)
                    viz_tracks = tracks
                    print(f"Visualizing all {len(viz_tracks)} tracks")

                    # Define custom colors for each track (RGB tuples)
                    custom_colors = [
                        (255, 0, 0),    # Red for track 0
                        (0, 255, 0),    # Green for track 1
                        (0, 0, 255),    # Blue for track 2
                        (255, 255, 0),  # Yellow for track 3
                        (255, 0, 255),  # Magenta for track 4
                        (0, 255, 255),  # Cyan for track 5
                        # Add more colors as needed...
                    ]
                    
                    # Extend colors if you have more tracks than colors
                    while len(custom_colors) < len(viz_tracks):
                        # Generate random bright colors for additional tracks
                        custom_colors.append(tuple(np.random.randint(50, 255, 3).tolist()))

                    # Create visualization output directory
                    viz_output_dir = os.path.join(PIPELINE_CONFIG['tracking_output'], 'visualizations')
                    os.makedirs(viz_output_dir, exist_ok=True)
                    
                    # Get all frame numbers for visualization
                    all_frame_numbers = set()
                    for track in tracks:
                        all_frame_numbers.update(track.frame_indices)
                    
                    debug_frames_all = sorted(list(all_frame_numbers))
                    print(f"Creating visualizations for {len(debug_frames_all)} frames...")
                    
                    # Create original (distorted) visualizations
                    # try:
                    #     print("Creating original coordinate visualizations...")
                    #     visualize_tracks(
                    #         tracks=viz_tracks,
                    #         image_dir="",  # Not used for black background
                    #         prefix=f"{camera_name}_original",
                    #         output_dir=viz_output_dir,
                    #         debug_frames=debug_frames_all,
                    #         target_tracks=PIPELINE_CONFIG['visualization_params']['target_tracks'],
                    #         track_fade_frames=PIPELINE_CONFIG['visualization_params']['track_fade_frames'],
                    #         show_trajectory_length=PIPELINE_CONFIG['visualization_params']['show_trajectory_length'],
                    #         show_legend=PIPELINE_CONFIG['visualization_params']['show_legend'],
                    #         show_epipolar_lines=False,  # Turn off epipolar lines for original coordinates to avoid F-matrix warnings
                    #         show_area=PIPELINE_CONFIG['visualization_params']['show_area'],
                    #         # fixed_canvas_size=canvas_size,
                    #         # epipolar_params={
                    #         #     'line_spacing': PIPELINE_CONFIG['visualization_params']['epipolar_line_spacing'],
                    #         #     'line_color': PIPELINE_CONFIG['visualization_params']['epipolar_line_color'],
                    #         #     'line_thickness': PIPELINE_CONFIG['visualization_params']['epipolar_line_thickness'],
                    #         # },
                    #         # stereo_rectifier=rectifier if 'rectifier' in locals() else None
                    #     )
                        
                    #     # Check if files were actually created
                    #     viz_files = glob.glob(os.path.join(viz_output_dir, f"{camera_name}_original_*.png"))
                    #     print(f"✓ Original visualizations created for {camera_name} ({len(viz_files)} files)")
                        
                    # except Exception as e:
                    #     print(f"✗ Original visualization failed for {camera_name}: {e}")
                    #     import traceback
                    #     traceback.print_exc()
                    
                    # Create rectified visualizations if enabled
                    if PIPELINE_CONFIG.get('use_stereo_rectification', False):
                        try:
                            # Use the already rectified tracks if available, otherwise rectify now
                            if 'rectified_tracks' in locals():
                                viz_rectified_tracks = rectified_tracks
                            else:
                                # Initialize rectifier if not already done
                                if 'rectifier' not in locals():
                                    print("Initializing stereo rectifier...")
                                    rectifier = StereoRectifier(PIPELINE_CONFIG['calibration_dir'])
                                
                                # Rectify tracks for visualization
                                camera_side = 'left' if camera_name == PIPELINE_CONFIG['detection_config']['left_camera_name'] else 'right'
                                viz_rectified_tracks = rectifier.rectify_track_positions(viz_tracks, camera_side)
                            
                            # Create rectified visualization directory
                            rectified_viz_dir = os.path.join(viz_output_dir, 'rectified')
                            os.makedirs(rectified_viz_dir, exist_ok=True)
                            
                            # Save rectification info (only once)
                            rectification_info_file = os.path.join(rectified_viz_dir, 'stereo_rectification_info.txt')
                            if not os.path.exists(rectification_info_file):
                                rectifier.save_rectification_info(rectified_viz_dir)
                            
                            # Create rectified visualizations
                            print(f"Creating rectified coordinate visualizations for {camera_name}...")
                            visualize_tracks_with_colors(
                                tracks=viz_rectified_tracks,
                                colors=custom_colors,
                                image_dir="",  # Not used for black background
                                prefix=f"{camera_name}_rectified",
                                output_dir=rectified_viz_dir,
                                debug_frames=debug_frames_all,
                                target_tracks=PIPELINE_CONFIG['visualization_params']['target_tracks'],
                                track_fade_frames=PIPELINE_CONFIG['visualization_params']['track_fade_frames'],
                                show_trajectory_length=PIPELINE_CONFIG['visualization_params']['show_trajectory_length'],
                                show_legend=PIPELINE_CONFIG['visualization_params']['show_legend'],
                                show_epipolar_lines=PIPELINE_CONFIG['visualization_params']['show_epipolar_lines'],  # Keep epipolar lines for rectified (horizontal lines work well)
                                show_area=PIPELINE_CONFIG['visualization_params']['show_area'],
                                # epipolar_params={
                                #     'line_spacing': PIPELINE_CONFIG['visualization_params']['epipolar_line_spacing'],
                                #     'line_color': PIPELINE_CONFIG['visualization_params']['epipolar_line_color'],
                                #     'line_thickness': PIPELINE_CONFIG['visualization_params']['epipolar_line_thickness'],
                                # },
                                # stereo_rectifier=rectifier if 'rectifier' in locals() else None
                            )
                            
                            # Check if files were created
                            rectified_viz_files = glob.glob(os.path.join(rectified_viz_dir, f"{camera_name}_rectified_*.png"))
                            print(f"✓ Rectified visualizations created: {len(rectified_viz_files)} files")
                            
                        except Exception as e:
                            print(f"✗ Rectified visualization failed for {camera_name}: {e}")
                            import traceback
                            traceback.print_exc()
                
            except Exception as e:
                print(f"✗ Tracking failed for {camera_name}: {e}")
                import traceback
                traceback.print_exc()
                
                # Clean up temp files even if tracking failed
                print(f"✓ Cleaning up temporary files after error: {temp_dir}")
                shutil.rmtree(temp_dir, ignore_errors=True)
                continue
    
    # Step 3: Create Summary Report
    print(f"\nSTEP 3: CREATING SUMMARY REPORT")
    print("-"*40)
    
    create_tracking_summary_report(
        PIPELINE_CONFIG['tracking_output'], 
        all_tracks['left'], 
        all_tracks['right']
    )
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"✅ Sample: {PIPELINE_CONFIG['sample_name']}")
    print(f"✅ Detection results: {PIPELINE_CONFIG['detection_output']}")
    print(f"✅ Tracking results: {PIPELINE_CONFIG['tracking_output']}")
    
    if PIPELINE_CONFIG['create_visualizations']:
        viz_dir = os.path.join(PIPELINE_CONFIG['tracking_output'], 'visualizations')
        print(f"✅ Visualizations: {viz_dir}")
        
        # List all visualization files
        all_viz_files = glob.glob(os.path.join(viz_dir, "*.png"))
        if PIPELINE_CONFIG.get('use_stereo_rectification', False):
            rectified_viz_files = glob.glob(os.path.join(viz_dir, "rectified", "*.png"))
            all_viz_files.extend(rectified_viz_files)
        
        print(f"   Total visualization files created: {len(all_viz_files)}")
    
    # Print summary of results
    left_count = len(all_tracks['left']) if all_tracks['left'] else 0
    right_count = len(all_tracks['right']) if all_tracks['right'] else 0
    
    print(f"\n📊 FINAL SUMMARY:")
    print(f"   Sample name:         {PIPELINE_CONFIG['sample_name']}")
    print(f"   Left camera tracks:  {left_count}")
    print(f"   Right camera tracks: {right_count}")
    print(f"   Total tracks:        {left_count + right_count}")
    
    return True

import argparse

def get_sample_dirs(base_dir):
    # List all directories in base_dir that contain metadata json (indicating a sample)
    sample_dirs = []
    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)
        if os.path.isdir(full_path):
            metadata_path = os.path.join(full_path, f"{entry}_metadata.json")
            if os.path.exists(metadata_path):
                sample_dirs.append(entry)
    return sample_dirs

def main():
    parser = argparse.ArgumentParser(description="Run detection and tracking pipeline for plankton samples.")
    parser.add_argument('--base-dir', type=str, default='../F4/', help='Base directory containing sample folders')
    parser.add_argument('--sample-name', type=str, help='Name of the sample to process')
    parser.add_argument('--all-samples', action='store_true', help='Process all samples in base_dir')
    args = parser.parse_args()

    base_dir = args.base_dir
    if args.all_samples:
        sample_names = get_sample_dirs(base_dir)
        if not sample_names:
            print(f"No valid samples found in {base_dir}")
            sys.exit(1)
        print(f"Found {len(sample_names)} samples in {base_dir}: {sample_names}")
        for sample_name in sample_names:
            print(f"\n=== Processing sample: {sample_name} ===")
            # Update config for this sample
            PIPELINE_CONFIG['sample_name'] = sample_name
            PIPELINE_CONFIG['detection_config']['left_images'] = os.path.join(base_dir, sample_name, "right/")
            PIPELINE_CONFIG['detection_config']['right_images'] = os.path.join(base_dir, sample_name, "left/")
            PIPELINE_CONFIG['detection_output'] = f'../swimming_in_salt_data/results/detection_results/{sample_name}'
            PIPELINE_CONFIG['tracking_output'] = f'../swimming_in_salt_data/results/tracking_results/{sample_name}'
            PIPELINE_CONFIG['detection_config']['sample_name'] = sample_name
            success = run_complete_pipeline()
            if not success:
                print(f"\n❌ Pipeline failed for sample {sample_name}!")
    else:
        sample_name = args.sample_name if args.sample_name else sample_name
        PIPELINE_CONFIG['sample_name'] = sample_name
        PIPELINE_CONFIG['detection_config']['left_images'] = os.path.join(base_dir, sample_name, "right/")
        PIPELINE_CONFIG['detection_config']['right_images'] = os.path.join(base_dir, sample_name, "left/")
        PIPELINE_CONFIG['detection_output'] = f'../swimming_in_salt_data/results/detection_results/{sample_name}'
        PIPELINE_CONFIG['tracking_output'] = f'../swimming_in_salt_data/results/tracking_results/{sample_name}'
        PIPELINE_CONFIG['detection_config']['sample_name'] = sample_name
        success = run_complete_pipeline()
        if not success:
            print("\n❌ Pipeline failed!")
            sys.exit(1)

if __name__ == "__main__":
    main()