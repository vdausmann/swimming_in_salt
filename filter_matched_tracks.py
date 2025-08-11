#!/usr/bin/env python3
"""
Filter Stereo-Matched Tracks Script

This script reads the stereo_track_matches.csv file and creates filtered versions 
of the track CSV files that exclude tracks that have been stereo-matched.

Usage:
    python filter_matched_tracks.py <data_directory>
    
The script will create:
    - upper_tracks_unmatched.csv
    - lower_tracks_unmatched.csv
    - track_filtering_report.txt
"""

import os
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime
import argparse

def filter_matched_tracks(data_dir: str, create_backup: bool = True, verbose: bool = True):
    """
    Filter out stereo-matched tracks from the main CSV files.
    
    Args:
        data_dir: Path to the data directory containing CSV files
        create_backup: Whether to create backup copies of original files
        verbose: Whether to print detailed information
    
    Returns:
        dict: Statistics about the filtering operation
    """
    
    data_path = Path(data_dir)
    
    # Check if directory exists
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Define file paths
    stereo_matches_file = data_path / "stereo_track_matches.csv"
    upper_tracks_file = data_path / "upper_tracks.csv"
    lower_tracks_file = data_path / "lower_tracks.csv"
    
    # Check if required files exist
    missing_files = []
    for file_path, name in [(stereo_matches_file, "stereo_track_matches.csv"),
                           (upper_tracks_file, "upper_tracks.csv"),
                           (lower_tracks_file, "lower_tracks.csv")]:
        if not file_path.exists():
            missing_files.append(name)
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
    
    # Load data
    if verbose:
        print("📁 Loading data files...")
    
    stereo_matches = pd.read_csv(stereo_matches_file)
    upper_tracks = pd.read_csv(upper_tracks_file)
    lower_tracks = pd.read_csv(lower_tracks_file)
    
    if verbose:
        print(f"   Loaded {len(stereo_matches)} stereo matches")
        print(f"   Loaded {len(upper_tracks)} upper track points")
        print(f"   Loaded {len(lower_tracks)} lower track points")
    
    # Get matched track IDs
    matched_upper_tracks = set(stereo_matches['upper_track_id'].unique())
    matched_lower_tracks = set(stereo_matches['lower_track_id'].unique())
    
    if verbose:
        print(f"\n🔗 Found matched tracks:")
        print(f"   {len(matched_upper_tracks)} unique upper tracks matched")
        print(f"   {len(matched_lower_tracks)} unique lower tracks matched")
        
        # Show the matches
        for _, match in stereo_matches.iterrows():
            print(f"   Match {match['match_id']}: Upper {match['upper_track_id']} ↔ Lower {match['lower_track_id']}")
    
    # Create backups if requested
    if create_backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = data_path / "filter_backups"
        backup_dir.mkdir(exist_ok=True)
        
        backup_files = [
            (upper_tracks_file, backup_dir / f"upper_tracks_backup_{timestamp}.csv"),
            (lower_tracks_file, backup_dir / f"lower_tracks_backup_{timestamp}.csv"),
            (stereo_matches_file, backup_dir / f"stereo_track_matches_backup_{timestamp}.csv")
        ]
        
        if verbose:
            print(f"\n💾 Creating backups in {backup_dir}...")
        
        for src, dst in backup_files:
            src_df = pd.read_csv(src)
            src_df.to_csv(dst, index=False)
            if verbose:
                print(f"   {src.name} → {dst.name}")
    
    # Filter tracks
    if verbose:
        print(f"\n🔍 Filtering tracks...")
    
    # Filter upper tracks (remove matched ones)
    upper_tracks_filtered = upper_tracks[~upper_tracks['track_id'].isin(matched_upper_tracks)]
    upper_removed = len(upper_tracks) - len(upper_tracks_filtered)
    
    # Filter lower tracks (remove matched ones)
    lower_tracks_filtered = lower_tracks[~lower_tracks['track_id'].isin(matched_lower_tracks)]
    lower_removed = len(lower_tracks) - len(lower_tracks_filtered)
    
    # Get statistics
    stats = {
        'original_upper_tracks': len(upper_tracks['track_id'].unique()),
        'original_lower_tracks': len(lower_tracks['track_id'].unique()),
        'original_upper_points': len(upper_tracks),
        'original_lower_points': len(lower_tracks),
        'matched_upper_tracks': len(matched_upper_tracks),
        'matched_lower_tracks': len(matched_lower_tracks),
        'filtered_upper_tracks': len(upper_tracks_filtered['track_id'].unique()),
        'filtered_lower_tracks': len(lower_tracks_filtered['track_id'].unique()),
        'filtered_upper_points': len(upper_tracks_filtered),
        'filtered_lower_points': len(lower_tracks_filtered),
        'removed_upper_points': upper_removed,
        'removed_lower_points': lower_removed,
        'stereo_matches': len(stereo_matches)
    }
    
    if verbose:
        print(f"   Upper tracks: {stats['original_upper_tracks']} → {stats['filtered_upper_tracks']} (removed {stats['matched_upper_tracks']})")
        print(f"   Lower tracks: {stats['original_lower_tracks']} → {stats['filtered_lower_tracks']} (removed {stats['matched_lower_tracks']})")
        print(f"   Upper points: {stats['original_upper_points']} → {stats['filtered_upper_points']} (removed {stats['removed_upper_points']})")
        print(f"   Lower points: {stats['original_lower_points']} → {stats['filtered_lower_points']} (removed {stats['removed_lower_points']})")
    
    # Save filtered files
    output_files = [
        (upper_tracks_filtered, data_path / "upper_tracks_unmatched.csv"),
        (lower_tracks_filtered, data_path / "lower_tracks_unmatched.csv")
    ]
    
    if verbose:
        print(f"\n💾 Saving filtered files...")
    
    for df, file_path in output_files:
        df.to_csv(file_path, index=False)
        if verbose:
            print(f"   {file_path.name} ({len(df)} points)")
    
    # Create detailed report
    report_file = data_path / "track_filtering_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("Track Filtering Report\n")
        f.write("=" * 50 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data directory: {data_dir}\n\n")
        
        f.write("Original Data:\n")
        f.write(f"  Upper tracks: {stats['original_upper_tracks']} tracks, {stats['original_upper_points']} points\n")
        f.write(f"  Lower tracks: {stats['original_lower_tracks']} tracks, {stats['original_lower_points']} points\n\n")
        
        f.write("Stereo Matches:\n")
        f.write(f"  Total matches: {stats['stereo_matches']}\n")
        f.write(f"  Matched upper tracks: {stats['matched_upper_tracks']}\n")
        f.write(f"  Matched lower tracks: {stats['matched_lower_tracks']}\n\n")
        
        f.write("Match Details:\n")
        for _, match in stereo_matches.iterrows():
            upper_points = len(upper_tracks[upper_tracks['track_id'] == match['upper_track_id']])
            lower_points = len(lower_tracks[lower_tracks['track_id'] == match['lower_track_id']])
            f.write(f"  Match {match['match_id']}: Upper {match['upper_track_id']} ({upper_points} pts) ↔ Lower {match['lower_track_id']} ({lower_points} pts)\n")
        
        f.write(f"\nFiltered Data:\n")
        f.write(f"  Upper tracks: {stats['filtered_upper_tracks']} tracks, {stats['filtered_upper_points']} points (removed {stats['removed_upper_points']})\n")
        f.write(f"  Lower tracks: {stats['filtered_lower_tracks']} tracks, {stats['filtered_lower_points']} points (removed {stats['removed_lower_points']})\n\n")
        
        f.write("Output Files:\n")
        f.write(f"  upper_tracks_unmatched.csv\n")
        f.write(f"  lower_tracks_unmatched.csv\n")
        f.write(f"  track_filtering_report.txt\n")
        
        if create_backup:
            f.write(f"\nBackups created in: filter_backups/\n")
    
    if verbose:
        print(f"   {report_file.name}")
        print(f"\n✅ Filtering complete! Check {report_file.name} for detailed information.")
    
    return stats

def get_available_data_directories(base_dir: str):
    """Get available data directories (same logic as in your main app)"""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []
    
    data_dirs = []
    for subdir in base_path.iterdir():
        if subdir.is_dir():
            upper_tracks = subdir / "upper_tracks.csv"
            lower_tracks = subdir / "lower_tracks.csv"
            if upper_tracks.exists() and lower_tracks.exists():
                data_dirs.append(str(subdir))
    
    return data_dirs

def main():
    parser = argparse.ArgumentParser(
        description="Filter stereo-matched tracks from track CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python filter_matched_tracks.py /path/to/data/directory
  python filter_matched_tracks.py /path/to/data/directory --no-backup
  python filter_matched_tracks.py --list-dirs
  python filter_matched_tracks.py --interactive
        """
    )
    
    parser.add_argument(
        "data_dir", 
        nargs="?",
        help="Path to the data directory containing track CSV files"
    )
    
    parser.add_argument(
        "--no-backup", 
        action="store_true",
        help="Don't create backup files before filtering"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true", 
        help="Reduce output verbosity"
    )
    
    parser.add_argument(
        "--list-dirs", "-l",
        action="store_true",
        help="List available data directories and exit"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode: choose from available directories"
    )
    
    parser.add_argument(
        "--base-dir",
        default="/Users/vdausmann/swimming_in_salt/detection_results",
        help="Base directory to search for data directories (default: your current base dir)"
    )
    
    args = parser.parse_args()
    
    try:
        # List directories mode
        if args.list_dirs:
            print("Available data directories:")
            dirs = get_available_data_directories(args.base_dir)
            if not dirs:
                print("  No data directories found.")
                return 1
            
            for i, dir_path in enumerate(dirs, 1):
                dir_name = Path(dir_path).name
                print(f"  {i}. {dir_name} ({dir_path})")
            return 0
        
        # Interactive mode
        if args.interactive:
            dirs = get_available_data_directories(args.base_dir)
            if not dirs:
                print("❌ No data directories found.")
                return 1
            
            print("Available data directories:")
            for i, dir_path in enumerate(dirs, 1):
                dir_name = Path(dir_path).name
                # Check if stereo matches exist
                stereo_file = Path(dir_path) / "stereo_track_matches.csv"
                if stereo_file.exists():
                    matches_df = pd.read_csv(stereo_file)
                    match_count = len(matches_df)
                    print(f"  {i}. {dir_name} ({match_count} stereo matches)")
                else:
                    print(f"  {i}. {dir_name} (no stereo matches)")
            
            while True:
                try:
                    choice = input(f"\nSelect directory (1-{len(dirs)}) or 'q' to quit: ").strip()
                    if choice.lower() == 'q':
                        return 0
                    
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(dirs):
                        args.data_dir = dirs[choice_idx]
                        break
                    else:
                        print(f"Please enter a number between 1 and {len(dirs)}")
                except ValueError:
                    print("Please enter a valid number or 'q'")
        
        # Check if data_dir is provided
        if not args.data_dir:
            print("❌ Error: Data directory is required.")
            print("Use --help for usage information.")
            return 1
        
        # Check if stereo matches exist
        stereo_file = Path(args.data_dir) / "stereo_track_matches.csv"
        if not stereo_file.exists():
            print(f"❌ No stereo matches found in {args.data_dir}")
            print("   Make sure you have created stereo matches using the annotation tool first.")
            return 1
        
        # Check number of matches
        stereo_matches = pd.read_csv(stereo_file)
        if len(stereo_matches) == 0:
            print(f"❌ No stereo matches found in {stereo_file}")
            print("   The file exists but contains no matches.")
            return 1
        
        print(f"🎯 Processing data directory: {Path(args.data_dir).name}")
        print(f"   Found {len(stereo_matches)} stereo matches to filter out")
        
        # Confirm operation
        if not args.quiet:
            response = input("\nProceed with filtering? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                print("Operation cancelled.")
                return 0
        
        # Perform filtering
        stats = filter_matched_tracks(
            args.data_dir, 
            create_backup=not args.no_backup,
            verbose=not args.quiet
        )
        
        if not args.quiet:
            print(f"\n📊 Summary:")
            print(f"   Removed {stats['matched_upper_tracks']} upper tracks ({stats['removed_upper_points']} points)")
            print(f"   Removed {stats['matched_lower_tracks']} lower tracks ({stats['removed_lower_points']} points)")
            print(f"   Remaining: {stats['filtered_upper_tracks']} upper, {stats['filtered_lower_tracks']} lower tracks")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())