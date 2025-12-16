#!/usr/bin/env python3
"""
Rosbag Crop Script
Crop first 5 seconds and last 5 seconds from each rosbag
Handles zstd-compressed MCAP files
"""

import os
import sys
import subprocess
import shutil
import yaml
import tempfile
from pathlib import Path

CROP_MARGIN = 5  # seconds to crop from start and end

# Colors for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
NC = '\033[0m'  # No Color


def run_command(cmd, check=True, capture_output=False):
    """Run a shell command"""
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=capture_output, text=True)
    if check and result.returncode != 0:
        if capture_output:
            print(f"  Error: {result.stderr}")
        return None
    return result


def get_bag_info(bag_path):
    """Get rosbag info from metadata.yaml"""
    metadata_path = os.path.join(bag_path, "metadata.yaml")
    if not os.path.exists(metadata_path):
        return None, None, None
    
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    info = metadata.get('rosbag2_bagfile_information', {})
    start_ns = info.get('starting_time', {}).get('nanoseconds_since_epoch', 0)
    duration_ns = info.get('duration', {}).get('nanoseconds', 0)
    
    start_sec = start_ns / 1e9
    duration_sec = duration_ns / 1e9
    end_sec = start_sec + duration_sec
    
    return start_sec, end_sec, duration_sec


def decompress_bag(bag_path):
    """Decompress zstd compressed bag file"""
    metadata_path = os.path.join(bag_path, "metadata.yaml")
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    info = metadata.get('rosbag2_bagfile_information', {})
    compression_format = info.get('compression_format', '')
    
    if compression_format != 'zstd':
        return True  # Not compressed, nothing to do
    
    # Find compressed files
    relative_paths = info.get('relative_file_paths', [])
    for rel_path in relative_paths:
        if rel_path.endswith('.zstd'):
            compressed_file = os.path.join(bag_path, rel_path)
            decompressed_file = compressed_file[:-5]  # Remove .zstd extension
            
            print(f"  Decompressing {rel_path}...")
            result = run_command(['zstd', '-d', '-f', compressed_file], check=False)
            if result is None or result.returncode != 0:
                print(f"{RED}  Failed to decompress {rel_path}{NC}")
                return False
    
    # Update metadata to remove compression info
    info['compression_format'] = ''
    info['compression_mode'] = ''
    info['relative_file_paths'] = [p[:-5] if p.endswith('.zstd') else p for p in relative_paths]
    
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    return True


def recompress_bag(bag_path):
    """Recompress bag file with zstd"""
    metadata_path = os.path.join(bag_path, "metadata.yaml")
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    info = metadata.get('rosbag2_bagfile_information', {})
    relative_paths = info.get('relative_file_paths', [])
    
    new_paths = []
    for rel_path in relative_paths:
        if not rel_path.endswith('.zstd'):
            file_path = os.path.join(bag_path, rel_path)
            if os.path.exists(file_path):
                print(f"  Compressing {rel_path}...")
                result = run_command(['zstd', '-f', '--rm', file_path], check=False)
                if result is None or result.returncode != 0:
                    print(f"{YELLOW}  Warning: Failed to compress {rel_path}{NC}")
                    new_paths.append(rel_path)
                else:
                    new_paths.append(rel_path + '.zstd')
            else:
                new_paths.append(rel_path)
        else:
            new_paths.append(rel_path)
    
    # Update metadata
    info['compression_format'] = 'zstd'
    info['compression_mode'] = 'FILE'
    info['relative_file_paths'] = new_paths
    
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)


def restore_original_bag(bag_path):
    """Restore bag to original compressed state (cleanup uncompressed files)"""
    metadata_path = os.path.join(bag_path, "metadata.yaml")
    with open(metadata_path, 'r') as f:
        metadata = yaml.safe_load(f)
    
    info = metadata.get('rosbag2_bagfile_information', {})
    relative_paths = info.get('relative_file_paths', [])
    
    # Delete uncompressed files and restore metadata
    for rel_path in relative_paths:
        file_path = os.path.join(bag_path, rel_path)
        compressed_path = file_path + '.zstd'
        
        # If both exist, keep compressed and remove uncompressed
        if os.path.exists(file_path) and os.path.exists(compressed_path):
            os.remove(file_path)
    
    # Restore compression info in metadata
    info['compression_format'] = 'zstd'
    info['compression_mode'] = 'FILE'
    info['relative_file_paths'] = [p + '.zstd' if not p.endswith('.zstd') else p for p in relative_paths]
    
    with open(metadata_path, 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False)


def crop_rosbag(bag_path, output_path):
    """Crop a single rosbag"""
    bag_name = os.path.basename(bag_path)
    print(f"{YELLOW}Processing: {bag_name}{NC}")
    
    # Get times
    start_time, end_time, duration = get_bag_info(bag_path)
    
    if start_time is None:
        print(f"{RED}  Failed to get bag times{NC}")
        return False
    
    print(f"  Original: start={start_time:.6f}, end={end_time:.6f}, duration={duration:.3f}s")
    
    # Check duration
    if duration < CROP_MARGIN * 2 + 1:
        print(f"{RED}  Skipping: Duration too short{NC}")
        return False
    
    # Calculate new times
    new_start = start_time + CROP_MARGIN
    new_end = end_time - CROP_MARGIN
    new_duration = duration - CROP_MARGIN * 2
    
    print(f"  Cropped:  start={new_start:.6f}, end={new_end:.6f}, duration={new_duration:.3f}s")
    
    # Remove existing output
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    
    # Decompress if needed
    print(f"  Preparing bag...")
    if not decompress_bag(bag_path):
        restore_original_bag(bag_path)
        return False
    
    # Run slice
    print(f"  Running slice...")
    result = run_command([
        'ros2', 'bag', 'slice',
        bag_path,
        '-o', output_path,
        '-b', str(new_start),
        '-e', str(new_end),
        '-s', 'mcap'
    ], check=False)
    
    # Restore original bag state
    restore_original_bag(bag_path)
    
    if result is None or result.returncode != 0:
        print(f"{RED}  Failed to crop{NC}")
        return False
    
    # Compress output
    print(f"  Compressing output...")
    recompress_bag(output_path)
    
    print(f"{GREEN}  Successfully cropped: {output_path}{NC}")
    return True


def replace_with_cropped(bag_path):
    """Replace original bag with cropped version"""
    bag_name = os.path.basename(bag_path)
    parent_dir = os.path.dirname(bag_path)
    cropped_path = os.path.join(parent_dir, f"{bag_name}_cropped")
    backup_path = os.path.join(parent_dir, f"{bag_name}_original")
    
    if os.path.exists(cropped_path):
        print(f"  Replacing: {bag_name}")
        shutil.move(bag_path, backup_path)
        shutil.move(cropped_path, bag_path)
        print(f"{GREEN}  Done{NC}")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, "dataset")
    
    print("======================================")
    print("  Rosbag Crop Script")
    print(f"  Cropping {CROP_MARGIN}s from start and end")
    print("======================================")
    
    # Handle arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--replace':
            print("Replacing original bags with cropped versions...")
            for split in ['train', 'val']:
                split_dir = os.path.join(dataset_dir, split)
                if not os.path.exists(split_dir):
                    continue
                for bag_name in sorted(os.listdir(split_dir)):
                    if bag_name.endswith('_cropped') or bag_name.endswith('_original'):
                        continue
                    bag_path = os.path.join(split_dir, bag_name)
                    if os.path.isdir(bag_path):
                        replace_with_cropped(bag_path)
            print("Done!")
            return
        
        elif sys.argv[1] == '--cleanup':
            print("Removing backup (original) files...")
            for split in ['train', 'val']:
                split_dir = os.path.join(dataset_dir, split)
                if not os.path.exists(split_dir):
                    continue
                for bag_name in os.listdir(split_dir):
                    if bag_name.endswith('_original'):
                        bag_path = os.path.join(split_dir, bag_name)
                        print(f"  Removing: {bag_name}")
                        shutil.rmtree(bag_path)
            print("Done!")
            return
    
    # Process bags
    for split in ['train', 'val']:
        print()
        print("======================================")
        print(f"  Processing {split.capitalize()} Dataset")
        print("======================================")
        
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"  {split_dir} not found, skipping")
            continue
        
        for bag_name in sorted(os.listdir(split_dir)):
            if bag_name.endswith('_cropped') or bag_name.endswith('_original'):
                continue
            
            bag_path = os.path.join(split_dir, bag_name)
            if not os.path.isdir(bag_path):
                continue
            
            output_path = os.path.join(split_dir, f"{bag_name}_cropped")
            crop_rosbag(bag_path, output_path)
            print()
    
    print()
    print("======================================")
    print("  Summary")
    print("======================================")
    print("Cropped bags are saved with '_cropped' suffix.")
    print()
    print("To replace originals with cropped versions, run:")
    print("  python3 crop_rosbags.py --replace")
    print()
    print("To delete backup (original) files after replacement, run:")
    print("  python3 crop_rosbags.py --cleanup")


if __name__ == '__main__':
    main()
