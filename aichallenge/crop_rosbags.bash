#!/bin/bash

# ==============================================
# Rosbag Crop Script
# Crop first 5 seconds and last 5 seconds from each rosbag
# Using ros2bag_extensions (slice command)
# ==============================================

set -e

SCRIPT_DIR=$(cd $(dirname $0) && pwd)
DATASET_DIR="${SCRIPT_DIR}/dataset"
EXTENSION_WS="/tmp/extension_ws"
CROP_MARGIN=5  # seconds to crop from start and end

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================"
echo "  Rosbag Crop Script"
echo "  Cropping ${CROP_MARGIN}s from start and end"
echo "======================================"

# Source ROS2
source /opt/ros/humble/setup.bash
source /autoware/install/setup.bash 2>/dev/null || true

# Function to install ros2bag_extensions if not already installed
install_ros2bag_extensions() {
    echo -e "${YELLOW}Setting up ros2bag_extensions...${NC}"
    
    mkdir -p ${EXTENSION_WS}/src
    cd ${EXTENSION_WS}/src
    
    if [ ! -d "ros2bag_extensions" ]; then
        git clone --depth 1 https://github.com/tier4/ros2bag_extensions.git
    fi
    
    cd ${EXTENSION_WS}
    rosdep update || true
    rosdep install --from-paths src --ignore-src --rosdistro=${ROS_DISTRO} -y || true
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
    
    echo -e "${GREEN}ros2bag_extensions setup complete${NC}"
}

# Check and install ros2bag_extensions
if [ ! -f "${EXTENSION_WS}/install/setup.bash" ]; then
    install_ros2bag_extensions
fi

# Source extension workspace and fix PYTHONPATH for symlink install
export PYTHONPATH=${EXTENSION_WS}/src/ros2bag_extensions/ros2bag_extensions:$PYTHONPATH
source ${EXTENSION_WS}/install/setup.bash

# Verify slice command is available
if ! ros2 bag slice --help &>/dev/null; then
    echo -e "${RED}Error: ros2 bag slice command not available${NC}"
    exit 1
fi
echo -e "${GREEN}ros2 bag slice command available${NC}"

# Python helper function for float arithmetic
calc() {
    python3 -c "print($1)"
}

# Function to get rosbag start and end time
get_bag_times() {
    local bag_path=$1
    local info_output=$(ros2 bag info "$bag_path" 2>/dev/null)
    
    # Extract start and end timestamps (in parentheses)
    local start_time=$(echo "$info_output" | grep "Start:" | grep -oP '\(\K[0-9.]+')
    local end_time=$(echo "$info_output" | grep "End:" | grep -oP '\(\K[0-9.]+')
    local duration=$(echo "$info_output" | grep "Duration:" | grep -oP '[0-9.]+(?=s)')
    
    echo "$start_time $end_time $duration"
}

# Function to crop a single rosbag
crop_rosbag() {
    local bag_path=$1
    local bag_name=$(basename "$bag_path")
    local parent_dir=$(dirname "$bag_path")
    local output_path="${parent_dir}/${bag_name}_cropped"
    
    echo -e "${YELLOW}Processing: ${bag_name}${NC}"
    
    # Get times
    read start_time end_time duration <<< $(get_bag_times "$bag_path")
    
    if [ -z "$start_time" ] || [ -z "$end_time" ]; then
        echo -e "${RED}  Failed to get bag times for ${bag_name}${NC}"
        return 1
    fi
    
    echo "  Original: start=${start_time}, end=${end_time}, duration=${duration}s"
    
    # Check if duration is long enough to crop (using Python for float comparison)
    min_duration=$(calc "$CROP_MARGIN * 2 + 1")
    can_crop=$(python3 -c "print(1 if $duration >= $min_duration else 0)")
    if [ "$can_crop" -eq 0 ]; then
        echo -e "${RED}  Skipping: Duration (${duration}s) too short to crop ${CROP_MARGIN}s from each end${NC}"
        return 1
    fi
    
    # Calculate new start and end times (using Python for float arithmetic)
    new_start=$(calc "$start_time + $CROP_MARGIN")
    new_end=$(calc "$end_time - $CROP_MARGIN")
    new_duration=$(calc "$duration - $CROP_MARGIN * 2")
    
    echo "  Cropped:  start=${new_start}, end=${new_end}, duration=${new_duration}s"
    
    # Remove existing output if exists
    if [ -d "$output_path" ]; then
        echo "  Removing existing cropped bag..."
        rm -rf "$output_path"
    fi
    
    # Run slice command (use -b for beginning time, -e for end time, -s mcap for MCAP format)
    echo "  Running slice..."
    ros2 bag slice "$bag_path" -o "$output_path" -b "$new_start" -e "$new_end" -s mcap
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  Successfully cropped: ${output_path}${NC}"
        return 0
    else
        echo -e "${RED}  Failed to crop ${bag_name}${NC}"
        return 1
    fi
}

# Function to replace original with cropped version
replace_with_cropped() {
    local bag_path=$1
    local bag_name=$(basename "$bag_path")
    local parent_dir=$(dirname "$bag_path")
    local cropped_path="${parent_dir}/${bag_name}_cropped"
    local backup_path="${parent_dir}/${bag_name}_original"
    
    if [ -d "$cropped_path" ]; then
        echo "  Replacing: ${bag_name}"
        echo "  Backing up original to ${bag_name}_original..."
        mv "$bag_path" "$backup_path"
        echo "  Renaming cropped to ${bag_name}..."
        mv "$cropped_path" "$bag_path"
        echo -e "${GREEN}  Replaced original with cropped version${NC}"
    fi
}

# Main execution
main() {
    echo ""
    echo "======================================"
    echo "  Processing Train Dataset"
    echo "======================================"
    
    # Process train rosbags
    for bag_dir in ${DATASET_DIR}/train/rosbag2_*; do
        if [ -d "$bag_dir" ] && [[ ! "$bag_dir" =~ _cropped$ ]] && [[ ! "$bag_dir" =~ _original$ ]]; then
            crop_rosbag "$bag_dir" || true
            echo ""
        fi
    done
    
    echo ""
    echo "======================================"
    echo "  Processing Validation Dataset"
    echo "======================================"
    
    # Process val rosbags
    for bag_dir in ${DATASET_DIR}/val/rosbag2_*; do
        if [ -d "$bag_dir" ] && [[ ! "$bag_dir" =~ _cropped$ ]] && [[ ! "$bag_dir" =~ _original$ ]]; then
            crop_rosbag "$bag_dir" || true
            echo ""
        fi
    done
    
    echo ""
    echo "======================================"
    echo "  Summary"
    echo "======================================"
    echo "Cropped bags are saved with '_cropped' suffix."
    echo ""
    echo "To replace originals with cropped versions, run:"
    echo "  ./crop_rosbags.bash --replace"
    echo ""
    echo "To delete backup (original) files after replacement, run:"
    echo "  ./crop_rosbags.bash --cleanup"
}

# Handle command line arguments
case "${1:-}" in
    --replace)
        echo "Replacing original bags with cropped versions..."
        for bag_dir in ${DATASET_DIR}/train/rosbag2_*; do
            if [ -d "$bag_dir" ] && [[ ! "$bag_dir" =~ _cropped$ ]] && [[ ! "$bag_dir" =~ _original$ ]]; then
                replace_with_cropped "$bag_dir"
            fi
        done
        for bag_dir in ${DATASET_DIR}/val/rosbag2_*; do
            if [ -d "$bag_dir" ] && [[ ! "$bag_dir" =~ _cropped$ ]] && [[ ! "$bag_dir" =~ _original$ ]]; then
                replace_with_cropped "$bag_dir"
            fi
        done
        echo "Done!"
        ;;
    --cleanup)
        echo "Removing backup (original) files..."
        rm -rf ${DATASET_DIR}/train/*_original
        rm -rf ${DATASET_DIR}/val/*_original
        echo "Done!"
        ;;
    *)
        main
        ;;
esac
