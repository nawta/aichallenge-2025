#!/usr/bin/env python3
import time
import numpy as np
import threading
from collections import deque
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from autoware_auto_control_msgs.msg import AckermannControlCommand

from tiny_lidar_net_controller_core import TinyLidarNetCore
from map_loader import load_lane_boundaries, LaneBoundaries
from bev_generator import BEVGenerator, quaternion_to_yaw

# Temporal model architectures that require sequence processing
TEMPORAL_ARCHITECTURES = ['stacked', 'bilstm', 'tcn']

# BEV-enabled model architectures
BEV_ARCHITECTURES = ['local_bev', 'global_bev', 'dual_bev']

# Map-enabled model architectures (static map image)
MAP_ARCHITECTURES = ['map_image']


class TinyLidarNetNode(Node):
    """ROS 2 Node for TinyLidarNet autonomous driving control.

    This node subscribes to LaserScan messages, processes them using the
    TinyLidarNetCore logic, and publishes AckermannControlCommand messages.
    
    Supports multiple architectures:
    - large, small, deep: Single-frame models
    - fusion: Single-frame with odometry
    - stacked, bilstm, tcn: Temporal models with sequence processing
    - local_bev, global_bev, dual_bev: BEV-enabled models with map data
    """

    def __init__(self):
        super().__init__('tiny_lidar_net_node')

        # --- Parameter Declaration ---
        self.declare_parameter('log_interval_sec', 5.0)
        self.declare_parameter('model.input_dim', 1080)
        self.declare_parameter('model.output_dim', 2)
        self.declare_parameter('model.state_dim', 13)
        self.declare_parameter('model.seq_len', 10)
        self.declare_parameter('model.hidden_size', 128)
        self.declare_parameter('model.architecture', 'large')
        self.declare_parameter('model.ckpt_path', '')
        self.declare_parameter('max_range', 30.0)
        self.declare_parameter('acceleration', 0.1)
        self.declare_parameter('control_mode', 'ai')
        self.declare_parameter('debug', False)
        
        # BEV parameters
        self.declare_parameter('bev.map_path', '')
        self.declare_parameter('bev.local_size', 64)
        self.declare_parameter('bev.local_resolution', 1.0)
        self.declare_parameter('bev.local_channels', 2)
        self.declare_parameter('bev.global_size', 128)
        self.declare_parameter('bev.global_resolution', 1.5)
        self.declare_parameter('bev.global_channels', 3)
        
        # Map image parameters (for map_image architecture)
        self.declare_parameter('map.image_path', '')
        self.declare_parameter('map.feature_dim', 128)

        # --- Initialization ---
        input_dim = self.get_parameter('model.input_dim').value
        output_dim = self.get_parameter('model.output_dim').value
        state_dim = self.get_parameter('model.state_dim').value
        seq_len = self.get_parameter('model.seq_len').value
        hidden_size = self.get_parameter('model.hidden_size').value
        architecture = self.get_parameter('model.architecture').value
        ckpt_path = self.get_parameter('model.ckpt_path').value
        max_range = self.get_parameter('max_range').value
        acceleration = self.get_parameter('acceleration').value
        control_mode = self.get_parameter('control_mode').value
        
        self.debug = self.get_parameter('debug').value
        self.log_interval = self.get_parameter('log_interval_sec').value
        
        # BEV parameters
        map_path = self.get_parameter('bev.map_path').value
        local_bev_size = self.get_parameter('bev.local_size').value
        local_bev_resolution = self.get_parameter('bev.local_resolution').value
        local_bev_channels = self.get_parameter('bev.local_channels').value
        global_bev_size = self.get_parameter('bev.global_size').value
        global_bev_resolution = self.get_parameter('bev.global_resolution').value
        global_bev_channels = self.get_parameter('bev.global_channels').value
        
        # Map image parameters
        map_image_path = self.get_parameter('map.image_path').value
        map_feature_dim = self.get_parameter('map.feature_dim').value
        
        # Check architecture type
        arch_lower = architecture.lower()
        self.use_fusion = arch_lower == 'fusion'
        self.is_temporal = arch_lower in TEMPORAL_ARCHITECTURES
        self.is_bev = arch_lower in BEV_ARCHITECTURES
        self.is_map = arch_lower in MAP_ARCHITECTURES
        self.seq_len = seq_len if self.is_temporal else 1
        
        # BEV and temporal/fusion models require odom
        self.needs_odom = self.use_fusion or self.is_temporal or self.is_bev
        
        # Initialize BEV components if needed
        self.bev_generator: Optional[BEVGenerator] = None
        self.lane_data: Optional[LaneBoundaries] = None
        self.map_offset: Optional[Tuple[float, float]] = None
        
        if self.is_bev:
            if not map_path:
                self.get_logger().error("BEV architecture requires 'bev.map_path' parameter")
                raise ValueError("Missing bev.map_path for BEV architecture")
            
            try:
                # Load lane boundaries
                self.lane_data = load_lane_boundaries(map_path, auto_offset=True)
                self.map_offset = self.lane_data.offset
                self.get_logger().info(f"Loaded lane data from {map_path}, offset: {self.map_offset}")
                
                # Initialize BEV generator
                self.bev_generator = BEVGenerator(
                    local_grid_size=local_bev_size,
                    local_resolution=local_bev_resolution,
                    global_grid_size=global_bev_size,
                    global_resolution=global_bev_resolution,
                    local_channels=local_bev_channels,
                    global_channels=global_bev_channels
                )
                # Auto-compute map center for global BEV
                self.bev_generator.auto_compute_map_center(self.lane_data)
                self.get_logger().info(
                    f"BEV generator initialized. Local: {local_bev_size}x{local_bev_size}@{local_bev_resolution}m, "
                    f"Global: {global_bev_size}x{global_bev_size}@{global_bev_resolution}m"
                )
            except Exception as e:
                self.get_logger().error(f"Failed to initialize BEV components: {e}")
                raise e

        try:
            self.core = TinyLidarNetCore(
                input_dim=input_dim,
                output_dim=output_dim,
                state_dim=state_dim,
                seq_len=seq_len,
                hidden_size=hidden_size,
                architecture=architecture,
                ckpt_path=ckpt_path,
                acceleration=acceleration,
                control_mode=control_mode,
                max_range=max_range,
                local_bev_size=local_bev_size,
                local_bev_channels=local_bev_channels,
                global_bev_size=global_bev_size,
                global_bev_channels=global_bev_channels,
                map_feature_dim=map_feature_dim,
                map_image_path=map_image_path
            )
            self.get_logger().info(
                f"Core initialized. Arch: {architecture}, MaxRange: {max_range}, "
                f"SeqLen: {self.seq_len}, Temporal: {self.is_temporal}, BEV: {self.is_bev}, Map: {self.is_map}"
            )
        except Exception as e:
            self.get_logger().error(f"Failed to initialize core logic: {e}")
            raise e

        # --- Communication Setup ---
        self.inference_times = []
        self.last_log_time = self.get_clock().now()
        
        # Thread-safe storage for latest odometry data
        self._odom_lock = threading.Lock()
        self._latest_odom = None
        self._latest_pose = None  # For BEV: (x, y, yaw)
        
        # Frame buffers for temporal models
        if self.is_temporal:
            self._scan_buffer = deque(maxlen=self.seq_len)
            self._odom_buffer = deque(maxlen=self.seq_len)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.sub_scan = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos
        )
        self.pub_control = self.create_publisher(
            AckermannControlCommand, "/awsim/control_cmd", 1
        )
        
        # Subscribe to kinematic_state if needed (fusion, temporal, or BEV)
        if self.needs_odom:
            self.sub_odom = self.create_subscription(
                Odometry, "/localization/kinematic_state", self.odom_callback, qos
            )
            self.get_logger().info("Subscribed to /localization/kinematic_state")

        self.get_logger().info("TinyLidarNetNode is ready.")

    def odom_callback(self, msg: Odometry):
        """Callback for Odometry subscription.

        Stores the latest kinematic state for fusion/BEV models.

        Args:
            msg (Odometry): The incoming ROS 2 Odometry message.
        """
        # Extract features from Odometry message
        odom_features = np.array([
            # Position (3)
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            # Orientation quaternion (4)
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
            # Linear velocity (3)
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            # Angular velocity (3)
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ], dtype=np.float32)
        
        # Extract pose for BEV generation
        ego_x = msg.pose.pose.position.x
        ego_y = msg.pose.pose.position.y
        ego_yaw = quaternion_to_yaw(
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        
        with self._odom_lock:
            self._latest_odom = odom_features
            self._latest_pose = (ego_x, ego_y, ego_yaw)

    def scan_callback(self, msg: LaserScan):
        """Callback for LaserScan subscription.

        Processes the scan data via the core logic and publishes a control command.

        Args:
            msg (LaserScan): The incoming ROS 2 LaserScan message.
        """
        start_time = time.monotonic()

        # 1. Convert ROS message to Numpy
        # We pass the raw array; the core logic handles NaN/Inf and normalization.
        ranges = np.array(msg.ranges, dtype=np.float32)

        # 2. Process via Core Logic
        if self.is_bev:
            # BEV models: generate BEV grids from map + pose
            with self._odom_lock:
                odom = self._latest_odom if self._latest_odom is not None else np.zeros(13, dtype=np.float32)
                pose = self._latest_pose
            
            if pose is None:
                if self.debug:
                    self.get_logger().warn("No pose data available for BEV, waiting...", throttle_duration_sec=5.0)
                return
            
            ego_x, ego_y, ego_yaw = pose
            
            # Generate BEV grids based on architecture
            local_bev = None
            global_bev = None
            
            if self.core.architecture in ['local_bev', 'dual_bev']:
                local_bev = self.bev_generator.generate_local(
                    self.lane_data, ego_x, ego_y, ego_yaw, self.map_offset
                )
            
            if self.core.architecture in ['global_bev', 'dual_bev']:
                global_bev = self.bev_generator.generate_global(
                    self.lane_data, ego_x, ego_y, self.map_offset
                )
            
            accel, steer = self.core.process_with_bev(ranges, local_bev, global_bev, odom)
            
        elif self.is_temporal:
            # Temporal models: use frame buffers
            # Get latest odometry data (thread-safe)
            with self._odom_lock:
                odom = self._latest_odom if self._latest_odom is not None else np.zeros(13, dtype=np.float32)
            
            # Add to buffers
            self._scan_buffer.append(ranges)
            self._odom_buffer.append(odom.copy())
            
            # Check if we have enough frames
            if len(self._scan_buffer) < self.seq_len:
                if self.debug:
                    self.get_logger().info(
                        f"Buffering frames: {len(self._scan_buffer)}/{self.seq_len}",
                        throttle_duration_sec=1.0
                    )
                return  # Not enough frames yet
            
            # Stack buffers into sequences
            scan_seq = np.stack(list(self._scan_buffer), axis=0)  # (seq_len, scan_dim)
            odom_seq = np.stack(list(self._odom_buffer), axis=0)  # (seq_len, state_dim)
            
            accel, steer = self.core.process_sequence(scan_seq, odom_seq)
        elif self.use_fusion:
            # Fusion model: single frame with odom
            with self._odom_lock:
                odom = self._latest_odom
            
            if odom is None:
                odom = np.zeros(13, dtype=np.float32)
                if self.debug:
                    self.get_logger().warn("No odom data available, using zeros", throttle_duration_sec=5.0)
            
            accel, steer = self.core.process(ranges, odom)
        else:
            # Standard single-frame model
            accel, steer = self.core.process(ranges)

        # 3. Publish Command
        cmd = AckermannControlCommand()
        cmd.stamp = self.get_clock().now().to_msg()
        cmd.longitudinal.acceleration = float(accel)
        cmd.lateral.steering_tire_angle = float(steer)
        self.pub_control.publish(cmd)

        # 4. Debug Logging
        if self.debug:
            duration_ms = (time.monotonic() - start_time) * 1000.0
            self.inference_times.append(duration_ms)
            self._log_performance_metrics()

    def _log_performance_metrics(self):
        """Logs internal performance metrics at fixed intervals."""
        now = self.get_clock().now()
        elapsed_sec = (now - self.last_log_time).nanoseconds / 1e9

        if elapsed_sec > self.log_interval:
            if self.inference_times:
                avg_time = np.mean(self.inference_times)
                max_time = np.max(self.inference_times)
                fps = 1000.0 / avg_time if avg_time > 0 else 0.0

                self.get_logger().info(
                    f"DEBUG: Avg Inference: {avg_time:.2f}ms ({fps:.2f}Hz) | "
                    f"Max: {max_time:.2f}ms"
                )
                self.inference_times.clear()
            
            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = TinyLidarNetNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
