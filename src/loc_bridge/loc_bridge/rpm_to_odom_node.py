#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Float32MultiArray, Float64MultiArray, Int32MultiArray
from geometry_msgs.msg import Quaternion, Twist, Vector3Stamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
import tf_transformations


def q_from_yaw(yaw: float) -> Quaternion:
    qx, qy, qz, qw = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
    q = Quaternion()
    q.x, q.y, q.z, q.w = float(qx), float(qy), float(qz), float(qw)
    return q


class RpmToOdomNode(Node):
    """
    /rpm または /rpm_twist /rpm_vec を受け取り、
    差動運動学から /wheel/odom_raw を生成。
    さらに /camera/twist を /camera/odom_from_twist に変換して出力。
    """

    def __init__(self) -> None:
        super().__init__('rpm_to_odom')

        # ===== パラメータ =====
        self.declare_parameter('rpm_msg_type', 'float32')
        self.declare_parameter('rpm_topic', '/rpm')
        self.declare_parameter('twist_topic', '/rpm_twist')
        self.declare_parameter('vec_topic', '/rpm_vec')

        self.declare_parameter('wheel_radius', 0.0635)    # [m]
        self.declare_parameter('wheel_separation', 0.35)  # [m]
        self.declare_parameter('publish_frame_id', 'odom')
        self.declare_parameter('publish_child_frame_id', 'base_link')

        self.declare_parameter('index_left', 0)
        self.declare_parameter('index_right', 1)
        self.declare_parameter('invert_left', False)
        self.declare_parameter('invert_right', False)
        self.declare_parameter('rpm_scale', 1.0)

        self.declare_parameter('vx_scale', 1.0)
        self.declare_parameter('wz_scale', 1.0)

        self.declare_parameter('camera_twist_topic', '/camera/twist_raw')
        self.declare_parameter('camera_odom_topic', '/camera/odom_from_twist')

        # ===== 読み出し =====
        self.rpm_msg_type = str(self.get_parameter('rpm_msg_type').value).lower()
        self.rpm_topic    = str(self.get_parameter('rpm_topic').value)
        twist_topic       = str(self.get_parameter('twist_topic').value)
        vec_topic         = str(self.get_parameter('vec_topic').value)

        self.R = float(self.get_parameter('wheel_radius').value)
        self.L = float(self.get_parameter('wheel_separation').value)
        self.frame_id = str(self.get_parameter('publish_frame_id').value)
        self.child_frame_id = str(self.get_parameter('publish_child_frame_id').value)

        self.idx_l = int(self.get_parameter('index_left').value)
        self.idx_r = int(self.get_parameter('index_right').value)
        self.inv_l = bool(self.get_parameter('invert_left').value)
        self.inv_r = bool(self.get_parameter('invert_right').value)
        self.rpm_scale = float(self.get_parameter('rpm_scale').value)

        self.vx_scale = float(self.get_parameter('vx_scale').value)
        self.wz_scale = float(self.get_parameter('wz_scale').value)

        self.cam_twist_topic = str(self.get_parameter('camera_twist_topic').value)
        self.cam_odom_topic  = str(self.get_parameter('camera_odom_topic').value)

        # ===== Publisher =====
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.odom_pub     = self.create_publisher(Odometry, '/wheel/odom_raw', qos)
        self.cam_odom_pub = self.create_publisher(Odometry, self.cam_odom_topic, qos)

        # ===== Subscriber（rpm入力）=====
        if self.rpm_msg_type == 'float64':
            self.sub_rpm = self.create_subscription(Float64MultiArray, self.rpm_topic, self.on_rpm_array_f64, qos)
        elif self.rpm_msg_type == 'float32':
            self.sub_rpm = self.create_subscription(Float32MultiArray, self.rpm_topic, self.on_rpm_array_f32, qos)
        elif self.rpm_msg_type == 'int32':
            self.sub_rpm = self.create_subscription(Int32MultiArray,  self.rpm_topic, self.on_rpm_array_i32, qos)
        else:
            raise RuntimeError(f"unsupported rpm_msg_type: {self.rpm_msg_type}")

        # 追加の代替入力
        self.sub_twist = self.create_subscription(Twist, twist_topic, self.on_twist, qos) if twist_topic else None
        self.sub_vec   = self.create_subscription(Vector3Stamped, vec_topic, self.on_vec, qos) if vec_topic else None

        # カメラ Twist→Odom ブリッジ
        self.sub_cam_twist = self.create_subscription(
            TwistWithCovarianceStamped, self.cam_twist_topic, self.on_cam_twist, qos
        )

        # 内部状態
        self.x = 0.0; self.y = 0.0; self.yaw = 0.0
        self.vx = 0.0; self.wz = 0.0
        self.last_stamp = self.get_clock().now()
        self.mode: Optional[str] = None

        self.timer = self.create_timer(0.05, self.on_timer)  # 20Hz

        self.get_logger().info("[rpm_to_odom] started, publishing /wheel/odom_raw and /camera/odom_from_twist")

    # ====== /rpm コールバック ======
    def on_rpm_array_common(self, data_list):
        if data_list is None or len(data_list) <= max(self.idx_l, self.idx_r):
            return
        rpm_l = float(data_list[self.idx_l]) * self.rpm_scale
        rpm_r = float(data_list[self.idx_r]) * self.rpm_scale
        if self.inv_l: rpm_l = -rpm_l
        if self.inv_r: rpm_r = -rpm_r

        wl = rpm_l * (2.0 * math.pi / 60.0)
        wr = rpm_r * (2.0 * math.pi / 60.0)

        self.vx = self.R * (wr + wl) / 2.0
        self.wz = self.R * (wr - wl) / self.L

    def on_rpm_array_f32(self, msg: Float32MultiArray): self.on_rpm_array_common(msg.data)
    def on_rpm_array_f64(self, msg: Float64MultiArray): self.on_rpm_array_common(msg.data)
    def on_rpm_array_i32(self, msg: Int32MultiArray): self.on_rpm_array_common(msg.data)

    # ====== 代替入力 ======
    def on_twist(self, msg: Twist):
        self.vx = float(msg.linear.x)  * self.vx_scale
        self.wz = float(msg.angular.z) * self.wz_scale

    def on_vec(self, msg: Vector3Stamped):
        self.on_rpm_array_common([msg.vector.x, msg.vector.y])

    # ====== カメラ Twist→Odom ブリッジ ======
    def on_cam_twist(self, msg: TwistWithCovarianceStamped):
        od = Odometry()
        od.header = msg.header
        od.child_frame_id = 'base_link'
        od.pose.covariance = [0.0]*36
        od.twist = msg.twist
        self.cam_odom_pub.publish(od)

    # ====== 周期 publish ======
    def on_timer(self) -> None:
        now = self.get_clock().now()
        dt = (now - self.last_stamp).nanoseconds * 1e-9
        self.last_stamp = now

        self.yaw += self.wz * dt
        self.x   += self.vx * math.cos(self.yaw) * dt * 1
        self.y   += self.vx * math.sin(self.yaw) * dt * 1

        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.frame_id
        odom.child_frame_id = self.child_frame_id

        odom.pose.pose.position.x = float(self.x)
        odom.pose.pose.position.y = float(self.y)
        odom.pose.pose.position.z = 0.0
        odom.pose.pose.orientation = q_from_yaw(self.yaw)

        pcov = [0.0]*36
        pcov[0] = 1e2; pcov[7] = 1e2; pcov[14] = 1e6; pcov[21] = 1e6; pcov[28] = 1e6; pcov[35] = 0.5
        odom.pose.covariance = pcov

        odom.twist.twist.linear.x  = float(self.vx)
        odom.twist.twist.angular.z = float(self.wz)

        tcov = [0.0]*36
        tcov[0] = 0.1; tcov[35] = 0.5
        odom.twist.covariance = tcov

        self.odom_pub.publish(odom)


def main() -> None:
    rclpy.init()
    node = RpmToOdomNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

