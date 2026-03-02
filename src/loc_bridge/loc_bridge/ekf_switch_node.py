#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion
import tf_transformations


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def angdiff(a: float, b: float) -> float:
    return wrap_pi(a - b)


def q_from_yaw(yaw: float) -> Quaternion:
    x, y, z, w = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
    q = Quaternion()
    q.x, q.y, q.z, q.w = float(x), float(y), float(z), float(w)
    return q


class EKFSwitchNode(Node):

    def __init__(self):
        super().__init__('ekf_switch_node')

        # ===== Parameters =====
        self.declare_parameter('window_deg0', 60.0)
        self.declare_parameter('window_deg180', 60.0)
        self.declare_parameter('yaw_cov', 0.001)
        self.declare_parameter('cooldown_sec', 2.0)

        self.declare_parameter('cam_jump_deg', 7.0)
        self.declare_parameter('cam_off_frames', 3)

        # ★ 追加：ロック時間
        self.declare_parameter('relock_sec', 10.0)

        self.win0 = math.radians(self.get_parameter('window_deg0').value)
        self.winpi = math.radians(self.get_parameter('window_deg180').value)
        self.yaw_cov = float(self.get_parameter('yaw_cov').value)
        self.cooldown_sec = float(self.get_parameter('cooldown_sec').value)

        self.cam_jump_th = math.radians(
            self.get_parameter('cam_jump_deg').value
        )
        self.cam_off_frames = int(
            self.get_parameter('cam_off_frames').value
        )

        self.relock_sec = float(
            self.get_parameter('relock_sec').value
        )

        qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # ===== Sub =====
        self.sub_vanish = self.create_subscription(
            Bool, '/cam/vanish_valid', self.on_vanish, qos
        )

        self.sub_cam = self.create_subscription(
            Float32, '/cam_theta', self.on_cam_theta, qos
        )

        self.sub_imu = self.create_subscription(
            Imu, '/um7/data', self.on_imu, qos
        )

        # ===== Pub =====
        self.pub_cam_pose = self.create_publisher(
            PoseWithCovarianceStamped, '/ekf/cam_pose', qos
        )

        # ===== State =====
        self.vanish = False
        self.vanish_prev = False

        self.imu_yaw = None

        self.offset = None
        self.last_cam_abs_unwrap = None

        self.row_index = 0
        self.last_snap_time = -1e9

        # jump guard
        self.prev_cam = None
        self.off_cnt = 0

        # ★ 追加：復帰ロック
        self.relock = False
        self.relock_start = 0.0

        self.get_logger().info(
            "✅ ekf_switch_node started (10s relock + cam jump guard)"
        )

    # ---- helpers ----

    def in_window0(self, ang: float) -> bool:
        return abs(wrap_pi(ang)) <= self.win0

    def in_windowpi(self, ang: float) -> bool:
        return abs(wrap_pi(ang - math.pi)) <= self.winpi

    def unwrap_to_ref(self, a_now: float, a_ref: float) -> float:
        candidates = [a_now + k * 2.0 * math.pi for k in (-2, -1, 0, 1, 2)]
        return min(candidates, key=lambda c: abs(c - a_ref))

    # ---- callbacks ----

    def on_imu(self, msg: Imu):

        q = msg.orientation

        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)

        self.imu_yaw = math.atan2(siny_cosp, cosy_cosp)

    def on_vanish(self, msg: Bool):

        self.vanish_prev = self.vanish
        self.vanish = bool(msg.data)

        now = time.time()

        # ★ 1 → 0：ロック開始
        if self.vanish_prev and (not self.vanish):

            self.relock = True
            self.relock_start = now

            self.offset = None
            self.prev_cam = None
            self.off_cnt = 0

            self.get_logger().info("[relock] start 10s lock")

        # 0→1 スナップ（従来処理）
        if (not self.vanish_prev) and self.vanish:

            if now - self.last_snap_time < self.cooldown_sec:
                return

            if self.imu_yaw is not None:

                if self.in_window0(self.imu_yaw):
                    tgt = 0.0

                elif self.in_windowpi(self.imu_yaw):
                    tgt = math.pi

                else:
                    tgt = wrap_pi(self.imu_yaw)

            else:
                tgt = 0.0

            self.offset = None
            self.last_cam_abs_unwrap = tgt
            self.last_snap_time = now
            self.row_index += 1

            self.prev_cam = None
            self.off_cnt = 0

            self.get_logger().info(
                f"[snap] row#{self.row_index} target={math.degrees(tgt):.1f} deg"
            )

    def on_cam_theta(self, msg: Float32):

        now = time.time()

        # ===== ロック中は完全無視 =====
        if self.relock:

            if now - self.relock_start < self.relock_sec:
                return

            if not self.vanish:
                return

            # 解除
            self.relock = False
            self.prev_cam = None
            self.off_cnt = 0

            self.get_logger().info("[relock] unlocked")

        # vanish無効なら使わない
        if not self.vanish:
            self.prev_cam = None
            return

        th_cam_rel = float(msg.data)

        # ===== jump guard =====
        if self.prev_cam is not None:

            d = abs(angdiff(th_cam_rel, self.prev_cam))

            if d > self.cam_jump_th:

                self.off_cnt = self.cam_off_frames

                self.get_logger().warn(
                    f"[cam_jump] |dθ|={math.degrees(d):.1f} deg"
                )

        self.prev_cam = th_cam_rel

        if self.off_cnt > 0:
            self.off_cnt -= 1
            return

        # ===== offset =====
        if self.offset is None:

            tgt = self.last_cam_abs_unwrap if self.last_cam_abs_unwrap is not None \
                else (wrap_pi(self.imu_yaw) if self.imu_yaw is not None else 0.0)

            self.offset = angdiff(tgt, th_cam_rel)

            self.get_logger().info(
                f"[snap] set offset={math.degrees(self.offset):.1f} deg"
            )

        # ===== absolute yaw =====
        yaw_abs = wrap_pi(th_cam_rel + self.offset)

        if self.last_cam_abs_unwrap is not None:
            yaw_abs = self.unwrap_to_ref(yaw_abs, self.last_cam_abs_unwrap)

        self.last_cam_abs_unwrap = yaw_abs

        # ===== publish =====
        pose = PoseWithCovarianceStamped()

        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'odom'

        pose.pose.pose.orientation = q_from_yaw(wrap_pi(yaw_abs))

        cov = [0.0] * 36
        cov[35] = self.yaw_cov

        pose.pose.covariance = cov

        self.pub_cam_pose.publish(pose)


def main():

    rclpy.init()

    node = EKFSwitchNode()

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

