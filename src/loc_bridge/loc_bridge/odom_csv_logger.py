#!/usr/bin/env python3
import os
import csv
import math
from datetime import datetime
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32
from nav_msgs.msg import Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


def wrap_pi(a: float) -> float:
    """(-pi, pi] wrap"""
    a = (a + math.pi) % (2.0 * math.pi) - math.pi
    return a


def yaw_from_quat(q) -> float:
    """quaternion -> yaw(rad)"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def roll_from_quat(q) -> float:
    """quaternion -> roll(rad)"""
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    return math.atan2(sinr_cosp, cosr_cosp)


class OdomCsvLogger(Node):
    def __init__(self):
        super().__init__('odom_csv_logger')

        # ========= 状態 =========
        self.x = 0.0
        self.y = 0.0
        self.ekf_yaw = None
        self.ekf_vx = None
        self.ekf_wz = None

        # 個別センサ
        self.imu_yaw = None
        self.imu_roll = None
        self.cam_yaw = None
        self.cam_vel = None
        self.enc_vel = None
        self.enc_x = None
        self.enc_y = None
        self.enc_yaw = None
        self.vanish = None

        # ★ 追加：cmd_vel
        self.cmd_vx = None
        self.cmd_wz = None

        # 参考用 カメラ積算（= X_cam）
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.last_time = None

        # エンコーダ由来X積算
        self.X_enc = 0.0
        self._last_time_xenc = None

        # ΔZ・融合距離S・最終オドムXY
        self.ds_raw = 0.0
        self.s_fused = 0.0
        self._last_s_fused = None
        self.S_cam = 0.0
        self.S_enc = 0.0
        self._last_time_s = None
        self.odom_x = 0.0
        self.odom_y = 0.0

        # ロール補正なしのXY (/odom_raw)
        self.odom_raw_x = 0.0
        self.odom_raw_y = 0.0

        # IMU / Cam で積算した参考XY
        self.xy_imu_x = 0.0
        self.xy_imu_y = 0.0
        self.xy_cam_x = 0.0
        self.xy_cam_y = 0.0

        # cam yaw unwrap
        self._cam_deg_last = None
        self._cam_deg_unwrap = None

        # plan
        self._last_plan = None

        # ========= QoS =========
        qos = QoSProfile(
            depth=50,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        # ========= Subscriber =========
        self.create_subscription(Odometry, '/odometry/filtered', self.cb_ekf, qos)
        self.create_subscription(Imu, '/um7/data', self.cb_imu, qos)
        self.create_subscription(PoseWithCovarianceStamped, '/ekf/cam_pose', self.cb_cam_pose, qos)
        self.create_subscription(Odometry, '/camera/odom_from_twist', self.cb_cam_odom, qos)
        self.create_subscription(Odometry, '/wheel/odom_raw', self.cb_enc_odom, qos)
        self.create_subscription(Bool, '/cam/vanish_valid', self.cb_vanish, qos)
        self.create_subscription(Float32, '/camera/ds_raw', self.cb_ds_raw, qos)
        self.create_subscription(Float32, '/forward/s_fused', self.cb_s_fused, qos)
        self.create_subscription(Odometry, '/odom', self.cb_odom, qos)
        self.create_subscription(Odometry, '/odom_raw', self.cb_odom_raw, qos)

        plan_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        self.create_subscription(Path, '/plan', self.cb_plan, plan_qos)

        # ★ 追加：cmd_vel
        self.create_subscription(Twist, '/cmd_vel', self.cb_cmd_vel, qos)

        # ========= CSV =========
        ws_root = os.path.expanduser('~/loc2_ws')
        save_dir = os.path.join(ws_root, 'logs')
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(save_dir, f'odom_log_{timestamp}.csv')
        self._fh = open(self.csv_path, 'w', newline='')
        self._writer = csv.writer(self._fh)

        self._writer.writerow([
            'time',
            'x','y',
            'ekf_yaw(rad)','ekf_yaw(deg)','ekf_vx(m/s)','ekf_wz(rad/s)',
            'imu_yaw(rad)','imu_yaw(deg)',
            'imu_roll(deg)',
            'cam_yaw(rad)','cam_yaw(deg)','cam_yaw_unwrap(deg)',
            'cam_vel(m/s)','enc_vel(m/s)',
            'enc_x(m)','enc_y(m)','enc_yaw(rad)','enc_yaw(deg)',
            'X_cam(m)','X_enc(m)',
            'cam_y(m)',
            'vanish_valid',
            'ds_raw(m)','s_fused(m)',
            'odom_x(m)','odom_y(m)',
            'odom_raw_x(m)','odom_raw_y(m)',
            'xy_imu_x(m)','xy_imu_y(m)',
            'xy_cam_x(m)','xy_cam_y(m)',
            'S_cam(m)','S_enc(m)',
            'plan_x(m)', 'plan_y(m)',
            # ★ 追加
            'cmd_vx(m/s)', 'cmd_wz(rad/s)'
        ])

        self._fh.flush()
        os.fsync(self._fh.fileno())

        self.timer = self.create_timer(0.1, self._tick_save)

    # ========= Callbacks =========
    def cb_ekf(self, msg: Odometry):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.ekf_yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.ekf_vx = msg.twist.twist.linear.x
        self.ekf_wz = msg.twist.twist.angular.z

    def cb_imu(self, msg: Imu):
        self.imu_yaw = yaw_from_quat(msg.orientation)
        self.imu_roll = roll_from_quat(msg.orientation)

    def cb_cam_pose(self, msg: PoseWithCovarianceStamped):
        self.cam_yaw = yaw_from_quat(msg.pose.pose.orientation)

    def cb_cam_odom(self, msg: Odometry):
        self.cam_vel = msg.twist.twist.linear.x
        if self.ekf_yaw is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.last_time is None:
            self.last_time = now
            return
        dt = now - self.last_time
        self.last_time = now
        self.cam_x += self.cam_vel * math.cos(self.ekf_yaw) * dt
        self.cam_y += self.cam_vel * math.sin(self.ekf_yaw) * dt

    def cb_enc_odom(self, msg: Odometry):
        self.enc_vel = msg.twist.twist.linear.x
        self.enc_x = msg.pose.pose.position.x
        self.enc_y = msg.pose.pose.position.y
        self.enc_yaw = yaw_from_quat(msg.pose.pose.orientation)

    def cb_vanish(self, msg: Bool):
        self.vanish = bool(msg.data)

    def cb_ds_raw(self, msg: Float32):
        self.ds_raw = float(msg.data)
        self.S_cam += float(msg.data)

    def cb_s_fused(self, msg: Float32):
        self.s_fused = float(msg.data)

    def cb_odom(self, msg: Odometry):
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y

    def cb_odom_raw(self, msg: Odometry):
        self.odom_raw_x = msg.pose.pose.position.x
        self.odom_raw_y = msg.pose.pose.position.y

    def cb_plan(self, msg: Path):
        if not msg.poses:
            return
        p = msg.poses[-1].pose.position
        self._last_plan = (float(p.x), float(p.y))

    # ★ 追加
    def cb_cmd_vel(self, msg: Twist):
        self.cmd_vx = msg.linear.x
        self.cmd_wz = msg.angular.z

    # ========= CSV tick =========
    def _tick_save(self):
        t = self.get_clock().now().nanoseconds * 1e-9

        if self._last_time_xenc is not None and self.enc_vel is not None and self.ekf_yaw is not None:
            dt_x = t - self._last_time_xenc
            self.X_enc += self.enc_vel * math.cos(self.ekf_yaw) * dt_x
        self._last_time_xenc = t

        if self._last_time_s is not None and self.enc_vel is not None:
            dt_s = t - self._last_time_s
            self.S_enc += self.enc_vel * dt_s
        self._last_time_s = t

        ekf_deg = (self.ekf_yaw*180.0/math.pi) if self.ekf_yaw is not None else None
        imu_deg = (self.imu_yaw*180.0/math.pi) if self.imu_yaw is not None else None
        imu_roll_deg = (self.imu_roll*180.0/math.pi) if self.imu_roll is not None else None
        cam_deg = (self.cam_yaw*180.0/math.pi) if self.cam_yaw is not None else None
        enc_yaw_deg = (self.enc_yaw*180.0/math.pi) if self.enc_yaw is not None else None

        cam_deg_unwrap = ''
        if cam_deg is not None:
            if self._cam_deg_unwrap is None:
                self._cam_deg_unwrap = cam_deg
                self._cam_deg_last = cam_deg
            else:
                delta = cam_deg - self._cam_deg_last
                while delta >= 180.0:
                    delta -= 360.0
                while delta < -180.0:
                    delta += 360.0
                self._cam_deg_unwrap += delta
                self._cam_deg_last = cam_deg
            cam_deg_unwrap = f'{self._cam_deg_unwrap:.3f}'

        plan_x, plan_y = self._last_plan if self._last_plan is not None else ('','')

        row = [
            f'{t:.6f}',
            f'{self.x:.6f}', f'{self.y:.6f}',
            (f'{self.ekf_yaw:.6f}' if self.ekf_yaw is not None else ''),
            (f'{ekf_deg:.3f}' if ekf_deg is not None else ''),
            (f'{self.ekf_vx:.6f}' if self.ekf_vx is not None else ''),
            (f'{self.ekf_wz:.6f}' if self.ekf_wz is not None else ''),
            (f'{self.imu_yaw:.6f}' if self.imu_yaw is not None else ''),
            (f'{imu_deg:.3f}' if imu_deg is not None else ''),
            (f'{imu_roll_deg:.3f}' if imu_roll_deg is not None else ''),
            (f'{self.cam_yaw:.6f}' if self.cam_yaw is not None else ''),
            (f'{cam_deg:.3f}' if cam_deg is not None else ''),
            cam_deg_unwrap,
            (f'{self.cam_vel:.6f}' if self.cam_vel is not None else ''),
            (f'{self.enc_vel:.6f}' if self.enc_vel is not None else ''),
            (f'{self.enc_x:.6f}' if self.enc_x is not None else ''),
            (f'{self.enc_y:.6f}' if self.enc_y is not None else ''),
            (f'{self.enc_yaw:.6f}' if self.enc_yaw is not None else ''),
            (f'{enc_yaw_deg:.3f}' if enc_yaw_deg is not None else ''),
            f'{self.cam_x:.6f}',
            f'{self.X_enc:.6f}',
            f'{self.cam_y:.6f}',
            (str(int(self.vanish)) if self.vanish is not None else ''),
            f'{self.ds_raw:.6f}', f'{self.s_fused:.6f}',
            f'{self.odom_x:.6f}', f'{self.odom_y:.6f}',
            f'{self.odom_raw_x:.6f}', f'{self.odom_raw_y:.6f}',
            f'{self.xy_imu_x:.6f}', f'{self.xy_imu_y:.6f}',
            f'{self.xy_cam_x:.6f}', f'{self.xy_cam_y:.6f}',
            f'{self.S_cam:.6f}', f'{self.S_enc:.6f}',
            plan_x, plan_y,
            # ★ 追加
            (f'{self.cmd_vx:.6f}' if self.cmd_vx is not None else ''),
            (f'{self.cmd_wz:.6f}' if self.cmd_wz is not None else '')
        ]

        self._writer.writerow(row)
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def destroy_node(self):
        try:
            self._fh.close()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = OdomCsvLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

