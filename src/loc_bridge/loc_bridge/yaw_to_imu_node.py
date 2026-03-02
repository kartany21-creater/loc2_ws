#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from std_msgs.msg import Header, Bool
import serial
import math
from collections import deque

# ---------- helpers ----------
def rpy_to_quat(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def wrap_pi(a: float):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


def unwrap_to_ref(a_now: float, a_ref: float) -> float:
    candidates = (a_now - 2*math.pi, a_now, a_now + 2*math.pi)
    return min(candidates, key=lambda c: abs(c - a_ref))


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class YawToImuNode(Node):

    def __init__(self):
        super().__init__('yaw_to_imu')

        # ===== parameters =====
        self.declare_parameter('serial_port', '/dev/ttyUSB1')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('frame_id', 'base_link')

        self.declare_parameter('stable_window_len', 50)
        self.declare_parameter('stable_std_deg', 2.1)
        self.declare_parameter('stable_wz_dps', 1.18)

        self.declare_parameter('imu_yaw_cov', 1.0e9)
        self.declare_parameter('imu_wz_cov', 1.0e9)

        self.port = self.get_parameter('serial_port').get_parameter_value().string_value
        self.baud = int(self.get_parameter('baudrate').get_parameter_value().integer_value
                        or 115200)
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value

        self.stable_window_len = int(self.get_parameter('stable_window_len').value)
        self.stable_std_deg = float(self.get_parameter('stable_std_deg').value)
        self.stable_wz_dps = float(self.get_parameter('stable_wz_dps').value)

        self.imu_yaw_cov = float(self.get_parameter('imu_yaw_cov').value)
        self.imu_wz_cov = float(self.get_parameter('imu_wz_cov').value)

        # ===== pubs / subs =====
        self.pub_imu = self.create_publisher(Imu, '/um7/data', 50)

        self.sub_vanish = self.create_subscription(Bool, '/cam/vanish_valid',
                                                   self._on_vanish, 10)
        self.sub_ekf = self.create_subscription(Odometry, '/odometry/filtered',
                                                self._on_ekf, 10)

        # ===== serial =====
        self.ser = None
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=0.01)
            self.ser.reset_input_buffer()
            self.get_logger().info(f"✅ UM7 接続完了 ({self.port}, {self.baud}bps)")
        except Exception as e:
            self.get_logger().error(f"❌ UM7 シリアル接続失敗: {e}")
        self._rx_buf = bytearray()

        # ===== state =====
        self.att_yaw_deg = None
        self.att_roll_deg = 0.0
        self.wz_dps = 0.0

        self.yaw_unwrap = None
        self.yaw0_offset_deg = 0.0

        # ★ Roll 用オフセット（Yaw と同じ方式）
        self.roll0_offset_deg = 0.0

        self.yaw = 0.0
        self.roll = 0.0

        self.imu_unwrap = None
        self.ekf_unwrap = None
        self.align_offset = None

        self.vanish = True
        self.prev_vanish = True

        self.yaw_win = deque(maxlen=self.stable_window_len)
        self.wz_win = deque(maxlen=self.stable_window_len)
        self.initialized = False
        self.logged_waiting = False

        # timer
        self.timer = self.create_timer(0.01, self._on_timer)

    # ========= subscribers =========

    def _on_vanish(self, msg: Bool):
        self.prev_vanish = self.vanish
        self.vanish = bool(msg.data)

        if self.prev_vanish and not self.vanish:
            if self.imu_unwrap is not None and self.ekf_unwrap is not None:
                self.align_offset = self.ekf_unwrap - self.imu_unwrap
                self.get_logger().info("✅ vanish OFF → EKF yaw と同期")

        if (not self.prev_vanish) and self.vanish:
            self.align_offset = None
            self.get_logger().info("✅ vanish ON → IMU 単独 yaw に戻す")

    def _on_ekf(self, msg: Odometry):
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        if self.ekf_unwrap is None:
            self.ekf_unwrap = yaw
        else:
            self.ekf_unwrap = unwrap_to_ref(yaw, self.ekf_unwrap)

    # ========= serial parsing =========

    def _read_serial(self):
        if self.ser is None:
            return
        try:
            data = self.ser.read(1024)
            if not data:
                return
            self._rx_buf.extend(data)
            lines = self._rx_buf.split(b'\n')
            self._rx_buf = lines[-1]

            for ln in lines[:-1]:
                s = ln.decode(errors='ignore').strip()
                if not s.startswith('$PCHR'):
                    continue

                parts = s.split(',')
                header = parts[0]

                # $PCHRA,time,roll,pitch,yaw,heading
                if header == '$PCHRA' and len(parts) >= 6:
                    try:
                        self.att_roll_deg = float(parts[2])
                        self.att_yaw_deg = float(parts[4])
                    except ValueError:
                        continue

                # $PCHRP,time,pn,pe,alt,roll,pitch,yaw,heading
                elif header == '$PCHRP' and len(parts) >= 9:
                    try:
                        self.att_roll_deg = float(parts[5])
                        self.att_yaw_deg = float(parts[7])
                    except ValueError:
                        continue

                # $PCHRR,time,vn,ve,vup,roll_rate,pitch_rate,yaw_rate
                elif header == '$PCHRR' and len(parts) >= 8:
                    try:
                        self.wz_dps = float(parts[7].split('*')[0])
                    except ValueError:
                        continue

        except Exception as e:
            self.get_logger().warn(f"UM7 read error: {e}")

    # ========= main timer =========

    def _on_timer(self):
        self._read_serial()

        if self.att_yaw_deg is None:
            return

        # unwrap 初期化
        if self.yaw_unwrap is None:
            self.yaw_unwrap = math.radians(self.att_yaw_deg)

        # 安定性評価用バッファ
        self.yaw_win.append(self.att_yaw_deg)
        self.wz_win.append(abs(self.wz_dps or 0.0))

        # ===== Yaw 安定化フェーズ =====
        if not self.initialized:
            if len(self.yaw_win) < self.yaw_win.maxlen:
                if not self.logged_waiting:
                    self.get_logger().info("⏳ Yaw 安定化待機中... (サンプル収集中)")
                    self.logged_waiting = True
                return

            yaw_span = max(self.yaw_win) - min(self.yaw_win)
            wz_max = max(self.wz_win)

            if yaw_span < self.stable_std_deg and wz_max < self.stable_wz_dps:
                self.yaw0_offset_deg  = self.att_yaw_deg
               # self.roll0_offset_deg = self.att_roll_deg   # ★ Roll も同時に 0 初期化
                self.initialized = True
                self.get_logger().info(
                    f"✅ IMU 安定化完了 → Yaw=0°, Roll=0° "
                    f"(span={yaw_span:.4f}deg, |wz|max={wz_max:.4f}deg/s)"
                )
            else:
                if not self.logged_waiting or (len(self.yaw_win) % 10 == 0):
                    self.get_logger().info(
                        f"⏳ Yaw 安定化待機中... "
                        f"(span={yaw_span:.4f}deg, |wz|max={wz_max:.4f}deg/s)"
                    )
                    self.logged_waiting = True
            return

        # ===== 安定化後の角度計算 =====
        yaw_base_rad = math.radians(self.att_yaw_deg - self.yaw0_offset_deg)
        roll_base_rad = math.radians(self.att_roll_deg)

        if (not self.vanish) and (self.align_offset is not None):
            yaw_out_rad = yaw_base_rad + self.align_offset
        else:
            yaw_out_rad = yaw_base_rad

        self.yaw = wrap_pi(yaw_out_rad)
        self.roll = roll_base_rad

        if self.imu_unwrap is None:
            self.imu_unwrap = self.yaw
        else:
            self.imu_unwrap = unwrap_to_ref(self.yaw, self.imu_unwrap)

        wz_rad = math.radians(self.wz_dps or 0.0)

        # ===== IMU publish =====
        msg = Imu()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        qx, qy, qz, qw = rpy_to_quat(self.roll, 0.0, self.yaw)
        msg.orientation.x = qx
        msg.orientation.y = qy
        msg.orientation.z = qz
        msg.orientation.w = qw

        msg.orientation_covariance = [
            99999.0, 0.0,     0.0,
            0.0,     99999.0, 0.0,
            0.0,     0.0,     self.imu_yaw_cov
        ]

        msg.angular_velocity.x = 0.0
        msg.angular_velocity.y = 0.0
        msg.angular_velocity.z = wz_rad
        msg.angular_velocity_covariance = [
            99999.0, 0.0,      0.0,
            0.0,     99999.0,  0.0,
            0.0,     0.0,      self.imu_wz_cov
        ]

        msg.linear_acceleration_covariance = [
            -1.0, 0.0, 0.0,
             0.0,-1.0, 0.0,
             0.0, 0.0,-1.0
        ]

        self.pub_imu.publish(msg)


def main():
    rclpy.init()
    node = YawToImuNode()
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

