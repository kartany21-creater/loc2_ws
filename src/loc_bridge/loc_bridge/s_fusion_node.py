#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import collections

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

import tf_transformations

# ===== 追加: TF broadcast 用 =====
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster


def yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def q_from_yaw(yaw: float):
    x, y, z, w = tf_transformations.quaternion_from_euler(0.0, 0.0, yaw)
    return (x, y, z, w)


class ForwardSFusionNode(Node):
    """
    入力:
      - /camera/ds_raw       : カメラの前進移動量ΔX [m]（各周期の差分）※あなたの希望：SではなくXとして扱う
      - /wheel/odom_raw      : エンコーダ速度 vx [m/s]（向きは持たない）
      - /cam/vanish_valid    : 消失点ON/OFF（直進/旋回の判定）
      - /cam_dx              : 畝間中心線に対する横ずれdx [m]（相対観測）
      - /um7/data            : IMU（旋回中のYawに使用）
      - /odometry/filtered   : 参考Yaw（ログ/姿勢表示用に残す。直進中のcos(YAW)投影に使うならこちらを使う）

    仕様（あなたの希望）:
      - 直進（vanish_valid=True）:
          X: カメラX（ds_raw積算） と エンコーダS×cos(YAW) を 1D EKFで融合
          Y: dxのみ（差分反映）→ Y(t) = Y_enter + (dx(t) - dx_enter)
      - 旋回（vanish_valid=False）:
          エンコーダ並進ΔS と IMU yaw で XY を積算
            X += ΔS * cos(yaw_imu)
            Y += ΔS * sin(yaw_imu)
      - OFF→ON で基準張替え（ジャンプ防止）:
          Y_enter を保持し、dx_enter を更新
          ★追加: X系（X_cam/X_enc/X_fused）も旋回終了時点のXに同期してから直進EKF再開
    """

    def __init__(self):
        super().__init__('s_fusion_node')

        # ===== タイマ/基本 =====
        self.declare_parameter('timer_dt', 0.1)   # [s]
        self.declare_parameter('deadband_v', 0.03)  # [m/s] エンコーダ微小速度の無視
        self.declare_parameter('enc_scale', 1.11)   # 例: 補正係数
        self.dt = float(self.get_parameter('timer_dt').value)
        self.deadband_v = float(self.get_parameter('deadband_v').value)
        self.enc_scale = float(self.get_parameter('enc_scale').value)

        # ===== ΔX(カメラ) フィルタ（LPF） =====
        self.declare_parameter('dx_clip_max', 0.20)   # [m/step]
        self.declare_parameter('dx_deadband', 0.01)   # [m/step]
        self.declare_parameter('dx_fc_hz', 0.8)       # [Hz]
        self.dx_clip_max = float(self.get_parameter('dx_clip_max').value)
        self.dx_deadband = float(self.get_parameter('dx_deadband').value)
        self.dx_fc = float(self.get_parameter('dx_fc_hz').value)
        self.alpha = math.exp(-2.0 * math.pi * self.dx_fc / max(1e-6, 1.0 / self.dt))
        self._dx_medbuf = collections.deque(maxlen=3)
        self._dx_f = 0.0

        # ===== 1次元EKF（X） =====
        self.declare_parameter('q_var', 1.0e-4)
        self.declare_parameter('r_cam', 4.0e-4)
        self.declare_parameter('r_enc', 1.0e-4)
        self.q_var = float(self.get_parameter('q_var').value)
        self.r_cam = float(self.get_parameter('r_cam').value)
        self.r_enc = float(self.get_parameter('r_enc').value)

        # ===== QoS/IO =====
        qos = QoSProfile(depth=50, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.sub_cam_dxstep = self.create_subscription(Float32, '/camera/ds_raw', self.on_cam_ds, qos)
        self.sub_enc = self.create_subscription(Odometry, '/wheel/odom_raw', self.on_enc_odom, qos)
        self.sub_vanish = self.create_subscription(Bool, '/cam/vanish_valid', self.on_vanish, qos)
        self.sub_dx = self.create_subscription(Float32, '/cam_dx', self.on_cam_dx, qos)

        self.sub_imu = self.create_subscription(Imu, '/um7/data', self.on_imu, qos)
        self.sub_ekf_odom = self.create_subscription(Odometry, '/odometry/filtered', self.on_ekf_odom, qos)

        self.pub_S = self.create_publisher(Float32, '/forward/s_fused', 10)      # 互換のため残す（ここでは「ΔS_enc積算」を出す）
        self.pub_odom_xy = self.create_publisher(Odometry, '/forward/odom_sxy', qos)
        self.pub_odom = self.create_publisher(Odometry, '/odom', qos)
        self.pub_odom_raw = self.create_publisher(Odometry, '/odom_raw', qos)   # 比較用

        # ===== 追加: TF broadcaster（odom -> base_link） =====
        self.tf_br = TransformBroadcaster(self)

        # ===== 状態 =====
        # センサ
        self.v_enc = 0.0
        self.yaw_imu = None
        self.yaw_ref = None   # /odometry/filtered の yaw（直進中のcos投影に使うならこれ）
        self._dx_buf = 0.0    # カメラΔXのバッファ

        # vanish と dx 状態
        self.vanish_valid = False
        self.prev_vanish = False
        self.dx_m = None
        self.dx_enter_m = None
        self.Y_enter = 0.0

        # 積算
        self.S_enc = 0.0      # エンコーダの距離（向きなし）
        self.X_cam = 0.0      # カメラX（直進時のみ有効）
        self.X_enc = 0.0      # エンコーダをXに投影した積算（直進時）
        self.X = 0.0
        self.Y = 0.0

        # 1D EKF: 状態 X_fused, 共分散 P
        self.X_fused = 0.0
        self.P = 1e-3

        # 比較用（生積分）
        self.X_raw = 0.0
        self.Y_raw = 0.0

        self.rows = 0
        self.timer = self.create_timer(self.dt, self.on_timer)
        self.get_logger().info("✅ s_fusion_node started (Straight: X EKF(camX & enc*cosYaw), Y=dx only / Turn: enc & IMU yaw integrate)")

    # ---- カメラΔXフィルタ ----
    def _filt_dx(self, dx_raw: float) -> float:
        dx = max(-self.dx_clip_max, min(self.dx_clip_max, dx_raw))
        if abs(dx) < self.dx_deadband:
            dx = 0.0
        self._dx_medbuf.append(dx)
        dx_med = sorted(self._dx_medbuf)[len(self._dx_medbuf) // 2]
        self._dx_f = self.alpha * self._dx_f + (1.0 - self.alpha) * dx_med
        return self._dx_f

    # ---- Callbacks ----
    def on_cam_ds(self, msg: Float32):
        # あなたの希望：camera/ds_raw は「ΔS」ではなく「ΔX」として扱う
        dx_f = self._filt_dx(float(msg.data))
        self._dx_buf += dx_f

    def on_enc_odom(self, msg: Odometry):
        self.v_enc = float(msg.twist.twist.linear.x)

    def on_vanish(self, msg: Bool):
        self.vanish_valid = bool(msg.data)

    def on_cam_dx(self, msg: Float32):
        self.dx_m = float(msg.data)

    def on_imu(self, msg: Imu):
        self.yaw_imu = yaw_from_quat(msg.orientation)

    def on_ekf_odom(self, msg: Odometry):
        # 直進中の cos(YAW) 投影に使う「参考Yaw」
        self.yaw_ref = yaw_from_quat(msg.pose.pose.orientation)

    # ---- 1D EKF(X) ----
    def ekf_predict(self):
        self.P += self.q_var

    def ekf_update(self, z: float, r: float):
        K = self.P / (self.P + r)
        self.X_fused = self.X_fused + K * (z - self.X_fused)
        self.P = (1.0 - K) * self.P

    # ---- Main timer ----
    def on_timer(self):
        dt = self.dt

        # エンコーダ速度 deadband
        v_enc = self.v_enc if abs(self.v_enc) >= self.deadband_v else 0.0
        dS_enc = v_enc * dt * self.enc_scale
        self.S_enc += dS_enc

        # 旋回中に使う yaw（必ずIMU）
        yaw_turn = self.yaw_imu

        # 直進中に使う yaw（cos投影用）：現状は /odometry/filtered を使う
        yaw_straight = self.yaw_ref

        # vanish遷移（OFF→ON）
        entered_vanish = (self.vanish_valid and (not self.prev_vanish))

        # ========= 旋回（vanish OFF） =========
        if (not self.vanish_valid):
            # 旋回中は「エンコーダΔS + IMU yaw」でXY積算する
            if yaw_turn is not None:
                self.X += dS_enc * math.cos(yaw_turn)
                self.Y += dS_enc * math.sin(yaw_turn)

                # 比較用（同じもの）
                self.X_raw += dS_enc * math.cos(yaw_turn)
                self.Y_raw += dS_enc * math.sin(yaw_turn)

            # 旋回中はカメラXやEKF更新はしない（直進ロジックへ混ざるのを防ぐ）
            self._dx_buf = 0.0

        # ========= 直進（vanish ON） =========
        else:
            # ★★ 追加：旋回→直進に入った瞬間、X系の内部状態を「旋回終了時点のX」に同期 ★★
            if entered_vanish:
                x0 = float(self.X)  # 旋回で積算された「現在の絶対X」を基準にする
                self.X_cam = x0
                self.X_enc = x0
                self.X_fused = x0
                self.P = 1e-3       # EKF共分散も初期化（暴れ防止）
                # 念のため：直進初回で古いΔXが混ざらないように
                self._dx_buf = 0.0

            # 直進中のカメラΔXを積算
            dX_cam = self._dx_buf
            self._dx_buf = 0.0
            self.X_cam += dX_cam

            # エンコーダは「距離S」なので、Xに使う前に cos(yaw) を掛けてXへ投影する
            if yaw_straight is not None:
                dX_enc = dS_enc * math.cos(yaw_straight)
            else:
                dX_enc = 0.0
            self.X_enc += dX_enc

            # 直進X：X_cam と X_enc を 1D EKF で融合
            self.ekf_predict()
            self.ekf_update(self.X_cam, self.r_cam)
            self.ekf_update(self.X_enc, self.r_enc)

            # 本命Xは融合結果
            self.X = float(self.X_fused)

            # 直進Y：dx差分のみ（絶対Y保持＋基準張替え）
            if self.dx_m is not None:
                if entered_vanish or (self.dx_enter_m is None):
                    self.dx_enter_m = self.dx_m
                    self.Y_enter = self.Y
                else:
                    self.Y = self.Y_enter + (self.dx_m - self.dx_enter_m)
            # dx未受信なら保持

            # 比較用：直進中に「yaw展開」したらどうなるか（デバッグ用）
            # （あなたの希望ではここは“使わない”が、比較表示用に残す）
            if yaw_straight is not None:
                self.X_raw += dX_enc
                self.Y_raw += dS_enc * math.sin(yaw_straight)

        # 次周期へ
        self.prev_vanish = self.vanish_valid

        # ===== Publish =====
        # 互換：/forward/s_fused は「S_enc」を出す（名前がSなので）
        self.pub_S.publish(Float32(data=float(self.S_enc)))

        od = Odometry()
        od.header.stamp = self.get_clock().now().to_msg()
        od.header.frame_id = 'odom'
        od.child_frame_id = 'base_link'
        od.pose.pose.position.x = float(self.X)
        od.pose.pose.position.y = float(self.Y)

        # 姿勢は見た目用：直進は yaw_ref、旋回は yaw_imu
        yaw_out = None
        if self.vanish_valid:
            yaw_out = yaw_straight
        else:
            yaw_out = yaw_turn

        if yaw_out is not None:
            qx, qy, qz, qw = q_from_yaw(yaw_out)
            od.pose.pose.orientation.x = qx
            od.pose.pose.orientation.y = qy
            od.pose.pose.orientation.z = qz
            od.pose.pose.orientation.w = qw

        od.twist.twist.linear.x = float(v_enc)
        od.twist.twist.angular.z = 0.0

        pcov = [0.0] * 36
        pcov[0] = 0.1
        pcov[7] = 0.1
        pcov[35] = 0.2
        od.pose.covariance = pcov

        tcov = [0.0] * 36
        tcov[0] = 0.05
        tcov[35] = 0.2
        od.twist.covariance = tcov

        self.pub_odom_xy.publish(od)
        self.pub_odom.publish(od)

        # ===== 追加: TF publish (odom -> base_link) =====
        t = TransformStamped()
        t.header.stamp = od.header.stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'
        t.transform.translation.x = float(self.X)
        t.transform.translation.y = float(self.Y)
        t.transform.translation.z = 0.0

        if yaw_out is not None:
            # 既に qx,qy,qz,qw を計算している場合はそれを使う
            # （yaw_out is not None のとき上で qx..qw が定義済み）
            t.transform.rotation.x = float(qx)
            t.transform.rotation.y = float(qy)
            t.transform.rotation.z = float(qz)
            t.transform.rotation.w = float(qw)
        else:
            # yawが無い場合は単位クォータニオン
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0

        self.tf_br.sendTransform(t)

        # 比較用 /odom_raw
        od_raw = Odometry()
        od_raw.header.stamp = od.header.stamp
        od_raw.header.frame_id = 'odom'
        od_raw.child_frame_id = 'base_link'
        od_raw.pose.pose.position.x = float(self.X_raw)
        od_raw.pose.pose.position.y = float(self.Y_raw)
        if yaw_out is not None:
            qx, qy, qz, qw = q_from_yaw(yaw_out)
            od_raw.pose.pose.orientation.x = qx
            od_raw.pose.pose.orientation.y = qy
            od_raw.pose.pose.orientation.z = qz
            od_raw.pose.pose.orientation.w = qw
        od_raw.twist.twist = od.twist.twist
        od_raw.pose.covariance = od.pose.covariance
        od_raw.twist.covariance = od.twist.covariance
        self.pub_odom_raw.publish(od_raw)

        self.rows += 1
        if self.rows % 10 == 0:
            dx_txt = f"{self.dx_m:.3f}" if self.dx_m is not None else "None"
            yaw_s = f"{yaw_straight:.3f}" if yaw_straight is not None else "None"
            yaw_t = f"{yaw_turn:.3f}" if yaw_turn is not None else "None"
            self.get_logger().info(
                f"[s_fusion] mode={'STRAIGHT' if self.vanish_valid else 'TURN'} "
                f"S_enc={self.S_enc:.3f}  X_cam={self.X_cam:.3f}  X_enc={self.X_enc:.3f}  X={self.X:.3f}  Y={self.Y:.3f} "
                f"dx={dx_txt}  yaw_ref={yaw_s}  yaw_imu={yaw_t}"
            )


def main():
    rclpy.init()
    node = ForwardSFusionNode()
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

