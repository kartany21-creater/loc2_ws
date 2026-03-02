#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cam_estimator_node (integrated upgrade):
- RealSense から color/depth を取得
- YOLOで株元候補を抽出

【前進移動量 ΔZ（=ΔS_cam）】は odom_box.py の高精度版を適用:
    1) BOX最下端（y2）の「横1行」の深度中央値で株元接地点Zを決める
    2) 近距離/遠距離を除外
    3) 二重検出排除（XZ距離しきい値）
    4) EKF3DTracker でフレーム間トラッキング（Hungarian割当）で dz を取る
    5) dz 全体平均 → ds_cam = -mean(dz)*0.8
- ΔZ を /camera/ds_raw (Float32, [m]) に publish
- 速度 /camera/twist_raw は互換維持（ds_raw/dt を vx として出す。運用で不要なら購読側で無視）

【角度推定】は angle.py の安定版を適用:
    1) depth_median(5x5) で (Z,X,u) 候補を作る（v=y2）
    2) DEDUP（ZX距離 & u距離）をRANSAC前に適用
    3) ZX平面で RANSAC ×2 で左右畝クラスタ抽出
    4) 各クラスタで u = A*(1/Z) + B をフィット
    5) 消失点 u_vp は切片平均 u_vp = 0.5*(B_L + B_R)（安定版）
    6) tanθ = (u_vp - cx)/fx → LPF → θ
    7) 外れ値除去: |θ| <= ANGLE_ABS_MAX_DEG のときのみ publish

出力:
  /cam/vanish_valid           std_msgs/Bool
  /cam_theta                  std_msgs/Float32   (rad)
  /camera/twist_raw           geometry_msgs/TwistWithCovarianceStamped  (互換維持)
  /camera/ds_raw              std_msgs/Float32   (この周期のΔZ[m])
  /cam_dx                     std_msgs/Float32   (dx = 0.5*(b1+b2) @ ZX直線, 互換維持)
"""

import math
import random
import numpy as np
import cv2
import pyrealsense2 as rs

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32
from geometry_msgs.msg import TwistWithCovarianceStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


# ========= 共有パラメータ（既存ベースを維持） =========
Z_VIEW_MAX = 10.0      # [m]
X_ABS_MAX  = 2.2       # [m]
NEAR_Z_CUT = 0.1       # [m] 角度側の近距離カット（angle.py系）
MIN_POINTS_PER_ROW = 2

# ========= 画像/深度のゲート（既存） =========
COLOR_STD_THR = 10.0
DEPTH_STD_THR = 1.0

# ========= LPF（角度用） =========
LPF_ENABLE = True
LPF_ALPHA  = 0.40

# ========= 速度デッドバンド（互換のため残す） =========
VX_DEADBAND = 0.05

# ========= 角度外れ値除去（angle.py系） =========
ANGLE_ABS_MAX_DEG = 30.0

# ========= dx 外れ値除去 & LPF（dx.py 同等） =========
DX_JUMP_MAX_M = 0.05     # [m] フレーム間で許容する最大dx変化
DX_LPF_ENABLE = True
DX_LPF_ALPHA  = 0.3      # dx.py と同じ値にする

# ========= ZX直線 RANSAC（左右畝抽出：角度側） =========
ZX_RANSAC_ITERS      = 300
ZX_RANSAC_THR_X_M    = 0.10
ZX_RANSAC_MIN_INLIER = 2

# ========= angle.py: DEDUP（二重検出除去） =========
DEDUP_ENABLE   = True
DEDUP_ZX_DIST_M = 0.10   # [m]
DEDUP_U_DIST_PX = 30.0   # [px]
DEDUP_KEEP_MODE = "conf" # "conf" or "near"

# ========= odom_box.py: ΔZ側パラメータ =========
NEAR_Z_REJECT_M   = 0.5
FAR_Z_REJECT_M    = 1.5
DEDUP_XZ_THRESH_M = 0.10   # ΔZ側の二重検出排除（XZ距離）
DZ_SCALE          = 1.0    # ΔZ補正係数（odom_box.py）

# ========= ΔZトラッカID寿命 =========
TRACKER_MAX_AGE = 10   # フレーム数（例：5フレーム = 0.5秒）

# ===== 1D RANSAC（定数モデル）: odom_box.py 完全一致用 =====
RANSAC_THRESH_M = 0.01
RANSAC_ITER = 200
RANSAC_MIN_INLIER = 3

def ransac_1d_constant(values, thresh_m, iters, min_inliers):
    """
    1D値列 values から、定数モデルに一致する inlier をRANSACで抽出して返す。
    戻り値: inlier_values（最低min_inliers未満なら元のvaluesを返す）
    """
    if values is None or len(values) == 0:
        return []

    vals = [float(v) for v in values]
    if len(vals) < 2:
        return vals

    best_inliers = vals
    for _ in range(int(iters)):
        m = random.choice(vals)  # 仮説（定数）
        inl = [v for v in vals if abs(v - m) <= float(thresh_m)]
        if len(inl) >= int(min_inliers) and len(inl) > len(best_inliers):
            best_inliers = inl

    return best_inliers

# ===== ユーティリティ =====
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def depth_median(depth_img: np.ndarray, u: float, v: float, win: int = 5) -> int:
    """angle.py/camimu_node既存系：u,v周辺 win×win の深度中央値"""
    h, w = depth_img.shape[:2]
    uu, vv = int(round(u)), int(round(v))
    if uu < 0 or vv < 0 or uu >= w or vv >= h:
        return 0
    r = win // 2
    u0, u1 = clamp(uu - r, 0, w - 1), clamp(uu + r, 0, w - 1)
    v0, v1 = clamp(vv - r, 0, h - 1), clamp(vv + r, 0, h - 1)
    patch = depth_img[v0:v1+1, u0:u1+1].reshape(-1)
    patch = patch[patch > 0]
    return int(np.median(patch)) if patch.size else 0

def depth_median_bottom_row(depth_img: np.ndarray, x1: int, x2: int, y2: int) -> int:
    """odom_box.py系：BOX最下端 y=y2 の横1行深度の中央値（raw depth）"""
    h, w = depth_img.shape[:2]
    cy = int(y2)
    if cy < 0 or cy >= h:
        return 0
    x1i = int(max(0, min(w-1, x1)))
    x2i = int(max(0, min(w-1, x2)))
    if x2i < x1i:
        x1i, x2i = x2i, x1i

    z_list = []
    for x in range(x1i, x2i + 1):
        d = int(depth_img[cy, x])
        if d > 0:
            z_list.append(d)

    if len(z_list) < 3:
        return 0
    return int(np.median(z_list))

def deproject(u: float, v: float, d_raw: int, intr: dict, depth_scale: float):
    """d_raw を使い (Z,X) を復元。Xは水平。"""
    if d_raw <= 0:
        return None
    Z = d_raw * depth_scale
    X = (u - intr['cx']) / intr['fx'] * Z
    return (Z, X)

# ===== angle.py: DEDUP helper =====
def _zx_u_dist(a, b):
    dz = a["Z"] - b["Z"]
    dx = a["X"] - b["X"]
    d_zx = float(math.sqrt(dz*dz + dx*dx))
    d_u  = float(abs(a["u"] - b["u"]))
    return d_zx, d_u

def dedup_candidates_before_ransac(cands,
                                  zx_thr_m: float = DEDUP_ZX_DIST_M,
                                  u_thr_px: float = DEDUP_U_DIST_PX,
                                  keep_mode: str = DEDUP_KEEP_MODE):
    """angle.py同等：RANSAC前に近接候補を統合して1点だけ残す"""
    if not cands:
        return []

    n = len(cands)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            d_zx, d_u = _zx_u_dist(cands[i], cands[j])
            if d_zx < zx_thr_m and d_u < u_thr_px:
                union(i, j)

    groups = {}
    for i in range(n):
        r = find(i)
        groups.setdefault(r, []).append(i)

    kept = []
    for _, idxs in groups.items():
        if len(idxs) == 1:
            kept.append(cands[idxs[0]])
            continue

        if keep_mode == "near":
            best = idxs[0]
            for k in idxs[1:]:
                if cands[k]["Z"] < cands[best]["Z"] - 1e-9:
                    best = k
                elif abs(cands[k]["Z"] - cands[best]["Z"]) < 1e-9:
                    if cands[k].get("cf", 0.0) > cands[best].get("cf", 0.0):
                        best = k
            kept.append(cands[best])
        else:
            best = idxs[0]
            for k in idxs[1:]:
                if cands[k].get("cf", 0.0) > cands[best].get("cf", 0.0) + 1e-12:
                    best = k
                elif abs(cands[k].get("cf", 0.0) - cands[best].get("cf", 0.0)) < 1e-12:
                    if cands[k]["Z"] < cands[best]["Z"]:
                        best = k
            kept.append(cands[best])

    return kept

# ===== ZX直線 RANSAC（角度側） =====
def ransac_line_zx(Z: np.ndarray, X: np.ndarray,
                   iters: int = ZX_RANSAC_ITERS,
                   thr_x_m: float = ZX_RANSAC_THR_X_M,
                   min_inliers: int = ZX_RANSAC_MIN_INLIER):
    Z = np.asarray(Z, float)
    X = np.asarray(X, float)
    if Z.size < 2:
        return None
    rng = np.random.default_rng()
    best_inliers = None
    best_model = None

    for _ in range(iters):
        i, j = rng.choice(Z.size, 2, replace=False)
        if abs(Z[j] - Z[i]) < 1e-9:
            continue
        a = (X[j] - X[i]) / (Z[j] - Z[i])
        b = X[i] - a * Z[i]
        err = np.abs(X - (a * Z + b))
        inl = np.where(err < thr_x_m)[0]
        if best_inliers is None or inl.size > best_inliers.size:
            best_inliers, best_model = inl, (a, b)

    if best_inliers is None or best_inliers.size < min_inliers:
        return None

    A = np.vstack([Z[best_inliers], np.ones_like(Z[best_inliers])]).T
    a, b = np.linalg.lstsq(A, X[best_inliers], rcond=None)[0]
    return (float(a), float(b), best_inliers)

def ransac_two_lines_cluster_ZX(Z: np.ndarray, X: np.ndarray):
    got1 = ransac_line_zx(Z, X)
    if got1 is None:
        return None
    a1, b1, inl1 = got1
    mask = np.ones(Z.size, dtype=bool)
    mask[inl1] = False
    if mask.sum() < 2:
        return (got1, None)

    got2 = ransac_line_zx(Z[mask], X[mask])
    if got2 is None:
        return (got1, None)
    inl2 = np.where(mask)[0][got2[2]]
    return (got1, (got2[0], got2[1], inl2))


# ===== u = A*(1/Z) + B フィット（angle.py互換：最小二乗で十分） =====
def fit_u_vs_invZ(u_list, Z_list):
    S = []
    U = []
    for u, Z in zip(u_list, Z_list):
        if Z > 0:
            S.append(1.0 / Z)
            U.append(u)
    if len(S) < 2:
        return None
    S = np.array(S, float)
    U = np.array(U, float)
    M = np.vstack([S, np.ones_like(S)]).T
    A, B = np.linalg.lstsq(M, U, rcond=None)[0]
    return float(A), float(B)


# ===== odom.py と同じ 3Dトラッカ（既存） =====
class EKF3DTracker:
    def __init__(self, initial_pos):
        self.dt = 0.1
        self.state = np.hstack([initial_pos, np.zeros(3)])
        self.P = np.eye(6) * 0.01
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i + 3] = self.dt
        self.Q = np.eye(6) * 0.001 * (self.dt ** 2)
        self.H = np.eye(3, 6)
        self.R = np.eye(3) * 0.01
        self.prev_z = initial_pos[2]

    def predict(self):
        self.state = self.F @ self.state
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.P = (np.eye(6) - K @ self.H) @ self.P

    def get_state(self):
        return self.state[:3]

    def get_z_movement(self, z_obs):
        dz = z_obs - self.prev_z
        self.prev_z = z_obs
        return dz


# ===== YOLO =====
class YoloStemDetector:
    def __init__(self, model_path: str = None, conf: float = 0.25, class_ids=None):
        self.ok = False
        self.model = None
        self.conf = conf
        self.class_ids = set(class_ids) if class_ids else None
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path) if model_path else YOLO()
            try:
                self.model.overrides['verbose'] = False
            except Exception:
                pass
            self.ok = True
        except Exception as e:
            print(f"[YOLO] disabled: {e}")

    def detect(self, color_bgr: np.ndarray):
        if not self.ok:
            return []
        res = self.model.predict(color_bgr, imgsz=640, conf=self.conf, verbose=False)[0]
        out = []
        if res.boxes is None or res.boxes.shape[0] == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        clses = res.boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), cf, cl in zip(xyxy, confs, clses):
            if self.class_ids is not None and cl not in self.class_ids:
                continue
            u = 0.5 * (x1 + x2)
            v = y2
            out.append((float(u), float(v), (int(x1), int(y1), int(x2), int(y2), float(cf), int(cl))))
        return out


# ===== ノード本体 =====
class CamEstimatorNode(Node):
    """
    θ と ΔZ/vx を同一点群から算出し、publish は独立：
    - ΔZ は odom_box.py の高精度版で毎周期 publish (/camera/ds_raw)
    - 速度 /camera/twist_raw は互換維持
    - 角度は angle.py の高精度版で vanish_valid True のときのみ publish
    """
    def __init__(self):
        super().__init__('cam_estimator_node')

        # パラメータ（既存そのまま）
        self.declare_parameter('yolo_model', None)
        self.declare_parameter('yolo_conf', 0.25)
        self.declare_parameter('yolo_classes', '')     # "0,1,2" 等
        self.declare_parameter('timer_dt', 0.1)        # [s]
        self.declare_parameter('lpf_alpha', LPF_ALPHA)
        self.declare_parameter('vx_deadband', VX_DEADBAND)

        mdl = self.get_parameter('yolo_model').get_parameter_value().string_value or None
        yconf = float(self.get_parameter('yolo_conf').get_parameter_value().double_value or 0.25)
        cls_txt = self.get_parameter('yolo_classes').get_parameter_value().string_value
        self.dt = float(self.get_parameter('timer_dt').get_parameter_value().double_value or 0.1)
        self.lpf_alpha = float(self.get_parameter('lpf_alpha').get_parameter_value().double_value or LPF_ALPHA)
        self.vx_deadband = float(self.get_parameter('vx_deadband').get_parameter_value().double_value or VX_DEADBAND)

        cls_ids = [int(s) for s in cls_txt.split(",") if s.strip().isdigit()] if cls_txt else None

        # RealSense
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        prof = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color)
        intr_c = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        depth_scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.intr = {'fx': intr_c.fx, 'fy': intr_c.fy, 'cx': intr_c.ppx, 'cy': intr_c.ppy}
        self.depth_scale = depth_scale

        # YOLO
        self.detector = YoloStemDetector(mdl, yconf, cls_ids)

        # 出力
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.pub_theta  = self.create_publisher(Float32, '/cam_theta', 10)
        self.pub_twist  = self.create_publisher(TwistWithCovarianceStamped, '/camera/twist_raw', qos)  # 互換
        self.pub_vanish = self.create_publisher(Bool, '/cam/vanish_valid', 10)
        self.pub_ds     = self.create_publisher(Float32, '/camera/ds_raw', 10)  # ΔZ [m]
        self.pub_dx     = self.create_publisher(Float32, '/cam_dx', 10)         # dx（互換）

        # 状態（角度用）
        self.k_tan_lpf = None
        
        #dx処理
        self.dx_prev = None
        self.dx_lpf = None

        # 状態（ΔZ/速度用）
        self.trackers = {}
        self.next_id = 0

        self.get_logger().info("✅ cam_estimator_node started (angle.py + odom_box.py integrated upgrade)")

        self.timer = self.create_timer(self.dt, self.on_timer)

    def on_timer(self):
        # フレーム取得（color基準にアライン）
        frames = self.pipe.wait_for_frames()
        frames = self.align.process(frames)
        dfrm = frames.get_depth_frame()
        cfrm = frames.get_color_frame()
        if not dfrm or not cfrm:
            self._publish_fail()
            return

        depth = np.asanyarray(dfrm.get_data())
        color = np.asanyarray(cfrm.get_data())

        # 真っ暗/ふさがれ
        if color.std() < COLOR_STD_THR or depth.std() < DEPTH_STD_THR:
            self._publish_fail()
            return

        # YOLO 検出
        dets = self.detector.detect(color)  # [(u,v,(x1,y1,x2,y2,conf,cls)), ...]
        if len(dets) == 0:
            self._publish_fail()
            return

        # ==========================================================
        #  A) ΔZ（odom_box.py方式）：BOX最下端1行の深度中央値
        # ==========================================================
        raw_detections_dz = []  # (u, (X,0,Z), bbox)
        for (u, v, meta) in dets:
            x1, y1, x2, y2, cf, cl = meta

            d_raw = depth_median_bottom_row(depth, x1, x2, y2)
            if d_raw <= 0:
                continue

            Z = d_raw * self.depth_scale
            if Z < NEAR_Z_REJECT_M or Z > FAR_Z_REJECT_M:
                continue

            cx = int(round(u))
            X = (cx - self.intr['cx']) / self.intr['fx'] * Z
            raw_detections_dz.append((float(u), (float(X), 0.0, float(Z)), meta))

        # ΔZ側：二重検出排除（XZ距離）
        detections_dz = []
        for cand in sorted(raw_detections_dz, key=lambda t: float(t[2][4]) if len(t[2]) >= 5 else 0.0, reverse=True):
            u_c, xyz_c, meta_c = cand
            too_close = False
            for kept in detections_dz:
                kx, _, kz = kept[1]
                dx = xyz_c[0] - kx
                dz = xyz_c[2] - kz
                if math.sqrt(dx*dx + dz*dz) < DEDUP_XZ_THRESH_M:
                    too_close = True
                    break
            if not too_close:
                detections_dz.append(cand)

        # ΔZ側：EKFトラッキング → dz収集
        predicted = []
        ids = list(self.trackers.keys())
        for tid in ids:
            self.trackers[tid][0].predict()
            predicted.append(self.trackers[tid][0].get_state())

        matched_det_idx = set()
        matched_tracker_ids = set()
        z_movements = []  # (u, dz, xyz, tid, bbox_meta)

        if predicted and detections_dz:
            det_xyz = [d[1] for d in detections_dz]
            dist = cdist(np.array(predicted), np.array(det_xyz))
            r_idx, c_idx = linear_sum_assignment(dist)
            for ri, ci in zip(r_idx, c_idx):
                if dist[ri][ci] < 0.3:
                    tid = ids[ri]
                    xyz = det_xyz[ci]
                    self.trackers[tid][0].update(xyz)
                    self.trackers[tid][1] = 0
                    matched_det_idx.add(ci)
                    matched_tracker_ids.add(tid)

                    dz = self.trackers[tid][0].get_z_movement(xyz[2])
                    z_movements.append((detections_dz[ci][0], dz, xyz, tid, detections_dz[ci][2]))

        for i, det in enumerate(detections_dz):
            if i not in matched_det_idx:
                u_i, xyz_i, meta_i = det
                self.trackers[self.next_id] = [EKF3DTracker(xyz_i), 0]
                dz0 = self.trackers[self.next_id][0].get_z_movement(xyz_i[2])  # 初回0
                z_movements.append((u_i, dz0, xyz_i, self.next_id, meta_i))
                self.next_id += 1
              
        # ==========================================================
        # ② ID寿命管理（未マッチ時のみ age++）※1フレーム1回
        # ==========================================================
        dead_ids = []
        for tid in list(self.trackers.keys()):
            if tid not in matched_tracker_ids:
                self.trackers[tid][1] += 1
            else:
                self.trackers[tid][1] = 0

            if self.trackers[tid][1] > TRACKER_MAX_AGE:
                dead_ids.append(tid)

        for tid in dead_ids:
            del self.trackers[tid]

 
        ds_cam = 0.0
        if len(z_movements) >= 1:
            dz_all = [dz for (_, dz, _, _, _) in z_movements]

            dz_inl = ransac_1d_constant(
                dz_all,
                RANSAC_THRESH_M,
                RANSAC_ITER,
                RANSAC_MIN_INLIER
            )

            ds_cam = -float(np.mean(dz_inl)) * float(DZ_SCALE)

        self.pub_ds.publish(Float32(data=float(ds_cam)))


        # 速度（互換維持）
        vx_mps = float(ds_cam) / float(self.dt) if self.dt > 1e-9 else 0.0
        if abs(vx_mps) < self.vx_deadband:
            vx_mps = 0.0

        tw = TwistWithCovarianceStamped()
        tw.header.stamp = self.get_clock().now().to_msg()
        tw.header.frame_id = 'base_link'
        tw.twist.twist.linear.x = float(vx_mps)
        tw.twist.twist.angular.z = 0.0
        cov = [0.0] * 36
        cov[0] = 0.02 if vx_mps != 0.0 else 0.50
        cov[35] = 1e3
        tw.twist.covariance = cov
        self.pub_twist.publish(tw)

        # ==========================================================
        #  B) 角度（angle.py方式）：DEDUP → ZX-RANSAC×2 → u-1/Z
        # ==========================================================
        cands = []  # [{'Z':..,'X':..,'u':..,'cf':..}, ...]
        for (u, v, meta) in dets:
            x1, y1, x2, y2, cf, cl = meta

            # angle.py系は depth_median(5x5) を使用（v=y2）
            d_raw = depth_median(depth, u, float(y2), win=5)
            p = deproject(u, float(y2), d_raw, self.intr, self.depth_scale)
            if p is None:
                continue
            Z, X = p
            if Z <= 0 or Z < NEAR_Z_CUT or Z > Z_VIEW_MAX:
                continue
            if abs(X) > X_ABS_MAX:
                continue

            cands.append({"Z": float(Z), "X": float(X), "u": float(u), "cf": float(cf)})

        if DEDUP_ENABLE:
            cands = dedup_candidates_before_ransac(
                cands,
                zx_thr_m=DEDUP_ZX_DIST_M,
                u_thr_px=DEDUP_U_DIST_PX,
                keep_mode=DEDUP_KEEP_MODE
            )

        if len(cands) < (MIN_POINTS_PER_ROW * 2):
            self.pub_vanish.publish(Bool(data=False))
            return

        Zs = np.array([c["Z"] for c in cands], float)
        Xs = np.array([c["X"] for c in cands], float)
        Us = np.array([c["u"] for c in cands], float)

        got = ransac_two_lines_cluster_ZX(Zs, Xs)
        if got is None or got[1] is None:
            self.pub_vanish.publish(Bool(data=False))
            return

        (a1, b1, inl1), (a2, b2, inl2) = got

        # dx（横ずれ）互換：ZX直線の切片平均（Z=0のX）
        try:
            # ===== dx 計算（既存） =====
            dx_raw = 0.5 * (b1 + b2)

            # ===== dx 外れ値除去（dx.py 同等：ジャンプ抑制） =====
            if self.dx_prev is None:
                dx_f = dx_raw
            else:
                if abs(dx_raw - self.dx_prev) > DX_JUMP_MAX_M:
                    dx_f = self.dx_prev   # ジャンプ → 保持
                else:
                    dx_f = dx_raw

            self.dx_prev = dx_f

            # ===== dx LPF（dx.py 同等） =====
            if DX_LPF_ENABLE:
                if self.dx_lpf is None:
                    self.dx_lpf = dx_f
                else:
                    self.dx_lpf = (1.0 - DX_LPF_ALPHA) * self.dx_lpf + DX_LPF_ALPHA * dx_f
                dx_out = self.dx_lpf
            else:
                dx_out = dx_f

            # ===== publish（既存と同じ） =====
            self.pub_dx.publish(Float32(data=float(dx_out)))

        except Exception as e:
            self.get_logger().warn(f"dx計算エラー: {e}")

        # 左右割当：X平均が小さい方を左
        mu1 = float(np.mean(Xs[inl1]))
        mu2 = float(np.mean(Xs[inl2]))
        idxL, idxR = (inl1, inl2) if mu1 < mu2 else (inl2, inl1)

        # u = A*(1/Z) + B フィット（各クラスタ）
        uL = Us[idxL].tolist()
        zL = Zs[idxL].tolist()
        uR = Us[idxR].tolist()
        zR = Zs[idxR].tolist()

        # 最低点数（angle.pyの思想）
        if len(uL) < 2 or len(uR) < 2:
            self.pub_vanish.publish(Bool(data=False))
            return

        modelL = fit_u_vs_invZ(uL, zL)
        modelR = fit_u_vs_invZ(uR, zR)
        if modelL is None or modelR is None:
            self.pub_vanish.publish(Bool(data=False))
            return

        A_L, B_L = modelL
        A_R, B_R = modelR

        # angle.pyの安定版：消失点 u_vp は切片平均
        u_vp = 0.5 * (B_L + B_R)

        # tanθ
        k_tan_raw = (u_vp - self.intr['cx']) / self.intr['fx']

        # LPF（tanθ）
        if LPF_ENABLE:
            if self.k_tan_lpf is None:
                self.k_tan_lpf = float(k_tan_raw)
            else:
                self.k_tan_lpf = (1.0 - float(self.lpf_alpha)) * float(self.k_tan_lpf) + float(self.lpf_alpha) * float(k_tan_raw)
            k_tan = float(self.k_tan_lpf)
        else:
            k_tan = float(k_tan_raw)

        theta = math.atan(k_tan)  # rad
        theta_deg = abs(math.degrees(theta))

        # 外れ値除去（angle.py思想）
        if theta_deg <= float(ANGLE_ABS_MAX_DEG):
            self.pub_vanish.publish(Bool(data=True))
            # 既存互換：符号は -theta
            self.pub_theta.publish(Float32(data=float(-theta)))
        else:
            self.pub_vanish.publish(Bool(data=False))

    def _publish_fail(self):
        # 失敗フレーム時：速度0（低信頼）、ΔZ=0、vanish False
        self.pub_vanish.publish(Bool(data=False))
        self.pub_ds.publish(Float32(data=0.0))

        tw = TwistWithCovarianceStamped()
        tw.header.stamp = self.get_clock().now().to_msg()
        tw.header.frame_id = 'base_link'
        tw.twist.twist.linear.x = 0.0
        tw.twist.twist.angular.z = 0.0
        cov = [0.0] * 36
        cov[0] = 0.50
        cov[35] = 1e3
        tw.twist.covariance = cov
        self.pub_twist.publish(tw)

    def destroy_node(self):
        try:
            self.pipe.stop()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = CamEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()

