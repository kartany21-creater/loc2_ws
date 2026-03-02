from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # IMU(yaw) → /um7/data へ変換
        Node(
            package='loc_bridge',
            executable='yaw_to_imu',
            name='yaw_to_imu',
            output='screen'
        ),

        # エンコーダ → /wheel/odom_raw（vx含む）
        Node(
            package='loc_bridge',
            executable='rpm_to_odom',
            name='rpm_to_odom',
            output='screen'
        ),

        # カメラ：角度＋ΔZを生成
        Node(
            package='loc_bridge',
            executable='camimu_node',
            name='camimu_node',
            output='screen',
            parameters=[{
                'yolo_model': '/home/maffin21/loc_ws/src/loc_bridge/農場（最終版）/best.pt',
                'yolo_conf': 0.25,
                'yolo_classes': '',
                'timer_dt': 0.10,
                'lpf_alpha': 0.40,
                'vx_deadband': 0.05
            }]
        ),

        # 角度の切替など（既存）
        Node(
            package='loc_bridge',
            executable='ekf_switch_node',
            name='ekf_switch_node',
            output='screen'
        ),

        # robot_localization：YawはIMU＋カメラ角度から
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_node',
            output='screen',
            parameters=['/home/maffin21/loc2_ws/src/loc_bridge/config/ekf_camimu.yaml']
        ),

        # ★ 距離S融合（ΔZ直積算＋エンコーダ積分）→ YawでXY展開
        Node(
            package='loc_bridge',
            executable='s_fusion_node',
            name='s_fusion_node',
            output='screen',
            parameters=[{
               'deadband': 0.03,
                'q_var': 1.0e-4,
                'r_cam': 4.0e-4,
                'r_enc': 1.0e-4
            }]
        ),

        # CSVロガー（既存）
        Node(
            package='loc_bridge',
            executable='odom_csv_logger',
            name='odom_csv_logger',
            output='screen'
        ),

        # ★★★ 新規追加：Nav2の /plan を保存する専用ノード ★★★
        Node(
            package='loc_bridge',
            executable='plan_csv_logger',
            name='plan_csv_logger',
            output='screen'
        ),
    ])

