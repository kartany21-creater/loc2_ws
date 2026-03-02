from setuptools import find_packages, setup
from glob import glob

package_name = 'loc_bridge'

data_files = [
    ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name + '/config', glob('config/*.yaml')),
    ('share/' + package_name + '/launch', glob('launch/*.py')),
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maffin21',
    maintainer_email='kanata20011004@gmail.com',
    description='Localization bridge nodes and EKF launch',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # IMU
            'yaw_to_imu = loc_bridge.yaw_to_imu_node:main',

            # Depthカメラ (統合版：角度＋ΔZ publish)
            'camimu_node = loc_bridge.camimu_node:main',

            # エンコーダ
            'rpm_to_odom = loc_bridge.rpm_to_odom_node:main',

            # EKF切替ノード
            'ekf_switch_node = loc_bridge.ekf_switch_node:main',

            # CSVロガー
            'odom_csv_logger = loc_bridge.odom_csv_logger:main',

            # ★追加：距離融合＋XY展開
            's_fusion_node = loc_bridge.s_fusion_node:main',
            
            'plan_csv_logger = loc_bridge.plan_csv_logger:main',
        ],
    },
)

