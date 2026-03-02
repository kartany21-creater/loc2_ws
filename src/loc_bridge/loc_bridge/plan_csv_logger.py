#!/usr/bin/env python3
import os
import csv
from datetime import datetime

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy


class PlanCsvLogger(Node):
    def __init__(self):
        super().__init__('plan_csv_logger')

        # Nav2 の /plan は RELIABLE で購読する必要がある
        plan_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(Path, '/plan', self.cb_plan, plan_qos)

        # CSV 保存用ディレクトリ
        ws_root = os.path.expanduser('~/loc2_ws')
        save_dir = os.path.join(ws_root, 'plan_logs')
        os.makedirs(save_dir, exist_ok=True)

        # ファイル名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(save_dir, f'plan_log_{timestamp}.csv')

        # CSV ファイルオープン
        self._fh = open(self.csv_path, mode='w', newline='')
        self._writer = csv.writer(self._fh)

        # ヘッダー
        self._writer.writerow(['time', 'seq', 'plan_x', 'plan_y'])

        self.get_logger().info(f'[plan_csv_logger] Start logging → {self.csv_path}')

        # シーケンス番号（1つの plan 内での点番号）
        self.seq = 0

    # -----------------------------------------------
    # /plan受信（Path） → CSVに保存
    # -----------------------------------------------
    def cb_plan(self, msg: Path):
        t = self.get_clock().now().nanoseconds * 1e-9

        self.get_logger().info(f'[plan] received {len(msg.poses)} points')

        self.seq = 0
        try:
            for pose in msg.poses:
                px = pose.pose.position.x
                py = pose.pose.position.y

                self._writer.writerow([
                    f'{t:.6f}',
                    self.seq,
                    f'{px:.6f}',
                    f'{py:.6f}',
                ])
                self.seq += 1

            # まとめて flush（1回だけ）
            self._fh.flush()
            os.fsync(self._fh.fileno())

        except Exception as e:
            self.get_logger().error(f'[plan] CSV error: {e}')

    # -----------------------------------------------
    # ノード終了処理
    # -----------------------------------------------
    def destroy_node(self):
        try:
            if hasattr(self, '_fh') and self._fh:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
        except Exception as e:
            self.get_logger().warn(f'close error: {e}')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = PlanCsvLogger()

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

