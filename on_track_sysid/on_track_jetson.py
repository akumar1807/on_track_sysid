import rclpy
from rclpy.node import Node
import csv
import yaml
import os
import numpy as np
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from on_track_sysid.train_model import nn_train

class SysIDForJetson(Node):
    def __init__(self):
        super().__init__('ontrack')
        self.racecar_version = 'JETSON'
        self.plot_model = True

        self.load_parameters()
        self.data_collection_duration = self.nn_params['data_collection_duration']
        self.rate = 50  # Hz
        self.storage_setup()

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float64, '/commands/servo/position', self.steering_callback, 10)

        # Shutdown flag
        self.shutdown_triggered = False

        # Main data collection timer
        self.create_timer(1.0 / self.rate, self.collect_data)
        # Shutdown check
        self.create_timer(0.1, self.check_shutdown)

    def load_parameters(self):
        yaml_file = os.path.join('src/on_track_sysid/params/nn_params.yaml')
        with open(yaml_file, 'r') as file:
            self.nn_params = yaml.safe_load(file)

    def storage_setup(self):
        self.timesteps = self.data_collection_duration * self.rate
        self.dataset = np.zeros((self.timesteps, 4))  # [vx, vy, yaw_rate, steering]
        self.current_state = np.zeros(4)
        self.counter = 0

    def odom_callback(self, msg):
        self.current_state[0] = abs(msg.twist.twist.linear.x)
        self.current_state[1] = abs(msg.twist.twist.linear.y)
        self.current_state[2] = abs(msg.twist.twist.angular.z)

    def steering_callback(self, msg):
        servo_val = msg.data
        offset = 0.5304
        gain = -1.2135
        self.current_state[3] = (servo_val-offset)/gain

    def collect_data(self):
        if self.counter < self.timesteps:
            if self.current_state[0] > 0.0:  # Only collect data when car is moving
                self.dataset[self.counter] = self.current_state
                self.counter += 1
                self.get_logger().info(f"No. of rows recorded: {self.counter}")
        elif not self.shutdown_triggered:
            self.get_logger().info("Data collection completed.")
            self.get_logger().info("Starting training...")
            nn_train(self.dataset, self.racecar_version, self.plot_model)
            self.export_data_as_csv()
            self.shutdown_triggered = True

    def check_shutdown(self):
        if self.shutdown_triggered:
            self.get_logger().info("Shutting down node...")
            self.destroy_node()
            rclpy.shutdown()

    def export_data_as_csv(self):
        ch = input("Save data to CSV? (y/n): ")
        if ch.strip().lower() == 'y':
            data_dir = os.path.join('src/on_track_sysid', 'data')
            os.makedirs(data_dir, exist_ok=True)
            csv_file = os.path.join(data_dir, f'{self.racecar_version}_sys_id_data.csv')
            with open(csv_file, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['speed_x', 'speed_y', 'omega', 'steering_angle'])
                for row in self.dataset:
                    writer.writerow(row)
            self.get_logger().info("Exported to CSV successfully.")

def main(args=None):
    rclpy.init(args=args)
    node = SysIDForJetson()
    rclpy.spin(node)
