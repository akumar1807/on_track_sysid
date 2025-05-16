#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
import os
import yaml
import csv
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Imu
from ament_index_python.packages import get_package_share_directory
from datetime import datetime
from tqdm import tqdm
from sys_id_py.train_model import nn_train

class OnTrackSysId(Node):
    def __init__(self):
        super().__init__('sys_id_py')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('racecar_version', 'default_version'),
                ('odom_topic', '/car_state/odom'),
                ('speed_topic', '/autodrive/f1tenth_1/speed'),
                ('steering_angle_topic', '/autodrive/f1tenth_1/steering'),
                ('imu_topic', '/autodrive/f1tenth_1/imu'),
                ('ackermann_cmd_topic', '/vesc/high_level/ackermann_cmd_mux/input/nav_1'),
                ('save_LUT_name', 'default_lut'),
                ('plot_model', False)
            ]
        )

        self.racecar_version = self.get_parameter('racecar_version').get_parameter_value().string_value
        self.rate = 50
        
        try:
            self.package_path = get_package_share_directory('sys_id_py')
        except Exception as e:
            self.get_logger().error(f"Error: Could not find package 'sys_id_py'. {e}")
            return
        
        self.load_parameters()
        self.setup_data_storage()
        self.timer = self.create_timer(1.0 / self.rate, self.collect_data)
        
        self.save_LUT_name = self.get_parameter('save_LUT_name').get_parameter_value().string_value
        self.plot_model = self.get_parameter('plot_model').get_parameter_value().bool_value
        
        #self.odom_sub = self.create_subscription(Odometry, self.get_parameter('odom_topic').get_parameter_value().string_value, self.odom_cb, 10)
        self.speed_sub = self.create_subscription(Twist, self.get_parameter('speed_topic').get_parameter_value().string_value, self.speed_cb, 10)
        self.imu_sub = self.create_subscription(Imu, self.get_parameter('speed_topic').get_parameter_value().string_value, self.imu_cb, 10)
        self.steering_sub = self.create_subscription(Float32, self.get_parameter('steering_angle_topic').get_parameter_value.string_value, self.steering_cb, 10)
        #self.ackermann_sub = self.create_subscription(AckermannDriveStamped, self.get_parameter('ackermann_cmd_topic').get_parameter_value().string_value, self.ackermann_cb, 10)
    
    def setup_data_storage(self):
        self.data_duration = self.nn_params['data_collection_duration']
        self.timesteps = self.data_duration * self.rate
        self.data = np.zeros((self.timesteps, 4))
        self.counter = 0
        self.current_state = np.zeros(4)
    
    def load_parameters(self):
        yaml_file = os.path.join(self.package_path, 'params/nn_params.yaml')
        with open(yaml_file, 'r') as file:
            self.nn_params = yaml.safe_load(file)
    
    def export_data_as_csv(self):
        user_input = input("\033[33m[WARN] Press 'Y' and then ENTER to export data as CSV, or press ENTER to continue without dumping: \033[0m")
        if user_input.lower() == 'y':
            data_dir = os.path.join(self.package_path, 'data', self.racecar_version)
            os.makedirs(data_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file = os.path.join(data_dir, f'{self.racecar_version}_sys_id_data_{timestamp}.csv')
            
            with open(csv_file, mode='w') as file:
                writer = csv.writer(file)
                writer.writerow(['v_x', 'v_y', 'omega', 'delta'])
                writer.writerows(self.data)
            self.get_logger().info(f"DATA HAS BEEN EXPORTED TO: {csv_file}")
    
    def speed_cb(self, msg):
        self.current_state[0] = msg.linear.x
        self.current_state[1] = msg.linear.y

    def imu_cb(self, msg):
        self.current_state[2] = msg.angular_velocity.z
        
    def steering_cb(self, msg):
        self.current_state[3] = msg.data
        
    def collect_data(self):
        if self.current_state[0] > 1:
            self.data = np.roll(self.data, -1, axis=0)
            self.data[-1] = self.current_state
            self.counter += 1
            self.pbar.update(1)
        if self.counter == self.timesteps + 1:
            self.pbar.close()
            self.get_logger().info("Data collection completed.")
            self.run_nn_train()
            self.export_data_as_csv()
            rclpy.shutdown()
            
    def run_nn_train(self):
        self.get_logger().info("Training neural network...")
        nn_train(self.data, self.racecar_version, self.plot_model)
    
    def loop(self):
        self.pbar = tqdm(total=self.timesteps, desc='Collecting data', ascii=True)
        rclpy.spin(self)
        
if __name__ == '__main__':
    rclpy.init()
    sys_id = OnTrackSysId()
    sys_id.loop()
    sys_id.destroy_node()
    rclpy.shutdown()
