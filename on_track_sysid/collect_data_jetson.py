import rclpy
from rclpy.node import Node
import csv
import yaml
import os
import numpy as np
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry

class JetsonDataLogger(Node):
    def __init__(self):
        super().__init__('jetson_data_logger')
        self.racecar_version = "JETSON"
        
        self.load_parameters()
        self.data_collection_duration = self.nn_params['data_collection_duration']
        self.rate = 50
        self.storage_setup()

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float64, '/commands/servo/position', self.steering_callback, 10)
        
        # Shutdown logic
        self.shutdown_triggered = False
        self.shutdown_timer = self.create_timer(0.1, self.check_shutdown)

    def load_parameters(self):
        yaml_file = os.path.join('src/on_track_sysid/params/nn_params.yaml')
        with open(yaml_file, 'r') as file:
            self.nn_params = yaml.safe_load(file)

    def storage_setup(self):
        self.timesteps = self.data_collection_duration*self.rate
        self.dataset = np.zeros((self.timesteps,4))
        self.current_state = np.zeros(4)
        self.counter = 0

    def odom_callback(self, msg):
        self.current_state[0] = abs(msg.twist.twist.linear.x)
        self.current_state[1] = abs(msg.twist.twist.linear.y)
        self.current_state[2] = abs(msg.twist.twist.angular.z)
        self.collect_data()

    def steering_callback(self, msg):
        servo_val = msg.data
        offset = 0.5304
        gain = -1.2135
        self.current_state[3] = (servo_val-offset)/gain
        self.collect_data()

    def collect_data(self):
        if self.counter <= self.timesteps:
            if self.current_state[0] > 0.0:  # collect only if moving
                self.dataset[self.counter] = self.current_state
                self.counter += 1
                self.get_logger().info(f"No. of rows recorded: {self.counter}")
            if self.counter == self.timesteps:
                self.get_logger().info("Data collection completed.")
                self.export_data_as_csv()
                self.shutdown_triggered = True

    def check_shutdown(self):
        if self.shutdown_triggered:
            self.get_logger().info("Shutting down node...")
            self.destroy_node()
            rclpy.shutdown()

    def export_data_as_csv(self):
        ch = input("Save data to csv? (y/n): ")
        if ch == "y":
            data_dir = os.path.join('src/on_track_sysid', 'data')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
            csv_file = os.path.join(data_dir, f'{self.racecar_version}_sys_id_data.csv')
            with open(csv_file, 'w') as file:
                writer = csv.writer(file)
                writer.writerow(['speed_x', 'speed_y', 'omega', 'steering_angle'])
                for row in self.dataset:
                    writer.writerow(row)
            self.get_logger().info("Exported to CSV successfully")
            file.close()
    
def main(args=None):
    rclpy.init(args=args)
    node = JetsonDataLogger()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
