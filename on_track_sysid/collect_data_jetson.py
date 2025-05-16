import rclpy
from rclpy.node import Node
import csv
import yaml
import os
import numpy as np
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped

class JetsonDataLogger(Node):
    def __init__(self):
        super().__init__('jetson_data_logger')
        self.racecar_version = "JETSON"
        
        self.load_parameters()
        self.data_collection_duration = self.nn_params['data_collection_duration']
        self.rate = 40
        self.storage_setup()

        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(AckermannDriveStamped, '/drive', self.steering_callback, 10)
        
    def load_parameters(self):
        yaml_file = os.path.join('src/sys_id_py/params/nn_params.yaml')
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
        self.steering = abs(msg.drive.steering_angle)
        self.collect_data()

    def collect_data(self):
            """
            Collects data during simulation.

            Adds the current state to the data array and updates the counter.
            Closes the progress bar and logs a message if data collection is complete.
            """
            self.get_logger().info("Data Collection Started")
            
            '''if self.current_state[0] > 0.0: # Only collect data when the car is moving
                self.data = np.roll(self.data, -1, axis=0)
                self.data[-1] = self.current_state
                self.counter += 1
            if self.counter == self.timesteps + 1:
                self.get_logger().info("Data collection completed.")'''
            
            while self.counter <= self.timesteps:
                if self.current_state[0] > 0.0: # Only collect data when the car is moving
                    self.data = np.roll(self.data, -1, axis=0)
                    self.data[-1] = self.current_state
                    self.counter += 1
                    self.get_logger().info(f"No. of rows recorded: {self.counter}")
            self.get_logger().info("Data collection completed.")
                

    def export_data_as_csv(self):
        ch = input("Save data to csv? (y/n): ")
        if ch == "y":
            data_dir = os.path.join('src/sys_id_py', 'data')
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
    
    def loop(self):
        while rclpy.ok():
            self.collect_data()
            print(self.counter)
            if self.counter == self.timesteps + 1:
                self.export_data_as_csv()
                self.destroy_node()
                rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = JetsonDataLogger()
    node.loop()

if __name__ == '__main__':
    main()
