import numpy as np
import os
import yaml
import rclpy
from rclpy.node import Node
import csv
from math import atan2
from std_msgs.msg import Float32, Float64
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Point

class DataCollector(Node):
 #----INITIALIZATION----#
    def __init__(self):
        super().__init__('data_collector')
        self.declare_parameter('racecar_version', 'JETSON')
        self.declare_parameter('save_csv', True)
        #self.racecar_version = input("Enter the racecar version (All caps): ")
        self.racecar_version = self.get_parameter('racecar_version').get_parameter_value().string_value
        self.load_parameters()
        self.data_collection_duration = self.nn_params['data_collection_duration']
        self.rate = 40
        self.prev_position = np.array([])
        self.x_dot = 0.0
        self.y_dot = 0.0
        self.yaw = 0.0
        self.storage_setup()

        # Shutdown flag
        self.shutdown_triggered = False

        if self.racecar_version == "SIM":
            #self.create_subscription(Float32, '/autodrive/f1tenth_1/speed', self.speed_callback, 10)
            self.create_subscription(Float32, '/autodrive/f1tenth_1/steering', self.steering_callback, 10)
            self.create_subscription(Imu, '/autodrive/f1tenth_1/imu', self.imu_callback, 10)
            self.create_subscription(Point, '/autodrive/f1tenth_1/ips', self.ips_callback, 10)
        else:
            self.create_subscription(Odometry, '/odom', self.odom_cb, 10)
            self.create_subscription(Float64, '/commands/servo/position', self.steering_cb, 10)

        #Shutdown timer
        self.create_timer(0.1, self.check_shutdown)

    def load_parameters(self):
        yaml_file = os.path.join('src/on_track_sysid/params/nn_params.yaml')
        with open(yaml_file, 'r') as file:
            self.nn_params = yaml.safe_load(file)

    def storage_setup(self):
        self.timesteps = self.data_collection_duration*self.rate
        self.dataset = np.zeros((self.timesteps,4))
        self.current_state = np.zeros(4)
        self.counter = 0

#----CALLBACKS FOR SIM----#

    '''def speed_callback(self, msg):
        self.current_state[0] = msg.data 
        self.current_state[1] = 0.001 #Assuming very little sideslip (v_y ~ 0)
        self.collect_data()'''

    def ips_callback(self, msg):
        if self.prev_position.size == 0:
            self.prev_position = np.array([msg.x, msg.y, msg.z])
        else:
            delta_pos = np.array([msg.x, msg.y, msg.z]) - self.prev_position
            dt = 1.0 / self.rate
            self.x_dot = delta_pos[0] / dt
            self.y_dot = delta_pos[1] / dt
            self.prev_position = np.array([msg.x, msg.y, msg.z])

    def steering_callback(self, msg):
        self.current_state[3] = msg.data
        self.collect_data()

    def imu_callback(self, msg):
        self.yaw = self.quaternion_to_yaw(msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w)
        self.current_state[0] = self.x_dot*np.cos(self.yaw) + self.y_dot*np.sin(self.yaw) #Speed in x
        self.current_state[1] = -self.x_dot*np.sin(self.yaw) + self.y_dot*np.cos(self.yaw) #Speed in y
        self.current_state[2] = msg.angular_velocity.z #Yaw rate
        self.collect_data()


#----CALLBACKS FOR JETSON----#

    def odom_cb(self, msg):
        self.current_state[0] = abs(msg.twist.twist.linear.x)
        self.current_state[1] = abs(msg.twist.twist.linear.y)
        self.current_state[2] = abs(msg.twist.twist.angular.z)
        self.collect_data()

    def steering_cb(self, msg):
        servo_val = msg.data
        offset = 0.5304
        gain = -1.2135
        self.current_state[3] = (servo_val-offset)/gain
        self.collect_data()

#----HELPER FUNCTIONS----#
    def quaternion_to_yaw(self, x, y, z, w):
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = atan2(siny_cosp, cosy_cosp)
        return yaw
    
    def check_shutdown(self):
        if self.shutdown_triggered:
            self.get_logger().info("Shutting down node...")
            self.destroy_node()
            rclpy.shutdown()

#----DATA COLLECTION & EXPORT----#
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

    def export_data_as_csv(self):
        save_to_csv = self.get_parameter('save_csv').get_parameter_value().bool_value
        #ch = input("Save data to csv? (y/n): ")
        if save_to_csv:
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
    node = DataCollector()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
