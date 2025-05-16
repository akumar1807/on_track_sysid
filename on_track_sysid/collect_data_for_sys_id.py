import rclpy
from rclpy.node import Node
import csv
from std_msgs.msg import Float32
from sensor_msgs.msg import Imu

class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')
        self.file = open('f1_training_data.csv', 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(['time', 'speed', 'steering', 'x_angle', 'y_angle', 'z_angle', 'imu_yaw_rate'])

        self.speed = None
        self.steering = None
        self.imu_data = None

        self.create_subscription(Float32, '/autodrive/f1tenth_1/speed', self.speed_callback, 10)
        self.create_subscription(Float32, '/autodrive/f1tenth_1/steering', self.steering_callback, 10)
        self.create_subscription(Imu, '/autodrive/f1tenth_1/imu', self.imu_callback, 10)

    def speed_callback(self, msg):
        self.speed = msg.data
        self.log_data()

    def steering_callback(self, msg):
        self.steering = msg.data
        self.log_data()

    def imu_callback(self, msg):
        self.imu_data = msg
        self.log_data()

    def log_data(self):
        timesteps = 0 
        if self.speed is not None and self.steering is not None and self.imu_data is not None and self.speed > 0.0:
            while timesteps < 750:
                timestamp = self.get_clock().now().to_msg().sec + self.get_clock().now().to_msg().nanosec * 1e-9
                self.writer.writerow([
                    timestamp,
                    self.speed,
                    self.steering,
                    self.imu_data.orientation.x,
                    self.imu_data.orientation.y,
                    self.imu_data.orientation.z,
                    self.imu_data.angular_velocity.z
                ])
                timesteps += 1
        print("Data Collected")
        self.file.close()
        self.destroy_node()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = DataLogger()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
