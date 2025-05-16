import numpy as np
import os
import yaml
import csv
from ament_index_python.packages import get_package_share_directory
#from collect_data_for_sys_id import DataLogger
from sys_id_py.train_model import nn_train

class RegularSysID():
    def __init__(self):
        try:
            self.package_path = get_package_share_directory('sys_id_py')
        except Exception as e:
            print(f"Error: Could not find package 'sys_id_py'")
            return
        self.rate = 50
        self.model_version = 'SIM'
        self.plot_model = True
        self.load_parameters()
        self.setup_data_storage()
        #self.timer = self.create_timer(1.0 / self.rate, self.collect_data)
    
    def setup_data_storage(self):
        '''self.data_duration = self.nn_params['data_collection_duration']
        self.timesteps = self.data_duration * self.rate'''
        self.file = open("src/sys_id_py/f1_training_data.csv", 'r')
        speed_x = np.array([])
        speed_y = np.array([])
        steering_angle = np.array([])
        omega = np.array([])
        count = 0
        next(self.file) #Skips header row
        for lines in self.file:
            speed_x = np.append(speed_x, (float(lines[1])*np.cos(float(lines[3]))))
            speed_y = np.append(speed_y, (float(lines[1])*np.cos(float(lines[4]))))
            steering_angle = np.append(steering_angle,float(lines[2]))
            omega = np.append(omega, float(lines[6]))        
        #print(speed_x.reshape(-1,1))
        self.dataset = np.array([speed_x, speed_y, omega, steering_angle]).T
        print(self.dataset.shape)

    def load_parameters(self):
        yaml_file = os.path.join('src/sys_id_py/params/nn_params.yaml')
        with open(yaml_file, 'r') as file:
            self.nn_params = yaml.safe_load(file)
        
    def run_nn_train(self):
        print("Begin Training")
        nn_train(self.dataset, self.model_version, self.plot_model)
        
def main():
    sysid = RegularSysID()
    sysid.run_nn_train()

if __name__ == '__main__':
    main()
