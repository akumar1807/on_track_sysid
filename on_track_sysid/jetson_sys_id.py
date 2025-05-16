import numpy as np
import os
import yaml
import csv
from ament_index_python.packages import get_package_share_directory
from sys_id_py.train_model import nn_train

class JetsonSysID():
    def __init__(self):
        try:
            self.package_path = get_package_share_directory('sys_id_py')
        except Exception as e:
            print(f"Error: Could not find package 'sys_id_py'")
            return
        self.rate = 50
        self.model_version = 'JETSON'
        self.plot_model = True
        self.load_parameters()
        self.setup_data_storage()
        #self.timer = self.create_timer(1.0 / self.rate, self.collect_data)
    
    def setup_data_storage(self):
        '''self.data_duration = self.nn_params['data_collection_duration']
        self.timesteps = self.data_duration * self.rate'''
        self.file = open("src/sys_id_py/jetson_training_data.csv", 'r')
        self.v_x = np.array([])
        self.v_y = np.array([])
        self.steering = np.array([])
        self.omega = np.array([])

        next(self.file) #Skips header row
        for lines in self.file:
            print(lines)
            self.v_x = np.append(self.v_x, lines[0])  
            self.v_y = np.append(self.v_y, lines[1]) 
            self.steering = np.append(self.v_x, lines[2]) 
            self.omega = np.append(self.v_x, lines[3])      
        #print(speed_x.reshape(-1,1))
        self.dataset = np.array([self.v_x, self.v_y, self.steering, self.omega]).T
        print(self.dataset)

    def load_parameters(self):
        yaml_file = os.path.join('src/sys_id_py/params/nn_params.yaml')
        with open(yaml_file, 'r') as file:
            self.nn_params = yaml.safe_load(file)
        
    def run_nn_train(self):
        print("Begin Training")
        nn_train(self.dataset, self.model_version, self.plot_model)
        
def main():
    sysid = JetsonSysID()
    sysid.run_nn_train()

if __name__ == '__main__':
    main()
