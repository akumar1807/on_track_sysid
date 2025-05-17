import numpy as np
import os
import yaml
import csv
from on_track_sysid.train_model import nn_train

class JetsonSysID():
    def __init__(self):
        '''try:
            self.package_path = get_package_share_directory('sys_id_py')
        except Exception as e:
            print(f"Error: Could not find package 'sys_id_py'")
            return'''
        self.rate = 50
        self.racecar_version = 'JETSON'
        self.plot_model = True
        self.load_parameters()
        self.setup_data_storage()
        #self.timer = self.create_timer(1.0 / self.rate, self.collect_data)
    
    def setup_data_storage(self):
        '''self.data_duration = self.nn_params['data_collection_duration']
        self.timesteps = self.data_duration * self.rate'''
        self.file = open(f"src/on_track_sysid/{self.racecar_version}_sys_id_data.csv", 'r')
        self.v_x = np.array([])
        self.v_y = np.array([])
        self.steering = np.array([])
        self.omega = np.array([])

        next(self.file) #Skips header row
        self.dataset = np.genfromtxt(f"src/on_track_sysid/{self.racecar_version}_sys_id_data.csv", delimiter=',')
        #print(speed_x.reshape(-1,1))
        '''self.dataset = np.array([self.v_x, self.v_y, self.steering, self.omega]).T
        print(self.dataset)'''

    def load_parameters(self):
        yaml_file = os.path.join('src/on_track_sysid/params/nn_params.yaml')
        with open(yaml_file, 'r') as file:
            self.nn_params = yaml.safe_load(file)
        
    def run_nn_train(self):
        print("Begin Training")
        nn_train(self.dataset, self.racecar_version, self.plot_model)
        
def main():
    sysid = JetsonSysID()
    sysid.run_nn_train()

if __name__ == '__main__':
    main()
