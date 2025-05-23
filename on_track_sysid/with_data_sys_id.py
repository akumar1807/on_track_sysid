import numpy as np
from on_track_sysid.train_model import nn_train

class RegularSysID():
    def __init__(self):
        self.rate = 50
        self.racecar_version = input("Enter the racecar version (All Caps): ")
        self.plot_model = True
        self.setup_dataset()
    
    def setup_dataset(self):
        self.dataset = np.genfromtxt(f"src/on_track_sysid/data/{self.racecar_version}_sys_id_data.csv", 
                                     delimiter=',', 
                                     skip_header=1)
        print(self.dataset)

    def run_nn_train(self):
        print("Begin Training")
        nn_train(self.dataset, self.racecar_version, self.plot_model)
        
def main():
    sysid = RegularSysID()
    sysid.run_nn_train()

if __name__ == '__main__':
    main()
