#Taken from ETH repo and modified
import ament_index_python
import yaml
import os

"""
    Retrieve model parameters for a given racecar version.
    Loads Pacejka tire and vehicle parameters from YAML files and constructs a model dictionary.
    
    Returns:
        dict: Model parameters including tire and vehicle properties.
    """
def get_model_param(racecar_version):
    #rospack = rospkg.RosPack()
    ros2pack = ament_index_python.packages
    package_path = 'src/sys_id_py' # Replace with your package name
    yaml_file = os.path.join(package_path, 'params/pacejka_params.yaml')
    with open(yaml_file, 'r') as file:
        pacejka_params = yaml.safe_load(file)
        
    # Load vehicle parameters
    #yaml_file = os.path.join(package_path, 'models', racecar_version, racecar_version + '_pacejka.txt')
    yaml_file = os.path.join(package_path, f'models/{racecar_version}.txt')
    with open(yaml_file, 'r') as file:
        vehicle_params = yaml.safe_load(file)

    # Construct model dictionary
    model = {
    "C_Pf_model": pacejka_params['pacejka_model']['C_Pf_model'],
    "C_Pr_model": pacejka_params['pacejka_model']['C_Pr_model'],
    "C_Pf_ref": pacejka_params['pacejka_ref']['C_Pf_ref'],
    "C_Pr_ref": pacejka_params['pacejka_ref']['C_Pr_ref'],
    "m": vehicle_params['m'],
    "I_z": vehicle_params['I_z'],
    "l_f": vehicle_params['l_f'],
    "l_r": vehicle_params['l_r'],
    "l_wb": vehicle_params['l_wb'],
    "racecar_version": racecar_version
    }
    return model