import os
import ament_index_python
import yaml

def get_nn_params():
    """
    Retrieve neural network parameters.

    Loads neural network parameters from a YAML file.

    Returns:
        dict: Neural network parameters.
    """
    ros2pack = ament_index_python.packages
    package_path = 'src/sys_id_py'  # Replace with your package name
    yaml_file = os.path.join(package_path, 'params/nn_params.yaml')
    with open(yaml_file, 'r') as file:
        nn_params = yaml.safe_load(file)
        
    return nn_params