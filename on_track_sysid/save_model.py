import yaml
import os
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

def save(model, overwrite_existing=True, verbose=False):
    '''try:
        package_path = get_package_share_directory('sys_id_py')
    except Exception as e:
        print(f"Error: Could not find package 'sys_id_py'. {e}")
        return'''
    
    package_path = 'src/sys_id_py'
    
    file_path = os.path.join(package_path, "models", model['model_name'], f"{model['model_name']}_{model['tire_model']}.txt")
    
    if os.path.isfile(file_path):
        if verbose:
            print("Model already exists")
        if overwrite_existing:
            if verbose:
                print("Overwriting...")
        else:
            if verbose:
                print("Not overwriting.")
            return 0
    
    try:
        model = model.to_dict()
    except AttributeError:
        pass  # Model is already a dictionary
    
    # Create necessary directories if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # Initialize ROS2 node
    rclpy.init()
    node = Node('model_saver')
    node.get_logger().info(f"MODEL IS SAVED TO: {file_path}")
    
    # Write data to the file
    with open(file_path, "w") as f:
        yaml.dump(model, f, default_flow_style=False)
    
    # Cleanup
    node.destroy_node()
    rclpy.shutdown()
