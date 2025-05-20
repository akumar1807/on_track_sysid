import yaml
import os
import rclpy
from rclpy.node import Node

def save(model, overwrite_existing=True, verbose=False):
    package_path = 'src/on_track_sysid'
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
