#Taken from ETH Repo and modified
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sys_id_py.filter_data import process_data
from sys_id_py.generate_predictions import generate_predictions
from sys_id_py.generate_inputs_errors import generate_inputs_errors
from sys_id_py.model_params import get_model_param
from sys_id_py.nn_params import get_nn_params
from sys_id_py.NN import NeuralNetwork
from sys_id_py.pacejka_formula import pacejka_formula
from sys_id_py.solve_pacejka import solve_pacejka
from sys_id_py.save_model import save
from sys_id_py.load_model import get_dotdict
from sys_id_py.plot_results import plot_results
from sys_id_py.gen_LUT import LookupGenerator
import rclpy
#import rospkg
from tqdm import tqdm

def simulated_data_gen(nn_model, model, avg_vel):
    C_Pf_model = model['C_Pf_model']
    C_Pr_model = model['C_Pr_model']
    g = 9.81
    l_f = model['l_f']
    l_r = model['l_r']
    l_wb = model['l_wb']
    m = model['m']
    I_z = model['I_z']
    F_zf = m * g * l_r / l_wb
    F_zr = m * g * l_f / l_wb
    dt = 0.02 # 0.02 for 50 Hz

    timesteps = 500 # Number of timesteps to simulate
    
    v_y = np.zeros(timesteps)  # Initial lateral velocity
    omega = np.zeros(timesteps)  # Initial yaw rate
    alpha_f = np.zeros(timesteps)  # Initial lateral velocity
    alpha_r = np.zeros(timesteps)  # Initial yaw rate
    
    v_x = np.ones(timesteps)*avg_vel  # Constant longitudinal velocity
    delta = np.linspace(0.0, 0.4, timesteps)
    
    # Simulation loop
    for t in range(timesteps-1):
        alpha_f[t] = -np.arctan((v_y[t] + omega[t] * l_f) / v_x[t]) + delta[t]
        alpha_r[t] = -np.arctan((v_y[t] - omega[t] * l_r) / v_x[t])

        # Calculate Pacejka lateral forces
        F_f = pacejka_formula(C_Pf_model, alpha_f[t], F_zf)
        F_r = pacejka_formula(C_Pr_model, alpha_r[t], F_zr)
        input = torch.tensor([v_x[t], v_y[t], omega[t], delta[t]], dtype=torch.float32)

        # Making predictions
        with torch.no_grad():
            predicted_means = nn_model(input)
        # Update vehicle states using the dynamics equations
        v_y_dot = (1/m) * (F_r + F_f * np.cos(delta[t]) - m * v_x[t]* omega[t])
        omega_dot = (1/I_z) * (F_f * l_f * np.cos(delta[t]) - F_r * l_r)

        # Euler integration for the next state
        v_y[t+1] = v_y[t] + v_y_dot * dt + predicted_means[0]
        omega[t+1] = omega[t] + omega_dot * dt + predicted_means[1]
    
    return v_x, v_y, omega, delta

def generate_training_set(training_data, model):
    """
    Generate training set for neural network training.

    Predicts the next step's lateral velocity and yaw rate using the vehicle model.
    Constructs input tensors and error tensors for training the neural network.

    Args:
        training_data (numpy.ndarray): Input training data with shape (n_samples, n_features).
        model (dict): Dictionary containing vehicle model parameters.

    Returns:
        tuple: Tuple containing input tensor and target error tensor for training.
    """
    
    # Generate predictions for the next step's lateral velocity and yaw rate
    v_y_next_pred, omega_next_pred = generate_predictions(training_data, model)
    
    # Generate input tensors and error tensors for training
    X_train, y_train = generate_inputs_errors(v_y_next_pred, omega_next_pred, training_data)
    
    return X_train, y_train

def nn_train(training_data, racecar_version, plot_model):
    """    
    Trains the neural network using the provided training data and model parameters.
    
    After training, it simulates the car behavior with the trained model and identifies
    Pacejka tire model coefficients. 
    
    Then it iteratively refines the model and repeats
    the training process. 
    
    Finally, it saves the trained model and generates a
    Look-Up Table (LUT) for the controller. 

    """
    # Get model and neural network parameters
    model = get_model_param(racecar_version)
    nn_params = get_nn_params()
    num_of_epochs = nn_params['num_of_epochs']
    lr = nn_params['lr']
    weight_decay = nn_params['weight_decay']
    num_of_iterations = nn_params['num_of_iterations']

    training_data = process_data(training_data, model)   
     
    avg_vel = np.mean(training_data[:,0]) # Defining average velocity for the simulation, NN will have more accurate predictions
    avg_vel = np.clip(avg_vel, 2.75, 4)
    
    # Iterative training loop
    for i in range(1, num_of_iterations+1):
        if i == num_of_iterations: # Determine if it's the last iteration to enable plotting (if plot_model is False)
            plot_model = True
            
        # Process training data and generate inputs and targets
        X_train, y_train = generate_training_set(training_data, model)
        
        # Initialize the network
        nn_model = NeuralNetwork()

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = Adam(nn_model.parameters(), lr=lr)

        nn_model.train()
        pbar = tqdm(total=num_of_epochs, desc=f"Iteration: {i}/{num_of_iterations}, Epoch:", ascii=True)

        # Training loop
        for epoch in range(1, num_of_epochs+1):
            pbar.update(1)
            # Forward pass on training data
            outputs = nn_model(X_train)
            train_loss = criterion(outputs, y_train) # + nn_model.l2_regularization_loss() # TODO add regularization if needed
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            # If it's the last epoch, simulate car behavior and identify model coefficients
            if (epoch == num_of_epochs):
                pbar.close()
                nn_model.eval()
                v_x, v_y, omega, delta = simulated_data_gen(nn_model, model, avg_vel)   
                C_Pf_identified, C_Pr_identified = solve_pacejka(model, v_x, v_y, omega, delta)

                print(f"C_Pf_identified at Iteration {i}:", C_Pf_identified)
                print(f"C_Pr_identified at Iteration {i}:", C_Pr_identified)
                
                if plot_model:
                    #rclpy.get_logger().warn("Close the plot window (press Q) to continue... ")
                    plot_results(model, v_x, v_y, omega, delta, C_Pf_identified, C_Pr_identified, i)
                    
                # Update model with identified coefficients
                model['C_Pf_model'] = C_Pf_identified
                model['C_Pr_model'] = C_Pr_identified
                
    # Save the trained model with identified coefficients
    model_name = racecar_version +"_pacejka"
    car_model = get_dotdict(model_name)
    car_model.C_Pf = C_Pf_identified
    car_model.C_Pr = C_Pr_identified
    save(car_model)
    save_LUT_name = model_name + "_LUT"
    print("Training complete!")
    
    ''' Generate Look-Up Table (LUT) with the updated model'''

    print("LUT is being generated...")
    LookupGenerator(racecar_version, save_LUT_name).run_generator()
